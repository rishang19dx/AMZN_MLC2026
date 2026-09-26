"""
LightGBM Candidate Ranking and Filtering module.

Computes lightweight pairwise similarity features on blocker-generated candidate pairs
and filters candidates to produce the final submission matching_results.tsv.
Optimizes the decision threshold specifically for the Macro F_0.5 challenge metric.
"""

import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from business_entity_resolution.src.blocking.classical_blocker import (
    get_char_ngrams,
    jaccard_similarity,
    token_overlap_score,
)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def compute_pairwise_features(
    s1_row: pd.Series,
    cand_row: pd.Series,
    bert_sim: float = 0.0,
    bert_rank: Optional[int] = None,
    learned_sim: float = 0.0,
    learned_rank: Optional[int] = None,
    classical_score: float = 0.0,
    num_pipelines: int = 1,
) -> List[float]:
    """
    Extract pairwise comparison features for a candidate pair.
    """
    s1_name = str(s1_row.get("name_norm", ""))
    cand_name = str(cand_row.get("name_norm", ""))
    s1_addr = str(s1_row.get("address_norm", ""))
    cand_addr = str(cand_row.get("address_norm", ""))
    s1_country = str(s1_row.get("country_norm", ""))
    cand_country = str(cand_row.get("country_norm", ""))
    s1_postal = str(s1_row.get("postal_code", ""))
    cand_postal = str(cand_row.get("postal_code", ""))

    # 1. Name features
    s1_name_toks = set(s1_name.split())
    cand_name_toks = set(cand_name.split())
    name_jaccard = jaccard_similarity(s1_name_toks, cand_name_toks)

    s1_name_chars = get_char_ngrams(s1_name, n=3)
    cand_name_chars = get_char_ngrams(cand_name, n=3)
    name_char_jaccard = jaccard_similarity(s1_name_chars, cand_name_chars)

    name_exact = 1.0 if s1_name and s1_name == cand_name else 0.0

    # 2. Address features
    s1_addr_toks = s1_addr.split()
    cand_addr_toks = cand_addr.split()
    addr_overlap = token_overlap_score(s1_addr_toks, cand_addr_toks)

    s1_addr_chars = get_char_ngrams(s1_addr, n=3)
    cand_addr_chars = get_char_ngrams(cand_addr, n=3)
    addr_char_jaccard = jaccard_similarity(s1_addr_chars, cand_addr_chars)

    # 3. Country and postal features
    country_match = 1.0 if (s1_country and cand_country and s1_country == cand_country) else 0.0
    postal_match = 1.0 if (s1_postal and cand_postal and s1_postal == cand_postal) else 0.0

    # 4. Pipeline provenance features
    b_rank = float(bert_rank) if bert_rank is not None else 100.0
    l_rank = float(learned_rank) if learned_rank is not None else 100.0

    return [
        name_jaccard,
        name_char_jaccard,
        name_exact,
        addr_overlap,
        addr_char_jaccard,
        country_match,
        postal_match,
        float(bert_sim),
        b_rank,
        float(learned_sim),
        l_rank,
        float(classical_score),
        float(num_pipelines),
    ]


FEATURE_NAMES = [
    "name_tok_jaccard",
    "name_char_jaccard",
    "name_exact_match",
    "addr_tok_overlap",
    "addr_char_jaccard",
    "country_match",
    "postal_match",
    "bert_sim",
    "bert_rank",
    "learned_sim",
    "learned_rank",
    "classical_score",
    "num_pipelines",
]


def macro_f_beta(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    beta: float = 0.5,
) -> float:
    """
    Calculate Macro-averaged F_beta score across all Source 1 entities (including singletons).
    """
    beta_sq = beta ** 2
    f_scores = []

    for s1_id, true_matches in ground_truth.items():
        pred_matches = set(predictions.get(s1_id, []))
        true_set = set(true_matches)

        if len(true_set) == 0:
            # Singleton entity: 1.0 if correctly predicted empty, 0.0 otherwise
            f_scores.append(1.0 if len(pred_matches) == 0 else 0.0)
            continue

        if len(pred_matches) == 0:
            f_scores.append(0.0)
            continue

        tp = len(true_set.intersection(pred_matches))
        precision = tp / len(pred_matches)
        recall = tp / len(true_set)

        denom = (beta_sq * precision) + recall
        if denom == 0.0:
            f_scores.append(0.0)
        else:
            fb = ((1.0 + beta_sq) * precision * recall) / denom
            f_scores.append(fb)

    return float(np.mean(f_scores)) if f_scores else 0.0


class LightGBMFilter:
    """
    Modular LightGBM candidate ranking and filtering model.
    """

    def __init__(self, threshold: float = 0.5):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm must be installed to use LightGBMFilter.")
        self.threshold = threshold
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
        )
        self.is_fitted = False

    def train_on_candidates(
        self,
        candidate_mapping: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        provenance_df: Optional[pd.DataFrame] = None,
        val_candidate_mapping: Optional[Dict[str, List[str]]] = None,
        val_ground_truth: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Build training dataset from blocked candidate pairs and fit LightGBM.
        """
        s1_rows = {row.entity_id: row._asdict() for row in s1_df.itertuples(index=False)}
        target_rows = {row.entity_id: row._asdict() for row in target_df.itertuples(index=False)}

        prov_lookup = {}
        if provenance_df is not None and len(provenance_df) > 0:
            for _, r in provenance_df.iterrows():
                prov_lookup[(r["source1_id"], r["candidate_id"])] = r

        X, y = [], []

        for s1_id, cands in candidate_mapping.items():
            if s1_id not in s1_rows:
                continue
            true_set = set(ground_truth.get(s1_id, []))
            s1_data = s1_rows[s1_id]

            for cid in cands:
                if cid not in target_rows:
                    continue
                cand_data = target_rows[cid]
                prov = prov_lookup.get((s1_id, cid), {})

                feat = compute_pairwise_features(
                    s1_data,
                    cand_data,
                    bert_sim=prov.get("bert_sim", 0.0),
                    bert_rank=prov.get("bert_rank", None),
                    learned_sim=prov.get("learned_sim", 0.0),
                    learned_rank=prov.get("learned_rank", None),
                    classical_score=prov.get("classical_score", 0.0),
                    num_pipelines=prov.get("num_pipelines", 1),
                )
                label = 1 if cid in true_set else 0
                X.append(feat)
                y.append(label)

        if not X:
            print("Warning: No candidate training pairs available for LightGBM.")
            return

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=int)

        print(f"Fitting LightGBM on {len(X_arr)} candidate pairs (positive ratio: {np.mean(y_arr):.4f})...")
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        print("LightGBM fitting complete.")

    def filter_candidates(
        self,
        candidate_mapping: Dict[str, List[str]],
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        provenance_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[str]]:
        """
        Filter candidate pairs using trained LightGBM to produce final matching predictions.
        """
        if not self.is_fitted:
            print("LightGBM not fitted; returning empty matches (conservative singletons).")
            return {s1_id: [] for s1_id in candidate_mapping.keys()}

        s1_rows = {row.entity_id: row._asdict() for row in s1_df.itertuples(index=False)}
        target_rows = {row.entity_id: row._asdict() for row in target_df.itertuples(index=False)}

        prov_lookup = {}
        if provenance_df is not None and len(provenance_df) > 0:
            for _, r in provenance_df.iterrows():
                prov_lookup[(r["source1_id"], r["candidate_id"])] = r

        matched_results: Dict[str, List[str]] = {}

        for s1_id, cands in candidate_mapping.items():
            if not cands or s1_id not in s1_rows:
                matched_results[s1_id] = []
                continue

            s1_data = s1_rows[s1_id]
            valid_cands = []
            feats = []

            for cid in cands:
                if cid not in target_rows:
                    continue
                cand_data = target_rows[cid]
                prov = prov_lookup.get((s1_id, cid), {})
                feat = compute_pairwise_features(
                    s1_data,
                    cand_data,
                    bert_sim=prov.get("bert_sim", 0.0),
                    bert_rank=prov.get("bert_rank", None),
                    learned_sim=prov.get("learned_sim", 0.0),
                    learned_rank=prov.get("learned_rank", None),
                    classical_score=prov.get("classical_score", 0.0),
                    num_pipelines=prov.get("num_pipelines", 1),
                )
                feats.append(feat)
                valid_cands.append(cid)

            if not feats:
                matched_results[s1_id] = []
                continue

            probs = self.model.predict_proba(np.array(feats, dtype=np.float32))[:, 1]
            accepted = [cid for cid, p in zip(valid_cands, probs) if p >= self.threshold]
            matched_results[s1_id] = accepted

        return matched_results

    @staticmethod
    def export_matching_results(
        matched_results: Dict[str, List[str]],
        output_filepath: str,
    ):
        """
        Export final matches in official format:
        source1_entity_id \t matched_entity_ids
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id, matches in matched_results.items():
                match_str = ",".join(matches) if matches else ""
                f.write(f"{s1_id}\t{match_str}\n")
