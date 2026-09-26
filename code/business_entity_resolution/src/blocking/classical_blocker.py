"""
Classical non-neural blocking module.

Generates candidate pairs using multi-key inverted indexing and lightweight string
similarity scoring (token Jaccard, character n-grams, postal code matching, address overlap).
Avoids full Cartesian product by restricting comparisons to key-indexed candidate buckets.
"""

from collections import defaultdict
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate character n-grams from text."""
    if not text:
        return set()
    cleaned = f"^{text.strip()}$"
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection) / float(union) if union > 0 else 0.0


def token_overlap_score(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute containment overlap: |A ∩ B| / min(|A|, |B|)."""
    if not tokens_a or not tokens_b:
        return 0.0
    sa, sb = set(tokens_a), set(tokens_b)
    intersection = len(sa.intersection(sb))
    min_size = min(len(sa), len(sb))
    return float(intersection) / float(min_size) if min_size > 0 else 0.0


def extract_address_signature(address: str) -> str:
    """Extract street number + first significant word of address."""
    if not address:
        return ""
    tokens = address.split()
    if not tokens:
        return ""
    # Look for leading numbers (e.g., '85') and first word
    num = ""
    word = ""
    for tok in tokens:
        if tok.isdigit() and not num:
            num = tok
        elif len(tok) > 2 and not word:
            word = tok
        if num and word:
            break
    if num and word:
        return f"{num}_{word}"
    return word if word else (num if num else "")


class ClassicalBlocker:
    """
    Multi-key classical blocker.
    
    Generates blocking keys:
    1. (country, first_token)
    2. (country, name_prefix)
    3. (country, postal_code)
    4. (country, addr_signature)
    
    Scoring:
    Combines name Jaccard, char 3-gram similarity, address overlap, and postal matching.
    """

    def __init__(
        self,
        top_k_source2: int = 30,
        top_k_source3: int = 30,
        max_bucket_size: int = 2000,
        min_score_threshold: float = 0.15,
    ):
        self.top_k_source2 = top_k_source2
        self.top_k_source3 = top_k_source3
        self.max_bucket_size = max_bucket_size
        self.min_score_threshold = min_score_threshold

        # Inverted indices: key -> list of (record_id, is_source2)
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        
        # Target records storage
        self.target_ids: List[str] = []
        self.target_is_s2: List[bool] = []
        self.target_name_tokens: List[Set[str]] = []
        self.target_name_char_ngrams: List[Set[str]] = []
        self.target_addr_tokens: List[List[str]] = []
        self.target_postals: List[str] = []
        self.target_countries: List[str] = []

    def _generate_keys(
        self,
        country: str,
        first_token: str,
        name_prefix: str,
        postal_code: str,
        address: str,
    ) -> List[str]:
        """Generate multiple blocking keys for a record."""
        keys = []
        c = country.strip() if country else "UNK"

        # Key 1: Country + first significant token
        if first_token and len(first_token) >= 2:
            keys.append(f"K1:{c}:{first_token}")

        # Key 2: Country + 4-char name prefix
        if name_prefix and len(name_prefix) >= 3:
            keys.append(f"K2:{c}:{name_prefix}")

        # Key 3: Country + postal code
        if postal_code and len(postal_code) >= 4:
            keys.append(f"K3:{c}:{postal_code}")

        # Key 4: Country + address signature (house num + first street word)
        addr_sig = extract_address_signature(address)
        if addr_sig and len(addr_sig) >= 3:
            keys.append(f"K4:{c}:{addr_sig}")

        return keys

    def fit_targets(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> "ClassicalBlocker":
        """
        Build inverted index over Source 2 and Source 3 records.
        """
        self.inverted_index.clear()
        self.target_ids.clear()
        self.target_is_s2.clear()
        self.target_name_tokens.clear()
        self.target_name_char_ngrams.clear()
        self.target_addr_tokens.clear()
        self.target_postals.clear()
        self.target_countries.clear()

        # Combine S2 and S3 with source tag
        frames = []
        if len(s2_df) > 0:
            df2 = s2_df.copy()
            df2["_is_s2"] = True
            frames.append(df2)
        if len(s3_df) > 0:
            df3 = s3_df.copy()
            df3["_is_s3"] = False
            df3["_is_s2"] = False
            frames.append(df3)

        if not frames:
            return self

        combined = pd.concat(frames, ignore_index=True)

        for idx, row in enumerate(combined.itertuples(index=False)):
            eid = row.entity_id
            is_s2 = getattr(row, "_is_s2", False)
            c = getattr(row, "country_norm", "")
            ft = getattr(row, "first_token", "")
            npfx = getattr(row, "name_prefix", "")
            pc = getattr(row, "postal_code", "")
            addr = getattr(row, "address_norm", "")
            name = getattr(row, "name_norm", "")

            self.target_ids.append(eid)
            self.target_is_s2.append(is_s2)
            self.target_countries.append(c)
            self.target_postals.append(pc)
            
            # Precompute token sets for fast scoring
            name_toks = set(name.split()) if name else set()
            self.target_name_tokens.append(name_toks)
            self.target_name_char_ngrams.append(get_char_ngrams(name, n=3))
            self.target_addr_tokens.append(addr.split() if addr else [])

            keys = self._generate_keys(c, ft, npfx, pc, addr)
            for k in keys:
                self.inverted_index[k].append(idx)

        # Cap overly large buckets to prevent degenerate keys
        for k in list(self.inverted_index.keys()):
            if len(self.inverted_index[k]) > self.max_bucket_size:
                # Truncate overly frequent bucket
                self.inverted_index[k] = self.inverted_index[k][: self.max_bucket_size]

        return self

    def score_pair(
        self,
        s1_name_tokens: Set[str],
        s1_name_chars: Set[str],
        s1_addr_tokens: List[str],
        s1_postal: str,
        s1_country: str,
        target_idx: int,
    ) -> float:
        """Calculate composite string similarity between S1 and a target candidate."""
        t_country = self.target_countries[target_idx]
        if s1_country and t_country and s1_country != t_country:
            return 0.0

        t_name_toks = self.target_name_tokens[target_idx]
        t_name_chars = self.target_name_char_ngrams[target_idx]
        t_addr_toks = self.target_addr_tokens[target_idx]
        t_postal = self.target_postals[target_idx]

        # Name similarities
        name_tok_jaccard = jaccard_similarity(s1_name_tokens, t_name_toks)
        name_char_jaccard = jaccard_similarity(s1_name_chars, t_name_chars)
        name_score = max(name_tok_jaccard, name_char_jaccard)

        # Address similarity
        addr_score = token_overlap_score(s1_addr_tokens, t_addr_toks)

        # Postal match
        postal_score = 1.0 if (s1_postal and t_postal and s1_postal == t_postal) else 0.0

        # When address is missing in one record, rely on name
        if not s1_addr_tokens or not t_addr_toks:
            return float(name_score)

        # When name is heavily corrupted but address matches
        if addr_score >= 0.7:
            return float(0.4 * name_score + 0.5 * addr_score + 0.1 * postal_score)

        # Standard weighted score
        return float(0.55 * name_score + 0.35 * addr_score + 0.10 * postal_score)

    def retrieve_candidates(
        self, s1_df: pd.DataFrame
    ) -> Dict[str, List[Tuple[str, int, float, str]]]:
        """
        Retrieve classical candidate pairs for Source 1 records.
        
        Returns:
            Dict mapping s1_id -> list of (candidate_id, rank, score, matched_keys)
        """
        results: Dict[str, List[Tuple[str, int, float, str]]] = {}

        for row in s1_df.itertuples(index=False):
            s1_id = row.entity_id
            c = getattr(row, "country_norm", "")
            ft = getattr(row, "first_token", "")
            npfx = getattr(row, "name_prefix", "")
            pc = getattr(row, "postal_code", "")
            addr = getattr(row, "address_norm", "")
            name = getattr(row, "name_norm", "")

            keys = self._generate_keys(c, ft, npfx, pc, addr)
            s1_name_tokens = set(name.split()) if name else set()
            s1_name_chars = get_char_ngrams(name, n=3)
            s1_addr_tokens = addr.split() if addr else []

            # Gather candidate target indices and matched keys
            cand_indices: Dict[int, List[str]] = defaultdict(list)
            for k in keys:
                matched_idxs = self.inverted_index.get(k, [])
                for idx in matched_idxs:
                    cand_indices[idx].append(k.split(":")[0])  # store key type, e.g. K1, K2

            if not cand_indices:
                results[s1_id] = []
                continue

            # Score candidates separately for S2 and S3
            scored_s2: List[Tuple[str, float, str]] = []
            scored_s3: List[Tuple[str, float, str]] = []

            for idx, matched_keys in cand_indices.items():
                score = self.score_pair(
                    s1_name_tokens,
                    s1_name_chars,
                    s1_addr_tokens,
                    pc,
                    c,
                    idx,
                )
                if score >= self.min_score_threshold:
                    cand_id = self.target_ids[idx]
                    is_s2 = self.target_is_s2[idx]
                    key_tag = "+".join(sorted(set(matched_keys)))
                    if is_s2:
                        scored_s2.append((cand_id, score, key_tag))
                    else:
                        scored_s3.append((cand_id, score, key_tag))

            # Sort descending by score
            scored_s2.sort(key=lambda x: x[1], reverse=True)
            scored_s3.sort(key=lambda x: x[1], reverse=True)

            top_s2 = scored_s2[: self.top_k_source2]
            top_s3 = scored_s3[: self.top_k_source3]

            combined_ranked: List[Tuple[str, int, float, str]] = []
            for rank, (cid, sc, kt) in enumerate(top_s2, start=1):
                combined_ranked.append((cid, rank, sc, kt))
            for rank, (cid, sc, kt) in enumerate(top_s3, start=1):
                combined_ranked.append((cid, rank, sc, kt))

            results[s1_id] = combined_ranked

        return results
