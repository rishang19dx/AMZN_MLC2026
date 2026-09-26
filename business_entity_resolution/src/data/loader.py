"""
Data loading and preprocessing module for Business Entity Resolution.

Handles:
- TSV reading with robust handling of missing values
- Applying normalization and feature extraction
- Ground truth parsing
- Entity-level train/validation split with strictly NO leakage
- Configurable reproducible subset sampling
"""

import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from business_entity_resolution.src.normalization.normalizer import (
    extract_first_token,
    extract_name_prefix,
    extract_postal_code,
    format_record_text,
    normalize_address,
    normalize_country,
    normalize_name,
)


def load_source_file(
    path: str,
    nrows: Optional[int] = None,
    keep_default_na: bool = False
) -> pd.DataFrame:
    """
    Load a source TSV file (Source 1, Source 2, or Source 3).
    Preserves raw fields while computing normalized representations.
    
    Columns added:
    - name_norm: cleaned business name
    - address_norm: cleaned address
    - country_norm: standardized open-set country string
    - postal_code: extracted 5- or 6-digit postal code
    - first_token: first significant token of name
    - name_prefix: first 4 characters of name
    - combined_text: structured representation for neural encoders
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found at: {path}")

    # Read tab-separated file with string dtype to prevent casting entity_ids or zip codes
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=nrows,
        dtype=str,
        keep_default_na=keep_default_na,
        na_values=[]
    )
    
    # Fill NaN values with empty string
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    # Ensure entity_id is present
    if "entity_id" not in df.columns:
        raise ValueError(f"Required column 'entity_id' missing from {path}")

    # Compute normalized fields (vectorized or list comp for speed)
    df["name_norm"] = [normalize_name(x) for x in df["business_name"]]
    df["address_norm"] = [normalize_address(x) for x in df["business_address"]]
    df["country_norm"] = [normalize_country(x) for x in df["country"]]
    
    # Extract blocking keys
    df["postal_code"] = [
        extract_postal_code(addr, ctry)
        for addr, ctry in zip(df["business_address"], df["country_norm"])
    ]
    df["first_token"] = [extract_first_token(n) for n in df["name_norm"]]
    df["name_prefix"] = [extract_name_prefix(n, length=4) for n in df["name_norm"]]
    
    # Combined formatted string for embeddings
    df["combined_text"] = [
        format_record_text(n, a, c, use_normalized=True)
        for n, a, c in zip(df["name_norm"], df["address_norm"], df["country_norm"])
    ]

    return df


def load_ground_truth(
    path: str,
    nrows: Optional[int] = None
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Load ground truth matching relationships.
    
    Returns:
    - gt_dict: mapping {source1_entity_id: [matched_entity_id_1, ...]}
    - positive_pairs: list of tuples [(s1_id, match_id), ...]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found at: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        nrows=nrows,
        dtype=str,
        keep_default_na=False,
        na_values=[]
    )
    
    if "source1_entity_id" not in df.columns or "matched_entity_ids" not in df.columns:
        raise ValueError(f"Ground truth must have 'source1_entity_id' and 'matched_entity_ids' columns.")

    gt_dict: Dict[str, List[str]] = {}
    positive_pairs: List[Tuple[str, str]] = []

    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"]).strip()
        if matched_str and matched_str != "nan":
            matches = [m.strip() for m in matched_str.split(",") if m.strip()]
        else:
            matches = []
        gt_dict[s1] = matches
        for m in matches:
            positive_pairs.append((s1, m))

    return gt_dict, positive_pairs


def create_entity_split(
    s1_df: pd.DataFrame,
    gt_dict: Dict[str, List[str]],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Create an entity-level train/validation split with STRICTLY NO LEAKAGE.
    
    If S1-X is assigned to validation, all its ground truth matches belong strictly
    to validation evaluation, and none of its positive pairs are seen during training.
    """
    rng = np.random.RandomState(seed)
    s1_ids = list(s1_df["entity_id"])
    rng.shuffle(s1_ids)

    n_val = int(len(s1_ids) * val_ratio)
    val_s1_set = set(s1_ids[:n_val])
    train_s1_set = set(s1_ids[n_val:])

    train_s1_df = s1_df[s1_df["entity_id"].isin(train_s1_set)].copy().reset_index(drop=True)
    val_s1_df = s1_df[s1_df["entity_id"].isin(val_s1_set)].copy().reset_index(drop=True)

    train_gt: Dict[str, List[str]] = {s1: gt_dict.get(s1, []) for s1 in train_s1_set}
    val_gt: Dict[str, List[str]] = {s1: gt_dict.get(s1, []) for s1 in val_s1_set}

    # Verify no overlap
    assert len(train_s1_set.intersection(val_s1_set)) == 0, "Train and Val S1 sets must be disjoint!"

    return train_s1_df, val_s1_df, train_gt, val_gt


def sample_records(
    df: pd.DataFrame,
    subset_ratio: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Subsample dataframe reproducibly based on subset_ratio.
    """
    if subset_ratio >= 1.0:
        return df
    
    n_samples = max(1, int(len(df) * subset_ratio))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(df))[:n_samples]
    return df.iloc[indices].copy().reset_index(drop=True)
