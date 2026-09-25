"""
Improved Blocking Module for Business Entity Resolution.

Strategy (multi-pass, union of blocks):
1. Country blocking (mandatory filter — never compare across countries)
2. Exact name_base match (high precision, catches trivial matches)
3. N-gram (character trigram) blocking on name_base via TF-IDF + cosine similarity
4. Token overlap blocking on address tokens (sorted first-N tokens)

Each pass generates candidate pairs. We take the UNION of all passes
so recall is maximised, then the downstream matcher filters for precision.
"""

import os
import sys
import csv
import time
import re
from collections import defaultdict
from itertools import islice

# Add src dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from preprocess import clean_text, extract_legal_terms, normalize_address

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def char_ngrams(text, n=3):
    """Generate character n-grams from text."""
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def word_tokens(text):
    """Split text into word tokens."""
    return text.split() if text else []


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def sorted_token_key(text, n_tokens=3):
    """Create a blocking key from the first N sorted tokens."""
    tokens = sorted(text.split())
    return " ".join(tokens[:n_tokens])


# ---------------------------------------------------------------------------
# Data loading (streaming, memory-efficient)
# ---------------------------------------------------------------------------

def load_source_records(filepath, max_rows=None):
    """
    Load source records from TSV, applying preprocessing on the fly.
    Returns a list of dicts with original + cleaned fields.
    """
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break

            country_clean = clean_text(row.get('country', ''))
            name_clean = clean_text(row.get('business_name', ''))
            addr_clean = clean_text(row.get('business_address', ''))
            name_base, name_legal = extract_legal_terms(name_clean)
            addr_norm = normalize_address(addr_clean, country_clean)

            records.append({
                'entity_id': row['entity_id'],
                'country': country_clean,
                'name_base': name_base,
                'name_clean': name_clean,
                'name_legal': name_legal,
                'addr_norm': addr_norm,
            })
    return records


# ---------------------------------------------------------------------------
# Blocking passes
# ---------------------------------------------------------------------------

def build_inverted_index(records, key_fn):
    """
    Build an inverted index: key -> list of record indices.
    key_fn(record) should return a list of keys for that record.
    """
    index = defaultdict(list)
    for idx, rec in enumerate(records):
        for key in key_fn(rec):
            if key:  # skip empty keys
                index[key].append(idx)
    return index


def block_exact_name(s1_records, target_records):
    """
    Pass 1: Exact match on (country, name_base).
    Very fast, very precise.
    """
    print("  [Block 1] Exact name_base + country ...")
    target_index = defaultdict(list)
    for idx, rec in enumerate(target_records):
        key = (rec['country'], rec['name_base'])
        if rec['name_base']:  # skip empty
            target_index[key].append(idx)

    candidates = defaultdict(set)
    for s1_rec in s1_records:
        key = (s1_rec['country'], s1_rec['name_base'])
        for tidx in target_index.get(key, []):
            candidates[s1_rec['entity_id']].add(target_records[tidx]['entity_id'])

    total = sum(len(v) for v in candidates.values())
    print(f"    -> {len(candidates)} S1 entities got {total} candidates")
    return candidates


def block_name_prefix(s1_records, target_records, prefix_len=4):
    """
    Pass 2: Match on (country, first N chars of name_base).
    Catches minor typos after the prefix.
    """
    print(f"  [Block 2] Name prefix({prefix_len}) + country ...")
    target_index = defaultdict(list)
    for idx, rec in enumerate(target_records):
        prefix = rec['name_base'][:prefix_len] if rec['name_base'] else ''
        if prefix:
            key = (rec['country'], prefix)
            target_index[key].append(idx)

    candidates = defaultdict(set)
    for s1_rec in s1_records:
        prefix = s1_rec['name_base'][:prefix_len] if s1_rec['name_base'] else ''
        if prefix:
            key = (s1_rec['country'], prefix)
            for tidx in target_index.get(key, []):
                candidates[s1_rec['entity_id']].add(target_records[tidx]['entity_id'])

    total = sum(len(v) for v in candidates.values())
    print(f"    -> {len(candidates)} S1 entities got {total} candidates")
    return candidates


def block_name_tokens(s1_records, target_records, n_tokens=2):
    """
    Pass 3: Sorted word-token blocking on name_base (within same country).
    Takes the first N sorted non-trivial word tokens as a blocking key.
    This is O(n+m) via inverted index and much faster than trigram scanning.
    """
    print(f"  [Block 3] Name sorted-word-token key (n={n_tokens}) + country ...")

    def make_key(name_base, n):
        tokens = sorted(set(name_base.split()))
        # Skip very short tokens (1 char) that aren't discriminative
        tokens = [t for t in tokens if len(t) > 1]
        return " ".join(tokens[:n])

    target_index = defaultdict(list)
    for idx, rec in enumerate(target_records):
        key_str = make_key(rec['name_base'], n_tokens)
        if key_str.strip():
            key = (rec['country'], key_str)
            target_index[key].append(idx)

    candidates = defaultdict(set)
    for s1_rec in s1_records:
        key_str = make_key(s1_rec['name_base'], n_tokens)
        if key_str.strip():
            key = (s1_rec['country'], key_str)
            for tidx in target_index.get(key, []):
                candidates[s1_rec['entity_id']].add(target_records[tidx]['entity_id'])

    total = sum(len(v) for v in candidates.values())
    print(f"    -> {len(candidates)} S1 entities got {total} candidates")
    return candidates


def block_address_tokens(s1_records, target_records, n_tokens=3):
    """
    Pass 4: Sorted-token blocking on address.
    Takes the first N sorted tokens of the normalized address + country as a blocking key.
    """
    print(f"  [Block 4] Address sorted-token key (n={n_tokens}) + country ...")
    target_index = defaultdict(list)
    for idx, rec in enumerate(target_records):
        key_str = sorted_token_key(rec['addr_norm'], n_tokens)
        if key_str.strip():
            key = (rec['country'], key_str)
            target_index[key].append(idx)

    candidates = defaultdict(set)
    for s1_rec in s1_records:
        key_str = sorted_token_key(s1_rec['addr_norm'], n_tokens)
        if key_str.strip():
            key = (s1_rec['country'], key_str)
            for tidx in target_index.get(key, []):
                candidates[s1_rec['entity_id']].add(target_records[tidx]['entity_id'])

    total = sum(len(v) for v in candidates.values())
    print(f"    -> {len(candidates)} S1 entities got {total} candidates")
    return candidates


# ---------------------------------------------------------------------------
# Main blocking pipeline
# ---------------------------------------------------------------------------

def merge_candidates(*candidate_dicts):
    """Union all candidate dicts into one."""
    merged = defaultdict(set)
    for d in candidate_dicts:
        for s1_id, cand_set in d.items():
            merged[s1_id].update(cand_set)
    return merged


def run_blocking(s1_path, s2_path, s3_path, output_path, s1_max_rows=None):
    """
    Runs the full multi-pass blocking pipeline.
    s1_max_rows: only sample S1 (for quick testing). S2 and S3 are always loaded fully
                 because true matches can be anywhere.
    """
    print("=" * 60)
    print("BLOCKING PIPELINE")
    print("=" * 60)

    t0 = time.time()

    print("\nLoading & preprocessing Source 1 ...")
    s1_records = load_source_records(s1_path, max_rows=s1_max_rows)
    print(f"  Loaded {len(s1_records)} S1 records")

    print("Loading & preprocessing Source 2 (full) ...")
    s2_records = load_source_records(s2_path)
    print(f"  Loaded {len(s2_records)} S2 records")

    print("Loading & preprocessing Source 3 (full) ...")
    s3_records = load_source_records(s3_path)
    print(f"  Loaded {len(s3_records)} S3 records")

    target_records = s2_records + s3_records
    print(f"\nTotal target records: {len(target_records)}")

    print("\nRunning blocking passes ...")
    c1 = block_exact_name(s1_records, target_records)
    c2 = block_name_prefix(s1_records, target_records, prefix_len=4)
    c3 = block_name_tokens(s1_records, target_records)
    c4 = block_address_tokens(s1_records, target_records, n_tokens=3)

    print("\nMerging all candidate sets (union) ...")
    all_candidates = merge_candidates(c1, c2, c3, c4)

    # Ensure every S1 entity has a row (even if empty)
    s1_ids = {rec['entity_id'] for rec in s1_records}
    for s1_id in s1_ids:
        if s1_id not in all_candidates:
            all_candidates[s1_id] = set()

    total_pairs = sum(len(v) for v in all_candidates.values())
    entities_with_cands = sum(1 for v in all_candidates.values() if v)
    print(f"\nFinal: {len(all_candidates)} S1 entities, "
          f"{entities_with_cands} with candidates, {total_pairs} total pairs")
    if s1_ids:
        print(f"Avg candidates per S1 entity: {total_pairs / len(s1_ids):.1f}")

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for s1_id in sorted(all_candidates.keys()):
            cand_str = ",".join(sorted(all_candidates[s1_id]))
            writer.writerow([s1_id, cand_str])

    elapsed = time.time() - t0
    print(f"\nBlocking completed in {elapsed:.1f}s. Output: {output_path}")
    return all_candidates, s1_ids


# ---------------------------------------------------------------------------
# Recall evaluation against ground truth
# ---------------------------------------------------------------------------

def evaluate_blocking_recall(candidates, gt_path, s1_ids=None):
    """
    Measures blocking recall: what fraction of true matches appear
    in the candidate set.
    s1_ids: if provided, only evaluate S1 entities in this set.
    """
    print("\n" + "=" * 60)
    print("BLOCKING RECALL EVALUATION")
    print("=" * 60)

    total_true = 0
    total_found = 0
    total_entities = 0
    perfect_recall = 0

    with open(gt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            s1_id = row['source1_entity_id']
            
            # Skip S1 entities not in our sample
            if s1_ids and s1_id not in s1_ids:
                continue
                
            matched_str = row.get('matched_entity_ids', '')
            if not matched_str or not matched_str.strip():
                continue  # singleton, skip for recall calc

            true_matches = set(matched_str.split(','))
            cand_set = candidates.get(s1_id, set())

            found = len(true_matches & cand_set)
            total_true += len(true_matches)
            total_found += found
            total_entities += 1
            if found == len(true_matches):
                perfect_recall += 1

    if total_true > 0:
        recall = total_found / total_true
        print(f"  True match pairs evaluated: {total_true}")
        print(f"  Found in candidates:        {total_found}")
        print(f"  Blocking Recall:            {recall:.4f} ({recall*100:.2f}%)")
        print(f"  Entities with perfect recall: {perfect_recall}/{total_entities} "
              f"({perfect_recall/total_entities*100:.1f}%)")
    else:
        print("  No true matches found in ground truth to evaluate.")

    return total_found / total_true if total_true > 0 else 0.0


if __name__ == "__main__":
    # Quick test: sample 1000 S1 entities, but load ALL of S2/S3
    s1_sample = 1000  # Set to None for full run

    candidates, s1_ids = run_blocking(
        s1_path=config.TRAIN_S1,
        s2_path=config.TRAIN_S2,
        s3_path=config.TRAIN_S3,
        output_path=config.CANDIDATE_PAIRS,
        s1_max_rows=s1_sample
    )

    evaluate_blocking_recall(candidates, config.TRAIN_GT, s1_ids=s1_ids)
