"""
Comprehensive Unit and Integration Tests for Business Entity Resolution Multi-Pipeline Blocking System.

Covers all 16 required test suites:
1. TSV loading
2. Normalization
3. Positive-pair construction & leak-free splitting
4. Negative-pair construction
5. Embedding generation & normalization
6. ANN retrieval
7. Classical blocking
8. Candidate union
9. Deduplication
10. Output format
11. Exactly one output row per S1 entity
12. No S1 IDs as candidates (no self-matches)
13. No nonexistent S2/S3 IDs
14. Deterministic output
15. Blocking evaluation & penalty metrics
16. LightGBM feature generation and filtering
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.src.normalization.normalizer import (
    normalize_name,
    normalize_address,
    normalize_country,
    extract_postal_code,
    extract_first_token,
    extract_name_prefix,
    format_record_text,
)
from business_entity_resolution.src.data.loader import (
    load_source_file,
    load_ground_truth,
    create_entity_split,
    sample_records,
)
from business_entity_resolution.src.blocking.ann_index import VectorIndex
from business_entity_resolution.src.blocking.classical_blocker import (
    ClassicalBlocker,
    get_char_ngrams,
    jaccard_similarity,
    token_overlap_score,
)
from business_entity_resolution.src.blocking.candidate_union import (
    CandidateUnion,
    compute_candidate_priority,
)
from business_entity_resolution.src.evaluation.blocking_metrics import (
    compute_blocking_metrics,
    format_metrics_report,
)
from business_entity_resolution.src.lightgbm_filter import (
    compute_pairwise_features,
    macro_f_beta,
)


@pytest.fixture
def mock_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock S1
        s1_data = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" \
                  "S1-1\tAcme Corp Inc\t100 Main Street, Springfield, 62701\tUS\n" \
                  "S1-2\tTata Consultancy Services Ltd\tBandra Kurla Complex, Mumbai, 400051\tIndia\n" \
                  "S1-3\tSociété Générale SAS\t29 Boulevard Haussmann, Paris, 75009\tFrance\n" \
                  "S1-4\tLonely Singleton LLC\t99 Nowhere Road\tUS\n"
        with open(os.path.join(tmpdir, "s1.tsv"), "w", encoding="utf-8") as f:
            f.write(s1_data)

        # Create mock S2
        s2_data = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" \
                  "S2-101\tAcme Corporation\t100 Main St, Springfield\tUS\n" \
                  "S2-102\tTata Consultancy\tBKC, Mumbai\tIndia\n" \
                  "S2-103\tUnrelated Corp\t1 Random Lane\tUS\n"
        with open(os.path.join(tmpdir, "s2.tsv"), "w", encoding="utf-8") as f:
            f.write(s2_data)

        # Create mock S3
        s3_data = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" \
                  "S3-201\tAcme Inc\t100 Main Street\tUS\n" \
                  "S3-202\tSoc Gen\t29 Blvd Haussmann, Paris 75009\tFrance\n" \
                  "S3-203\tAnother Unrelated\t404 Not Found St\tIndia\n"
        with open(os.path.join(tmpdir, "s3.tsv"), "w", encoding="utf-8") as f:
            f.write(s3_data)

        # Create mock Ground Truth
        gt_data = "source1_entity_id\tmatched_entity_ids\n" \
                  "S1-1\tS2-101,S3-201\n" \
                  "S1-2\tS2-102\n" \
                  "S1-3\tS3-202\n" \
                  "S1-4\t\n"
        with open(os.path.join(tmpdir, "gt.tsv"), "w", encoding="utf-8") as f:
            f.write(gt_data)

        yield tmpdir


# 1. TSV Loading Test
def test_tsv_loading(mock_data_dir):
    s1_path = os.path.join(mock_data_dir, "s1.tsv")
    df = load_source_file(s1_path)
    assert len(df) == 4
    assert "entity_id" in df.columns
    assert "name_norm" in df.columns
    assert "address_norm" in df.columns
    assert "country_norm" in df.columns
    assert "postal_code" in df.columns
    assert df.loc[0, "name_norm"] == "acme corp inc"
    assert df.loc[0, "postal_code"] == "62701"


# 2. Normalization Test
def test_normalization():
    # Names
    assert normalize_name("Acme & Sons Private Limited") == "acme and sons pvt ltd"
    assert normalize_name("Foo Bar, Inc.") == "foo bar inc"
    assert normalize_name("Café Müller GmbH") == "cafe muller gmbh"

    # Addresses
    assert normalize_address("123 Main Road, Suite #4") == "123 main rd ste 4"
    assert normalize_address("Opposite SBI ATM, 5th Cross") == "opp sbi atm 5th cross"

    # Countries (open set preservation)
    assert normalize_country("United States of America") == "US"
    assert normalize_country("Republic of India") == "INDIA"
    assert normalize_country("France") == "FRANCE"
    assert normalize_country("Germany") == "GERMANY"
    assert normalize_country("Brazil") == "BRAZIL"

    # Postal codes
    assert extract_postal_code("Bangalore, 560001", "INDIA") == "560001"
    assert extract_postal_code("Beverly Hills, CA 90210-1234", "US") == "90210"
    assert extract_postal_code("Paris, 75008", "FRANCE") == "75008"

    # First token & prefix
    assert extract_first_token("the acme corp") == "acme"
    assert extract_name_prefix("acme corp", length=4) == "acme"


# 3. Positive-pair construction & Leak-free splitting
def test_positive_pair_construction(mock_data_dir):
    gt_path = os.path.join(mock_data_dir, "gt.tsv")
    s1_path = os.path.join(mock_data_dir, "s1.tsv")
    gt_dict, pos_pairs = load_ground_truth(gt_path)

    assert len(gt_dict) == 4
    assert len(pos_pairs) == 4
    assert ("S1-1", "S2-101") in pos_pairs
    assert ("S1-1", "S3-201") in pos_pairs
    assert ("S1-2", "S2-102") in pos_pairs
    assert ("S1-3", "S3-202") in pos_pairs

    s1_df = load_source_file(s1_path)
    train_df, val_df, train_gt, val_gt = create_entity_split(s1_df, gt_dict, val_ratio=0.5, seed=42)

    # Strictly disjoint S1 entities
    train_ids = set(train_df["entity_id"])
    val_ids = set(val_df["entity_id"])
    assert len(train_ids.intersection(val_ids)) == 0

    # Strictly disjoint positive matches (no leakage)
    train_matches = {m for s1 in train_ids for m in train_gt.get(s1, [])}
    val_matches = {m for s1 in val_ids for m in val_gt.get(s1, [])}
    assert len(train_matches.intersection(val_matches)) == 0


# 4. Negative-pair construction test
def test_negative_pair_construction():
    s1_id = "S1-10"
    true_matches = ["S2-101", "S3-201"]
    candidate_pool = ["S2-101", "S2-999", "S3-201", "S3-888"]
    
    # Non-matches only
    negatives = [c for c in candidate_pool if c not in set(true_matches) and not c.startswith("S1-")]
    assert "S2-101" not in negatives
    assert "S3-201" not in negatives
    assert "S2-999" in negatives
    assert "S3-888" in negatives


# 5. Embedding generation & Normalization test
def test_embedding_generation():
    dim = 64
    rng = np.random.RandomState(42)
    raw_vecs = rng.randn(10, dim)
    index = VectorIndex(dim=dim, use_faiss=False, device="cpu")
    ids = [f"E-{i}" for i in range(10)]
    index.build(raw_vecs, ids)

    # Check normalization
    norms = np.linalg.norm(index.embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# 6. ANN Retrieval test
def test_ann_retrieval():
    dim = 32
    target_vecs = np.zeros((5, dim), dtype=np.float32)
    # Unit vectors on distinct axes
    for i in range(5):
        target_vecs[i, i] = 1.0

    target_ids = [f"T-{i}" for i in range(5)]
    index = VectorIndex(dim=dim, use_faiss=False, device="cpu")
    index.build(target_vecs, target_ids)

    query = np.zeros((1, dim), dtype=np.float32)
    query[0, 2] = 1.0  # Exact match for T-2

    scores, idxs = index.search(query, top_k=3)
    assert len(scores[0]) == 3
    assert index.ids[idxs[0][0]] == "T-2"
    assert pytest.approx(scores[0][0], 1e-4) == 1.0


# 7. Classical Blocking test
def test_classical_blocking(mock_data_dir):
    s1 = load_source_file(os.path.join(mock_data_dir, "s1.tsv"))
    s2 = load_source_file(os.path.join(mock_data_dir, "s2.tsv"))
    s3 = load_source_file(os.path.join(mock_data_dir, "s3.tsv"))

    blocker = ClassicalBlocker(top_k_source2=10, top_k_source3=10, min_score_threshold=0.1)
    blocker.fit_targets(s2, s3)
    results = blocker.retrieve_candidates(s1)

    assert "S1-1" in results
    cand_ids = [cid for cid, _, _, _ in results["S1-1"]]
    # Should catch S2-101 and S3-201
    assert "S2-101" in cand_ids
    assert "S3-201" in cand_ids
    assert "S2-103" not in cand_ids  # Unrelated should not rank high


# 8. Candidate Union test
def test_candidate_union():
    union = CandidateUnion(max_candidates_per_entity=50)
    all_s1 = ["S1-1", "S1-2"]

    bert = {"S1-1": [("S2-101", 1, 0.95), ("S3-201", 2, 0.85)]}
    learned = {"S1-1": [("S2-101", 1, 0.92), ("S3-202", 2, 0.70)]}
    classical = {"S1-1": [("S2-101", 1, 0.90, "K1"), ("S2-103", 2, 0.40, "K2")]}

    final, debug_df = union.merge_candidates(
        all_s1_ids=all_s1,
        bert_candidates=bert,
        learned_candidates=learned,
        classical_candidates=classical,
    )

    s1_1_cands = final["S1-1"]
    # Union must contain candidates from all three pipelines
    assert "S2-101" in s1_1_cands  # In all 3
    assert "S3-201" in s1_1_cands  # In BERT
    assert "S3-202" in s1_1_cands  # In Learned
    assert "S2-103" in s1_1_cands  # In Classical

    # S2-101 found in 3 pipelines should be highest priority (first element)
    assert s1_1_cands[0] == "S2-101"

    # Singleton S1-2 must exist and be empty
    assert "S1-2" in final
    assert final["S1-2"] == []


# 9. Deduplication test
def test_deduplication():
    union = CandidateUnion()
    all_s1 = ["S1-A"]
    bert = {"S1-A": [("S2-X", 1, 0.9), ("S2-Y", 2, 0.8)]}
    learned = {"S1-A": [("S2-X", 1, 0.95)]}
    classical = {"S1-A": [("S2-X", 1, 0.85, "K1"), ("S2-Y", 2, 0.75, "K2")]}

    final, _ = union.merge_candidates(all_s1, bert, learned, classical)
    cands = final["S1-A"]
    assert len(cands) == len(set(cands)), "Duplicate candidates found in candidate list!"
    assert set(cands) == {"S2-X", "S2-Y"}


# 10. Output Format test
def test_output_format(tmp_path):
    out_file = tmp_path / "candidate_pairs.tsv"
    mapping = {
        "S1-1": ["S2-101", "S3-201"],
        "S1-2": [],
    }
    CandidateUnion.export_candidate_pairs(mapping, str(out_file))

    with open(out_file, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        assert header == "source1_entity_id\tcandidate_entity_ids"
        line1 = f.readline().rstrip("\n")
        assert line1 == "S1-1\tS2-101,S3-201"
        line2 = f.readline().rstrip("\n")
        assert line2 == "S1-2\t"


# 11. Exactly one output row per S1 entity
def test_exactly_one_output_row_per_s1_entity(tmp_path):
    all_s1 = [f"S1-{i:04d}" for i in range(100)]
    union = CandidateUnion()
    final, _ = union.merge_candidates(all_s1_ids=all_s1)

    out_file = tmp_path / "candidate_pairs.tsv"
    union.export_candidate_pairs(final, str(out_file))

    df = pd.read_csv(out_file, sep="\t", keep_default_na=False)
    assert len(df) == 100
    assert df["source1_entity_id"].tolist() == all_s1


# 12. No S1 IDs as candidates (no self-matches)
def test_no_s1_ids_as_candidates():
    union = CandidateUnion()
    all_s1 = ["S1-1"]
    # Poisoned input containing an S1 self-match
    poisoned_bert = {"S1-1": [("S1-1", 1, 1.0), ("S2-100", 2, 0.9)]}
    final, _ = union.merge_candidates(all_s1, bert_candidates=poisoned_bert)

    assert "S1-1" not in final["S1-1"]
    assert final["S1-1"] == ["S2-100"]


# 13. No nonexistent S2/S3 IDs test
def test_no_nonexistent_ids(mock_data_dir):
    s1 = load_source_file(os.path.join(mock_data_dir, "s1.tsv"))
    s2 = load_source_file(os.path.join(mock_data_dir, "s2.tsv"))
    s3 = load_source_file(os.path.join(mock_data_dir, "s3.tsv"))

    valid_targets = set(s2["entity_id"]).union(set(s3["entity_id"]))
    blocker = ClassicalBlocker()
    blocker.fit_targets(s2, s3)
    results = blocker.retrieve_candidates(s1)

    for s1_id, cands in results.items():
        for cid, _, _, _ in cands:
            assert cid in valid_targets, f"Candidate {cid} does not exist in target set!"


# 14. Deterministic output test
def test_deterministic_output():
    all_s1 = ["S1-1", "S1-2", "S1-3"]
    bert = {
        "S1-1": [("S2-B", 1, 0.8), ("S2-A", 2, 0.8)],
        "S1-2": [("S3-Z", 1, 0.5)],
    }
    union = CandidateUnion()
    run1, _ = union.merge_candidates(all_s1, bert_candidates=bert)
    run2, _ = union.merge_candidates(all_s1, bert_candidates=bert)

    assert run1 == run2, "Candidate union output must be deterministic!"


# 15. Blocking Evaluation & Penalty metrics test
def test_blocking_evaluation():
    gt = {
        "S1-1": ["S2-A", "S3-B"],
        "S1-2": ["S2-C"],
        "S1-3": [],  # Singleton
    }
    cands = {
        "S1-1": ["S2-A", "S3-B", "S2-Wasted1"],
        "S1-2": ["S2-Wasted2"],  # Missed S2-C
        "S1-3": [],
    }

    metrics = compute_blocking_metrics(
        candidate_mapping=cands,
        ground_truth=gt,
        num_total_targets=1000,
        candidate_budget_penalty_k=50.0,
    )

    assert metrics["total_true_matches"] == 3
    assert metrics["true_matches_recovered"] == 2
    assert pytest.approx(metrics["candidate_recall"], 1e-4) == 2 / 3
    assert metrics["missed_true_matches"] == 1
    assert metrics["zero_candidate_entities"] == 1
    assert metrics["wasted_candidates"] == 2
    assert metrics["reduction_ratio"] > 0.99
    assert metrics["composite_diagnostic_score"] > 0.0

    report = format_metrics_report(metrics)
    assert "Blocking Evaluation Report" in report


# 16. LightGBM Feature generation and Macro F_0.5 test
def test_lightgbm_features_and_f05():
    s1_row = pd.Series({
        "name_norm": "acme corporation",
        "address_norm": "100 main st",
        "country_norm": "US",
        "postal_code": "10001",
    })
    cand_row = pd.Series({
        "name_norm": "acme corp",
        "address_norm": "100 main street",
        "country_norm": "US",
        "postal_code": "10001",
    })

    feats = compute_pairwise_features(s1_row, cand_row, bert_sim=0.95, classical_score=0.90)
    assert len(feats) == 13
    assert feats[5] == 1.0  # country match
    assert feats[6] == 1.0  # postal match
    assert feats[7] == 0.95 # bert sim

    # Test Macro F_0.5 metric with singleton
    gt = {
        "S1-1": ["S2-A", "S3-B"],
        "S1-2": [],  # singleton
    }
    preds_perfect = {
        "S1-1": ["S2-A", "S3-B"],
        "S1-2": [],
    }
    score_perfect = macro_f_beta(preds_perfect, gt, beta=0.5)
    assert pytest.approx(score_perfect, 1e-4) == 1.0

    # False merge on singleton hurts score
    preds_imperfect = {
        "S1-1": ["S2-A", "S3-B"],
        "S1-2": ["S2-FalseMerge"],
    }
    score_imperfect = macro_f_beta(preds_imperfect, gt, beta=0.5)
    assert score_imperfect < 1.0
