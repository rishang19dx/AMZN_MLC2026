"""
Standalone test runner for all blocking system tests without external runner dependencies.
"""

import sys
import traceback
import tempfile
import os

from tests.test_blocking_system import (
    test_tsv_loading,
    test_normalization,
    test_positive_pair_construction,
    test_negative_pair_construction,
    test_embedding_generation,
    test_ann_retrieval,
    test_classical_blocking,
    test_candidate_union,
    test_deduplication,
    test_output_format,
    test_exactly_one_output_row_per_s1_entity,
    test_no_s1_ids_as_candidates,
    test_no_nonexistent_ids,
    test_deterministic_output,
    test_blocking_evaluation,
    test_lightgbm_features_and_f05,
    mock_data_dir,
)

class TmpPathWrapper:
    def __init__(self, path):
        self.path = path
    def __truediv__(self, other):
        return os.path.join(self.path, other)

def run_all_tests():
    tests = [
        ("1. TSV Loading", test_tsv_loading, True),
        ("2. Normalization", test_normalization, False),
        ("3. Positive Pair & Leak-Free Split", test_positive_pair_construction, True),
        ("4. Negative Pair Construction", test_negative_pair_construction, False),
        ("5. Embedding Generation & Normalization", test_embedding_generation, False),
        ("6. ANN Retrieval", test_ann_retrieval, False),
        ("7. Classical Blocking", test_classical_blocking, True),
        ("8. Candidate Union", test_candidate_union, False),
        ("9. Deduplication", test_deduplication, False),
        ("10. Output Format", test_output_format, "tmp_path"),
        ("11. Exactly One Row Per S1 Entity", test_exactly_one_output_row_per_s1_entity, "tmp_path"),
        ("12. No S1 IDs as Candidates", test_no_s1_ids_as_candidates, False),
        ("13. No Nonexistent S2/S3 IDs", test_no_nonexistent_ids, True),
        ("14. Deterministic Output", test_deterministic_output, False),
        ("15. Blocking Evaluation & Penalty", test_blocking_evaluation, False),
        ("16. LightGBM Features & F_0.5 Metric", test_lightgbm_features_and_f05, False),
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("RUNNING ALL 16 BLOCKING SYSTEM TEST SUITES")
    print("=" * 60)

    for name, test_fn, fixture_mode in tests:
        try:
            if fixture_mode == True:
                # Needs mock_data_dir
                gen = mock_data_dir()
                dir_path = next(gen)
                try:
                    test_fn(dir_path)
                finally:
                    try:
                        next(gen)
                    except StopIteration:
                        pass
            elif fixture_mode == "tmp_path":
                with tempfile.TemporaryDirectory() as td:
                    test_fn(TmpPathWrapper(td))
            else:
                test_fn()

            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED out of {len(tests)} tests.")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
