#!/usr/bin/env bash
# End-to-end pipeline: raw data -> both submission files.
#
#   splits -> native-script dictionary -> blocking -> features -> matcher -> validate
#
# Stages whose outputs already exist are skipped (delete them, or pass FORCE=1,
# to recompute), so a disconnected Kaggle/Colab session can simply be rerun.
#
# Usage (from code/business_entity_resolution/):
#   bash scripts/run_pipeline.sh              # local_val (train + score) and test (predict)
#   SPLITS=local_val bash scripts/run_pipeline.sh   # dev loop only
#
# Decisions baked in (see docs/PIPELINE.md §5):
#   * the dictionary is learned from local_train only and used for every split,
#     so the local_val score stays honest and train/test features are consistent;
#   * the matcher is trained on local_val candidates (cross-fit for the score and
#     the decoding choice, then a final fit on all of local_val).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
SPLITS="${SPLITS:-local_val test}"
FORCE="${FORCE:-0}"
DATA="${BER_DATA_DIR:-../../dataset}"
CACHE="${BER_CACHE_DIR:-../../cache}"
OUT="${BER_OUTPUT_DIR:-../../output}"

step() { echo; echo "=== $* ($(date +%H:%M:%S))"; }
need() { [ "$FORCE" = 1 ] || [ ! -e "$1" ]; }

step "1. local splits"
need "$DATA/splits/local_val/local_val_ground_truth.tsv" && $PY src/data_loader.py || echo "exists, skipped"

step "2. native-script dictionary (from local_train)"
need "$CACHE/translit.json" && $PY src/translit.py --split local_train || echo "exists, skipped"

for s in $SPLITS; do
    step "3. blocking: $s"
    need "$CACHE/$s/candidates.parquet" && $PY src/blocking.py --split "$s" || echo "exists, skipped"
    step "4. features: $s"
    need "$CACHE/$s/features/part-000.parquet" && $PY src/features.py --split "$s" || echo "exists, skipped"
done

step "5. matcher: cross-fit score + decoding choice on local_val"
need "$CACHE/models/decode.json" && $PY src/match.py --split local_val --cv || echo "exists, skipped"
step "6. matcher: final fit on local_val"
need "$CACHE/models/stage2.txt" && $PY src/match.py --split local_val --fit || echo "exists, skipped"

if [[ " $SPLITS " == *" test "* ]]; then
    step "7. predict test"
    $PY src/match.py --split test --predict ${SHIFT:+--shift "$SHIFT"}
    step "8. validate submission files"
    python3 ../../utils/validate_submission.py --matching "$OUT/test/matching_results.tsv" \
        --candidate "$OUT/test/candidate_pairs.tsv" --test-dir "$DATA/test"
fi
step "done"
