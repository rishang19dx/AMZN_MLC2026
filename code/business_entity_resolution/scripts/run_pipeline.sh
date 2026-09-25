#!/usr/bin/env bash
# End-to-end pipeline: raw data -> both submission files.
#
#   splits -> native-script dictionary -> blocking -> features -> matcher -> validate
#
# Stopping and resuming: finished stages are skipped (each writes a completion
# marker last), and blocking/features also keep their finished parts, so after
# a stop (Ctrl-C, scripts/mem_guard.sh, a crash) rerunning the same command
# continues where it stopped. FORCE=1 recomputes everything.
#
# Usage (from code/business_entity_resolution/):
#   bash scripts/mem_guard.sh &                      # recommended on a shared laptop
#   bash scripts/run_pipeline.sh                     # local_val (train + score) and test (predict)
#   SPLITS=local_val bash scripts/run_pipeline.sh    # dev loop only
#   SHIFT=-0.5 bash scripts/run_pipeline.sh          # stricter decoding for test
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
# run_unless <marker> <command...>: skip if the marker exists; otherwise run,
# and let a failure stop the whole pipeline (set -e) instead of continuing.
run_unless() {
    local marker="$1"; shift
    if [ "$FORCE" != 1 ] && [ -e "$marker" ]; then echo "done earlier, skipped"; else "$@"; fi
}

step "1. local splits"
run_unless "$DATA/splits/local_val/local_val_ground_truth.tsv" $PY src/data_loader.py

step "2. native-script dictionary (from local_train)"
run_unless "$CACHE/translit.json" $PY src/translit.py --split local_train

for s in $SPLITS; do
    step "3. blocking: $s"
    run_unless "$CACHE/$s/blocking.done" $PY src/blocking.py --split "$s"
    step "4. features: $s"
    run_unless "$CACHE/$s/features/_DONE" $PY src/features.py --split "$s"
done

step "5. matcher: cross-fit score + decoding choice on local_val"
run_unless "$CACHE/models/decode.json" $PY src/match.py --split local_val --cv
step "6. matcher: final fit on local_val"
run_unless "$CACHE/models/stage2.txt" $PY src/match.py --split local_val --fit

if [[ " $SPLITS " == *" test "* ]]; then
    step "7. predict test"
    $PY src/match.py --split test --predict ${SHIFT:+--shift "$SHIFT"}
    step "8. validate submission files"
    python3 ../../utils/validate_submission.py --matching "$OUT/test/matching_results.tsv" \
        --candidate "$OUT/test/candidate_pairs.tsv" --test-dir "$DATA/test"
fi
step "done"
