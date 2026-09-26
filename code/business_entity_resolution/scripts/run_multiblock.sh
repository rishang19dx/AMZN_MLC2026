#!/usr/bin/env bash
# End-to-end: multi-pipeline blocker -> basic LightGBM matcher -> validated submission.
# Run from anywhere; every stage can be re-run on its own (see docs/BLOCKING.md).
#
#   bash scripts/run_multiblock.sh
#   TRAIN_FRACTION=0.1 EPOCHS=1 bash scripts/run_multiblock.sh      # quick run
#   STAGES="block_test predict validate" bash scripts/run_multiblock.sh
#   PIPELINES=classical bash scripts/run_multiblock.sh               # no neural training at all
#
# Stages: splits train block_val check_val fit block_test predict validate
# Overridable: DATA (dir with train/ test/ splits/), ART (artifacts), OUT (outputs), CONFIG,
#              TRAIN_FRACTION, MATCHER_FRACTION, EPOCHS, DEVICE, PIPELINES, PY
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PROJ/../.." && pwd)"
DATA="${DATA:-${BER_DATA_DIR:-$REPO/dataset}}"
ART="${ART:-$REPO/artifacts}"
OUT="${OUT:-$REPO/output}"
CONFIG="${CONFIG:-$PROJ/configs/blocking.yaml}"
PY="${PY:-python}"
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
MATCHER_FRACTION="${MATCHER_FRACTION:-1.0}"
PIPELINES="${PIPELINES:-classical,bert,jepa}"
export BER_DATA_DIR="$DATA"
export PYTHONUNBUFFERED=1

wanted() { [ -z "${STAGES:-}" ] || [[ " $STAGES " == *" $1 "* ]]; }
run() { echo "== $*"; "$@"; }
cd "$PROJ"

LEARNED=$(echo "$PIPELINES" | tr ',' '\n' | grep -E '^(bert|jepa)$' | paste -sd, - || true)
EXTRA=()
[ -n "${EPOCHS:-}" ] && EXTRA+=(--epochs "$EPOCHS")
[ -n "${DEVICE:-}" ] && EXTRA+=(--device "$DEVICE")

# 1. entity-level closed-universe splits (local_train / local_val) from src/data_loader.py
if wanted splits && [ ! -f "$DATA/splits/local_val/local_val_source1.tsv" ]; then
    run "$PY" src/data_loader.py
fi

# 2. fine-tune the learned encoders on local_train (validation entities never seen). With no learned
#    pipeline this only builds the native-script dictionary ($ART/translit.json) that every run uses.
if wanted train; then
    run "$PY" src/train_blocker.py --train-dir "$DATA/splits/local_train" --val-dir "$DATA/splits/local_val" \
        --output-dir "$ART" --pipelines "${LEARNED:-none}" --train-fraction "$TRAIN_FRACTION" --no-eval \
        --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"}
fi

# 3. blocker on the labelled holdout: candidates (= matcher training data) + blocking report + K sweep
wanted block_val && run "$PY" src/generate_candidates.py --data-dir "$DATA/splits/local_val" \
    --artifacts-dir "$ART" --output-dir "$OUT/local_val" --pipelines "$PIPELINES" --k-sweep --config "$CONFIG"
wanted check_val && run "$PY" src/check_blocking.py --candidates "$OUT/local_val/candidate_pairs.tsv" \
    --data-dir "$DATA/splits/local_val" --debug "$OUT/local_val/debug_candidate_scores" \
    --json "$OUT/local_val/check_blocking.json"

# 4. LightGBM matcher, grouped K-fold on the holdout candidates, threshold tuned for F0.5
wanted fit && run "$PY" src/lgbm_matcher.py fit --data-dir "$DATA/splits/local_val" \
    --candidates-dir "$OUT/local_val" --artifacts-dir "$ART" --train-fraction "$MATCHER_FRACTION" --config "$CONFIG"

# 5. test: candidates, then matches (a subset of the candidates)
wanted block_test && run "$PY" src/generate_candidates.py --data-dir "$DATA/test" --artifacts-dir "$ART" \
    --output-dir "$OUT/test" --pipelines "$PIPELINES" --config "$CONFIG"
wanted predict && run "$PY" src/lgbm_matcher.py predict --data-dir "$DATA/test" --candidates-dir "$OUT/test" \
    --artifacts-dir "$ART" --output-dir "$OUT/test" --config "$CONFIG"

# 6. official format check
wanted validate && run python3 "$REPO/utils/validate_submission.py" --matching "$OUT/test/matching_results.tsv" \
    --candidate "$OUT/test/candidate_pairs.tsv" --test-dir "$DATA/test"
echo "submission files: $OUT/test/matching_results.tsv  $OUT/test/candidate_pairs.tsv"
