#!/usr/bin/env bash
# End-to-end pipeline on an Apple Silicon Mac (tested on M4, 10 cores, 16 GB).
#
#   bash scripts/mac_run.sh                  # everything; rerun to resume after a crash
#   STAGES="fit eval" bash scripts/mac_run.sh   # only these stages
#   FORCE=1 STAGES=fit bash scripts/mac_run.sh  # redo a finished stage
#
# Stages (in order):
#   splits      local_train / local_val / local_fit            (data_loader.py)
#   translit    native-script dictionaries from local_train and from train
#   block_fit   feat_fit    candidates + features for local_fit   (training split)
#   block_val   feat_val    candidates + features for local_val   (holdout)
#   fit         5-fold x 2-stage LightGBM on local_fit; models saved per fold
#   eval        ensemble on local_val: holdout F0.5, tunes decode.json
#   block_test  feat_test   candidates + features for test
#   predict     output/test/matching_results.tsv
#   validate    utils/validate_submission.py
#
# Every stage logs to $LOG_DIR/<stage>.log (with wall time and peak memory from
# /usr/bin/time -l) and leaves a marker in $BER_CACHE_DIR/_done/, so a rerun
# skips it. The Mac is kept awake with caffeinate for the whole run.
#
# Overridable: BER_DATA_DIR BER_CACHE_DIR BER_OUTPUT_DIR BER_THREADS
#              BER_FIT_FRACTION (0.15) FOLDS (5) PY (python to use) LOG_DIR
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PROJ/../.." && pwd)"

# keep the machine awake (display may sleep; system, disk and idle sleep may not)
if [ -z "${BER_CAFFEINATED:-}" ] && command -v caffeinate > /dev/null; then
    export BER_CAFFEINATED=1
    exec caffeinate -ims bash "${BASH_SOURCE[0]}" "$@"
fi

export BER_DATA_DIR="${BER_DATA_DIR:-$(cd "$REPO/.." && pwd)/student_resource/dataset}"
export BER_CACHE_DIR="${BER_CACHE_DIR:-$REPO/cache}"
export BER_OUTPUT_DIR="${BER_OUTPUT_DIR:-$REPO/output}"
export BER_THREADS="${BER_THREADS:-$(sysctl -n hw.ncpu)}"
export OMP_NUM_THREADS="$BER_THREADS"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES   # blocking.py forks worker processes
export PYTHONUNBUFFERED=1
FOLDS="${FOLDS:-5}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"
DONE="$BER_CACHE_DIR/_done"
PY="${PY:-$PROJ/.venv/bin/python}"
mkdir -p "$BER_CACHE_DIR" "$BER_OUTPUT_DIR" "$LOG_DIR" "$DONE"

if [ ! -x "$PY" ]; then
    echo "== creating $PROJ/.venv"
    if command -v uv > /dev/null; then
        uv venv --python 3.12 "$PROJ/.venv" && uv pip install --python "$PROJ/.venv/bin/python" -r "$PROJ/requirements-mac.txt"
    else
        python3 -m venv "$PROJ/.venv" && "$PROJ/.venv/bin/pip" install -r "$PROJ/requirements-mac.txt"
    fi
    PY="$PROJ/.venv/bin/python"
fi
[ -f "$BER_DATA_DIR/train/train_source1.tsv" ] || { echo "no data in $BER_DATA_DIR/train"; exit 1; }

echo "data   $BER_DATA_DIR"
echo "cache  $BER_CACHE_DIR   (models: $BER_CACHE_DIR/models)"
echo "output $BER_OUTPUT_DIR   logs: $LOG_DIR"
echo "threads $BER_THREADS, folds $FOLDS, fit fraction ${BER_FIT_FRACTION:-0.15}"

cd "$PROJ"

wanted() { [ -z "${STAGES:-}" ] || [[ " $STAGES " == *" $1 "* ]]; }

# The normaliser reads $BER_CACHE_DIR/translit.json. Training/holdout splits must
# use the dictionary learned from local_train (keeps local_val honest); test
# uses the one learned from all of train. Switch before every stage.
use_dict() {
    local src="$BER_CACHE_DIR/translit_$1.json"
    [ -f "$src" ] || { echo "missing $src: run the translit stage first"; exit 1; }
    cp "$src" "$BER_CACHE_DIR/translit.json"
}

need_disk() {   # GB free on the cache volume
    local free
    free=$(df -g "$BER_CACHE_DIR" | awk 'NR==2 {print $4}')
    if [ "$free" -lt "$1" ]; then
        echo "only ${free} GB free on the cache volume; $2 needs about $1 GB. Free some space and rerun."
        exit 1
    fi
}

stage() {   # stage <name> <dict|-> <command...>
    local name="$1" dict="$2"
    shift 2
    wanted "$name" || return 0
    if [ -f "$DONE/$name" ] && [ -z "${FORCE:-}" ]; then
        echo "== $name: done already ($(cat "$DONE/$name")), skipping"
        return 0
    fi
    [ "$dict" = "-" ] || use_dict "$dict"
    echo "== $name: $* ($(date '+%a %H:%M:%S'))"
    local t0=$SECONDS
    /usr/bin/time -l "$@" 2>&1 | tee "$LOG_DIR/$name.log"
    local secs=$((SECONDS - t0))
    local peak
    peak=$(awk '/maximum resident set size/ {printf "%.1f GB", $1 / 1073741824}' "$LOG_DIR/$name.log")
    echo "$((secs / 60)) min $((secs % 60)) s, peak RAM ${peak:-?}, finished $(date '+%a %H:%M')" > "$DONE/$name"
    echo "== $name finished: $(cat "$DONE/$name")"
}

stage splits     -           "$PY" src/data_loader.py
stage translit   -           bash -c 'set -e; for sp in local_train train; do
    "$0" src/translit.py --split $sp; cp "$BER_CACHE_DIR/translit.json" "$BER_CACHE_DIR/translit_$sp.json"; done' "$PY"
stage block_fit  local_train "$PY" src/blocking.py --split local_fit
stage feat_fit   local_train "$PY" src/features.py --split local_fit
stage block_val  local_train "$PY" src/blocking.py --split local_val
stage feat_val   local_train "$PY" src/features.py --split local_val
stage fit        -           "$PY" src/match.py --split local_fit --fit --folds "$FOLDS"
stage eval       -           "$PY" src/match.py --split local_val --eval
if wanted block_test || wanted feat_test; then need_disk 12 "test blocking + features"; fi
stage block_test train       "$PY" src/blocking.py --split test
stage feat_test  train       "$PY" src/features.py --split test --chunk 2000000
stage predict    -           "$PY" src/match.py --split test --predict
stage validate   -           python3 "$REPO/utils/validate_submission.py" \
    --matching "$BER_OUTPUT_DIR/test/matching_results.tsv" \
    --candidate "$BER_OUTPUT_DIR/test/candidate_pairs.tsv" --test-dir "$BER_DATA_DIR/test"

echo
echo "== summary"
for f in splits translit block_fit feat_fit block_val feat_val fit eval block_test feat_test predict validate; do
    [ -f "$DONE/$f" ] && printf '  %-11s %s\n' "$f" "$(cat "$DONE/$f")"
done
echo "models:     $BER_CACHE_DIR/models/  ($(ls "$BER_CACHE_DIR/models" 2>/dev/null | tr '\n' ' '))"
echo "submission: $BER_OUTPUT_DIR/test/matching_results.tsv + candidate_pairs.tsv"
