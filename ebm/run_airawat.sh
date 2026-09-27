#!/usr/bin/env bash
# Pair energy model v2, complete pipeline on airawat (one GPU, capped CPU threads).
#
#   ROOT=/scratch/s25017 bash ebm/run_airawat.sh          # from the folder that contains ebm/
#
#   1. preprocess train (+ native-script dictionary from train pairs) and test
#   2. block train: S1_FRACTION of train-split Source 1 + ALL validation Source 1,
#      against ALL train targets (test-sized pool)          -> training candidates
#   3. block test: every Source 1                           -> test candidates
#   4. train the ranker on candidate-pool negatives; end-to-end validation on
#      VAL_ANCHORS held-out Source 1 with full candidate lists
#   5. predict test: score, decode, write matching_results.tsv + candidate_pairs.tsv
# Each step is skipped when its output exists, so rerunning resumes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${ROOT:?set ROOT, e.g. ROOT=/scratch/s25017}"
DATA="${DATA:-$ROOT/dataset}"
CACHE="${CACHE:-$ROOT/ebm_cache}"
OUT="${OUT:-$ROOT/ebm_artifacts}"
PY="${PY:-python}"
K_ADDR="${K_ADDR:-30}" K_FULL="${K_FULL:-50}"
export EBM_WORKERS="${EBM_WORKERS:-48}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" PYTHONUNBUFFERED=1
mkdir -p "$CACHE" "$OUT"
say() { echo "=== $* ($(date +%H:%M:%S))"; }

say "1. preprocess"
[ -f "$CACHE/train/meta.json" ] || $PY -m ebm.preprocess --split-dir "$DATA/train" --prefix train --out "$CACHE/train" --learn-dictionary
[ -f "$CACHE/test/meta.json" ]  || $PY -m ebm.preprocess --split-dir "$DATA/test" --prefix test --out "$CACHE/test" --dictionary "$CACHE/train/translit.json"

say "2. block train (S1 fraction ${S1_FRACTION:-0.3} + all validation Source 1)"
[ -f "$CACHE/train/candidates.npz" ] || $PY -m ebm.block --cache "$CACHE/train" --k-addr "$K_ADDR" --k-full "$K_FULL" --s1-fraction "${S1_FRACTION:-0.3}"
say "3. block test"
[ -f "$CACHE/test/candidates.npz" ] || $PY -m ebm.block --cache "$CACHE/test" --k-addr "$K_ADDR" --k-full "$K_FULL"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES=$(timeout 20 nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1 || true)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
say "4. train on GPU $CUDA_VISIBLE_DEVICES"
[ -f "$OUT/best.pt" ] || $PY -m ebm.train --cache "$CACHE/train" --out "$OUT" \
    --epochs "${EPOCHS:-3}" --batch "${BATCH:-512}" --hard "${HARD:-15}" --mined-share "${MINED_SHARE:-0.25}" \
    --d "${D:-128}" --layers "${LAYERS:-2}" --val-anchors "${VAL_ANCHORS:-50000}" --val-every "${VAL_EVERY:-1000}"

say "5. predict test"
$PY -m ebm.predict --model "$OUT/best.pt" --cache "$CACHE/test" --out "$OUT/test"
if [ -f "$ROOT/utils/validate_submission.py" ] || [ -f utils/validate_submission.py ]; then
    V=$( [ -f utils/validate_submission.py ] && echo utils/validate_submission.py || echo "$ROOT/utils/validate_submission.py" )
    $PY "$V" --matching "$OUT/test/matching_results.tsv" --candidate "$OUT/test/candidate_pairs.tsv" --test-dir "$DATA/test" | tail -2
fi
say "done: $OUT/test/matching_results.tsv"
