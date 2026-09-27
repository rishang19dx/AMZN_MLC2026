#!/usr/bin/env bash
# Pair energy model v2 end to end on airawat (one GPU, capped CPU threads).
#
#   ROOT=/scratch/s25017 bash ebm/run_airawat.sh           # from the repo root (the folder with ebm/)
#
# Steps (each skipped if its output exists, so rerunning resumes):
#   1. preprocess train  (learns the native-script dictionary from train pairs)
#   2. preprocess test   (reuses that dictionary)
#   3. train             (bf16 on the freest GPU)
#   4. score test candidates, if $CANDIDATES points at a candidate_pairs.tsv
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${ROOT:?set ROOT, e.g. ROOT=/scratch/s25017}"
DATA="${DATA:-$ROOT/dataset}"
CACHE="${CACHE:-$ROOT/ebm_cache}"
OUT="${OUT:-$ROOT/ebm_artifacts}"
PY="${PY:-python}"
export EBM_WORKERS="${EBM_WORKERS:-32}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" PYTHONUNBUFFERED=1
mkdir -p "$CACHE" "$OUT"

[ -f "$CACHE/train/meta.json" ] || $PY -m ebm.preprocess --split-dir "$DATA/train" --prefix train --out "$CACHE/train" --learn-dictionary
[ -f "$CACHE/test/meta.json" ]  || $PY -m ebm.preprocess --split-dir "$DATA/test" --prefix test --out "$CACHE/test" --dictionary "$CACHE/train/translit.json"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(timeout 20 nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1 || true)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
echo "GPU $CUDA_VISIBLE_DEVICES"
[ -f "$OUT/best.pt" ] || $PY -m ebm.train --cache "$CACHE/train" --out "$OUT" \
    --epochs "${EPOCHS:-3}" --batch "${BATCH:-512}" --hard "${HARD:-7}" --d "${D:-128}" --layers "${LAYERS:-2}"

if [ -n "${CANDIDATES:-}" ]; then
    $PY -m ebm.score --model "$OUT/best.pt" --cache "$CACHE/test" --candidates "$CANDIDATES" --out "$OUT/test_scores.tsv"
fi
echo "done: $OUT"
