#!/usr/bin/env bash
# Full K=50 run on airawat: retrain the matcher on scale_val and predict test,
# with blocking at --k-addr 30 --k-full 50 (scale_val recall 97.5% vs 96.9% at 20/30).
#
# Two chains run in parallel, then predict:
#   A: scale_val  blocking -> features -> match --cv (score + decoding) -> match --fit
#   B: test       blocking -> features
#   C: test       match --predict -> validator
# Artifacts go to separate folders (cache_k50/, output_k50/), so the K=20/30
# submission (LB 0.959) is never touched. Stages with finished outputs are
# skipped, so rerunning the script resumes.
#
# Usage (from code/business_entity_resolution/ on airawat):
#   ROOT=/path/to/scratch bash scripts/airawat_k50.sh
#   tail -f $ROOT/output_k50/logs/*.log          # progress
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="${ROOT:?set ROOT to the scratch folder, e.g. ROOT=/scratch/$USER/mlc26}"
PY="${PY:-python}"
export BER_DATA_DIR="${BER_DATA_DIR:-$ROOT/dataset}"          # dataset/{train,test} + splits/
export BER_CACHE_DIR="$ROOT/cache_k50"
export BER_OUTPUT_DIR="$ROOT/output_k50"
export BER_DUCKDB_MEM="${BER_DUCKDB_MEM:-64GB}" BER_BLOCK_MEM_GB="${BER_BLOCK_MEM_GB:-48}"
export BER_FEATURE_CHUNK="${BER_FEATURE_CHUNK:-5000000}"
THREADS_EACH="${THREADS_EACH:-64}"                            # per chain; 2 chains in parallel
K="--k-addr 30 --k-full 50"
LOG="$BER_OUTPUT_DIR/logs"; mkdir -p "$LOG" "$BER_CACHE_DIR"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/main.log"; }

# shared prerequisites (once): splits and the native-script dictionary
[ -e "$BER_DATA_DIR/splits/local_val/local_val_ground_truth.tsv" ] || $PY src/data_loader.py
[ -e "$BER_DATA_DIR/splits/ce_train/ce_train_source1.tsv" ] || $PY src/data_loader.py --ce-train
[ -e "$BER_DATA_DIR/splits/scale_val/scale_val_source1.tsv" ] || $PY src/data_loader.py --scale-val
[ -e "$BER_CACHE_DIR/translit.json" ] || $PY src/translit.py --split local_train

block() {   # blocking + features for one split
    local s="$1"
    [ -e "$BER_CACHE_DIR/$s/blocking.done" ] || $PY src/blocking.py --split "$s" $K
    [ -e "$BER_CACHE_DIR/$s/features/_DONE" ] || { $PY src/features.py --split "$s" --index-only && $PY src/features.py --split "$s"; }
}

chain_scale() {
    export BER_WORKERS=$THREADS_EACH
    block scale_val
    [ -e "$BER_CACHE_DIR/models/decode.json" ] || $PY src/match.py --split scale_val --cv
    [ -e "$BER_CACHE_DIR/models/stage2.txt" ] || $PY src/match.py --split scale_val --fit
}
chain_test() {
    export BER_WORKERS=$THREADS_EACH
    block test
}

say "start: data $BER_DATA_DIR, cache $BER_CACHE_DIR, $THREADS_EACH threads per chain"
chain_scale > "$LOG/scale_val.log" 2>&1 & A=$!
chain_test  > "$LOG/test.log" 2>&1 & B=$!
wait $A && say "scale_val chain done (models in $BER_CACHE_DIR/models)" || { say "scale_val chain FAILED, see $LOG/scale_val.log"; exit 1; }
wait $B && say "test chain done" || { say "test chain FAILED, see $LOG/test.log"; exit 1; }

say "predict test"
BER_WORKERS=$((2 * THREADS_EACH)) $PY src/match.py --split test --predict > "$LOG/predict.log" 2>&1
$PY ../../utils/validate_submission.py --matching "$BER_OUTPUT_DIR/test/matching_results.tsv" \
    --candidate "$BER_OUTPUT_DIR/test/candidate_pairs.tsv" --test-dir "$BER_DATA_DIR/test" | tee -a "$LOG/main.log"
grep -h "best:\|f05 " "$LOG/scale_val.log" | tail -3 | tee -a "$LOG/main.log"
say "done: $BER_OUTPUT_DIR/test/matching_results.tsv and candidate_pairs.tsv"
