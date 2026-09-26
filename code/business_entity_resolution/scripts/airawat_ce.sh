#!/usr/bin/env bash
# Cross-encoder on the shared `airawat` node (8x H200, 224 threads, no SLURM).
#
# The laptop exports the text pairs (cross_encoder.py export); this script trains
# the listwise mDeBERTa cross-encoder and scores the uncertain band of scale_val
# and test. Only ~0.5 GB crosses between machines:
#   in : $BER_CACHE_DIR/ce/{ce_train,scale_val,test}.parquet   (from the laptop)
#   out: $BER_CACHE_DIR/{scale_val,test}/ce_scores.parquet      (back to the laptop)
#
# Home has a 10 GB quota, so env, cache and model live under /storage.
# Shared machine: one GPU (the least used), capped CPU threads, nice.
#
# Usage (on airawat, from the repo's code/business_entity_resolution/):
#   bash scripts/airawat_ce.sh setup     # once: conda env with CUDA torch
#   bash scripts/airawat_ce.sh run       # train + score scale_val + score test
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STORE="${STORE:-/storage/$USER}"
ENV="$STORE/envs/ber"
CONDA="${CONDA:-$STORE/miniconda3}"
export BER_CACHE_DIR="${BER_CACHE_DIR:-$STORE/ber_cache}"
export BER_WORKERS="${BER_WORKERS:-32}"
export OMP_NUM_THREADS="$BER_WORKERS" TOKENIZERS_PARALLELISM=true
export HF_HOME="$STORE/hf_cache"

case "${1:-}" in
setup)
    source "$CONDA/etc/profile.d/conda.sh"
    [ -d "$ENV" ] || conda create -y -p "$ENV" python=3.10
    conda activate "$ENV"
    pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
    pip install transformers==5.17.0 tokenizers==0.23.2 safetensors==0.8.0 \
        sentencepiece protobuf duckdb==1.5.5 pandas==2.3.3 numpy==2.2.6
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'GPUs')"
    ;;
run)
    source "$CONDA/etc/profile.d/conda.sh"
    conda activate "$ENV"
    for s in ce_train scale_val test; do
        [ -f "$BER_CACHE_DIR/ce/$s.parquet" ] || { echo "missing $BER_CACHE_DIR/ce/$s.parquet (copy it from the laptop)"; exit 1; }
    done
    # the GPU with the most free memory right now
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    echo "using GPU $CUDA_VISIBLE_DEVICES, $BER_WORKERS CPU threads, cache $BER_CACHE_DIR"
    nice python src/cross_encoder.py train --loss listwise
    nice python src/cross_encoder.py score --split scale_val
    nice python src/cross_encoder.py score --split test
    ls -la "$BER_CACHE_DIR"/scale_val/ce_scores.parquet "$BER_CACHE_DIR"/test/ce_scores.parquet
    ;;
*)
    echo "usage: $0 setup|run"; exit 1 ;;
esac
