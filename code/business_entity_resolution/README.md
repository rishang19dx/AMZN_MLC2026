# Business Entity Resolution Pipeline

This repository contains an end-to-end pipeline for the Business Entity Resolution Challenge.

## Pipeline Architecture
1. **Data Loading & Validation:** Splits `train_source1.tsv` into a local train/validation set to tune models offline.
2. **Blocking (Splink):** Uses Splink + DuckDB to generate a high-recall candidate set (`candidate_pairs.tsv`).
3. **Matching (DeBERTa-v3):** Uses a pre-trained NLI Cross-Encoder (`cross-encoder/nli-deberta-v3-base`) to score semantic similarity and outputs the final `matching_results.tsv`.

## Setup
1. Create a Python 3.10+ virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies (we recommend installing the CPU-only version of PyTorch if you are testing locally without a GPU):
   ```bash
   # If you have CUDA 11/12:
   pip install -r requirements.txt
   
   # Or for CPU-only (saves disk space):
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

## Running the Pipeline
Ensure the raw datasets are placed in `../../dataset/` (relative to this directory).

**1. Create local validation split:**
```bash
python src/data_loader.py
```

**2. Run Blocking to generate Candidate Pairs:**
```bash
python src/blocking.py
```

**3. Run Matching to generate Final Results:**
```bash
python src/matching.py
```

Your final output files will be generated in `../../output/`.
