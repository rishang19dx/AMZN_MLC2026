# Business Entity Resolution Pipeline

This repository contains an end-to-end pipeline for the Business Entity Resolution Challenge.

## Pipeline Architecture
1. **Local validation split:** Carves the training data into two closed universes (`local_train`, `local_val`) that mirror the test setup (see below).
2. **Blocking:** Generates a high-recall candidate set (`candidate_pairs.tsv`). *Being rewritten.*
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
Writes `../../dataset/splits/{local_train,local_val}/<split>_source{1,2,3}.tsv` and `<split>_ground_truth.tsv` (same layout as `dataset/train/`). About 1 minute.

- `local_val` = 10% of S1 entities (seeded md5 of the ID) + every S2/S3 record matched to them + 10% of the S2/S3 records that match nothing.
- `local_train` = everything else. S2/S3 are split exactly, because every S2/S3 record matches at most one S1 entity.

This keeps each split a closed universe with the train ratios (4.7 targets per S1, 26% unmatched targets). Scoring held-out S1 against the *full* S2/S3 would be misleading: 90% of the targets would belong to S1 entities outside the split.

**Scoring on the local split** (same macro-averaged F0.5 as the leaderboard, singletons included):
```bash
python src/evaluate.py --split local_val --matching ../../output/local_val/matching_results.tsv
python src/evaluate.py --split local_val --candidates ../../output/local_val/candidate_pairs.tsv
```
The matching report breaks the score down by country, singletons, precision and recall. The candidate report gives pair recall, reduction ratio, candidates per entity, and `f05_ceiling` (the best score a perfect matcher could reach on those candidates). Run `python tests/test_evaluate.py` to check the scorer against the problem-statement example.

All challenge TSVs must be read with quoting disabled (`data_loader.read_tsv`): some fields contain literal `"` characters.

**2. Run Blocking to generate Candidate Pairs:**
```bash
python src/blocking.py
```

**3. Run Matching to generate Final Results:**
```bash
python src/matching.py
```

Your final output files will be generated in `../../output/`.
