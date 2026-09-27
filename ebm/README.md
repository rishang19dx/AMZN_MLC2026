# Pair energy model v2 (`ebm/`)

A rework of `train_pair_distribution.py` (kept unchanged for comparison) for the full dataset:
- extensive preprocessing, done once and in parallel;
- look-alike negatives instead of random ones;
- a scaled encoder with token-level interaction;
- a validation that resembles test.

| | v1 (`train_pair_distribution.py`) | v2 (`ebm/`) |
|---|---|---|
| Normalisation | lowercase + `\w` tokens | NFKC; native-script dictionary learned from train pairs; accents; dotted initialisms; legal forms (US/India/France) in one pass; honorifics, `M/s`; leetspeak; web names; address abbreviations (US/India/France), ordinals, states; house number / unit / postcode split out; repeated words |
| Preprocessing cost | re-tokenised in Python for every pair, every epoch (SQLite joins) | once, N processes, memory-mapped `int32` arrays (`local_val`: 1.25M records in 26 s on 12 processes) |
| Negatives | 1 random per positive | the anchor's own blocked non-matches (whole candidate pool, 15 per step) + 25% same-country look-alikes; never a true match (every target has ≤ 1 Source 1) |
| Encoder | mean of hashed embeddings (d = 64) | token + field embeddings → 2-layer Transformer → attention + mean pooling |
| Head | MLP on entity vectors | entity interactions + late interaction (token alignment, overall / name / address, both directions) |
| Loss | BCE | listwise CE over [positive + K hard] + BCE + in-batch bi-encoder InfoNCE |
| Validation | positives + 1 random negative per source (far too easy) | end to end: up to 50k held-out Source 1 with full blocked candidate lists, blocking misses counted, same decoding as predict |
| Size | ~17M | ~136M (d = 128, 1M buckets) |

## Pipeline

The model is a **ranker**: it only scores candidates that blocking produced, and never searches.

```
preprocess (once)  ->  block (per-country TF-IDF top-K)  ->  train (candidate-pool negatives)  ->  predict (score + decode)
```

| Step | Module | What it does | Measured on `local_val` (laptop) |
|---|---|---|---|
| 1 | `ebm.preprocess` | Normalise + tokenise every record once, in parallel → memory-mapped arrays; native-script dictionary learned from train pairs | 1.25M records in 26 s (12 processes) |
| 2 | `ebm.block` | Two TF-IDF passes (address; name+address) per country, IDF from the split, top-K 30/50 (default), union. Train: a Source 1 sample + **all** validation Source 1 against **all** targets | K 20/30: **98.89%** of true pairs kept, 38 per Source 1 (the main pipeline measured 98.8%) |
| 3 | `ebm.train` | Negatives = each anchor's own non-matching candidates (the whole pool, K per step) + 25% mined look-alikes; listwise + BCE + in-batch loss. **End-to-end validation** on up to 50k held-out Source 1 with full candidate lists (blocking misses count), same decoding as predict | Tiny CPU model, 60 steps: F0.5 0.477 (ceiling 0.995), AUC 0.84; the real run is on GPU |
| 4 | `ebm.predict` | Score all test candidates, one Source 1 per target, caps S2 ≤ 5 / S3 ≤ 6, validation threshold → `matching_results.tsv` + `candidate_pairs.tsv` (+ `scores.npz`) | – |

## Run

```bash
ROOT=/scratch/s25017 bash ebm/run_airawat.sh      # expects $ROOT/dataset/{train,test}
```
- Runs steps 1–4 and the submission validator; each step resumes if its output exists.
- Knobs: `K_ADDR`, `K_FULL`, `S1_FRACTION` (train blocking sample, default 0.3), `EPOCHS`, `BATCH`, `HARD` (negatives per anchor, default 15), `MINED_SHARE`, `VAL_ANCHORS` (default 50,000; 0 = all), `D`, `LAYERS`, `EBM_WORKERS`, `CUDA_VISIBLE_DEVICES`.
- `ebm.score` scores an external `candidate_pairs.tsv`, for example the main pipeline's, as a stage-2 feature.

## Use with the main pipeline

`test_scores.tsv` gives `p_match` per (Source 1, candidate). Joined onto the main pipeline's candidates, it becomes one more stage-2 feature, the same integration path as the cross-encoder (`match.py --fit --stage2-only`). Only the pair score is used: blocking and decoding stay as they are.

## Smoke tests (laptop CPU, tiny model: d = 32, 1 layer)

- **First version**, mined-negative validation: AUC 0.81 → 0.83 in 60 steps.
- **Current pipeline** on `local_val` (blocking K 20/30), validation on 3,000 held-out Source 1 with full candidate lists: F0.5 0.467 → 0.477, AUC 0.82 → 0.84 in 60 steps; ceiling 0.995, blocking recall 0.987.

These only show that the pipeline runs and learns; the real model trains on the GPU. The numbers are not comparable with the main pipeline's 0.967, which is measured on `scale_val`.
