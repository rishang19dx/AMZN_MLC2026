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
| Negatives | 1 random per positive | K same-country look-alikes (shared rare name word or house number), never a true match (every target has ≤ 1 Source 1) |
| Encoder | mean of hashed embeddings (d = 64) | token + field embeddings → 2-layer Transformer → attention + mean pooling |
| Head | MLP on entity vectors | entity interactions + late interaction (token alignment, overall / name / address, both directions) |
| Loss | BCE | listwise CE over [positive + K hard] + BCE + in-batch bi-encoder InfoNCE |
| Validation | positives + 1 random negative per source (far too easy) | held-out Source 1 against their positives + 20 mined look-alikes; macro F0.5, AUC, top-1 |
| Size | ~17M | ~136M (d = 128, 1M buckets) |

## Run

```bash
ROOT=/scratch/s25017 bash ebm/run_airawat.sh
ROOT=/scratch/s25017 CANDIDATES=/path/to/test/candidate_pairs.tsv bash ebm/run_airawat.sh
```
- The first command preprocesses train and test, then trains.
- The second also scores the given test candidates → `$ROOT/ebm_artifacts/test_scores.tsv`.
- Knobs: `EPOCHS`, `BATCH`, `HARD`, `D`, `LAYERS`, `EBM_WORKERS`, `CUDA_VISIBLE_DEVICES`.
- Individual steps: `python -m ebm.preprocess | ebm.train | ebm.score --help`.

## Use with the main pipeline

`test_scores.tsv` gives `p_match` per (Source 1, candidate). Joined onto the main pipeline's candidates, it becomes one more stage-2 feature, the same integration path as the cross-encoder (`match.py --fit --stage2-only`). Only the pair score is used: blocking and decoding stay as they are.

## Smoke test (CPU, tiny model, 60 steps on `local_val`)

On the hard validation (mined look-alikes), AUC went from 0.81 to 0.83 and macro F0.5 from 0.60 to 0.63, and the top-scored candidate is a true match 87% of the time. These numbers are **not** comparable with the main pipeline's F0.5, which is measured on its own candidate lists.
