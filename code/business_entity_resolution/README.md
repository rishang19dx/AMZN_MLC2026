# Business Entity Resolution Pipeline (MLC 2026)

For each Source 1 (S1) business, find every matching S2/S3 record. Scored by F0.5, computed per S1 record and averaged, with no-match records included.
**Deadline: Sun 27 Sep 2026, 23:59 IST. Code freeze: Sun 16:00 IST.**

**Teammates:** read [`docs/REPORT.md`](docs/REPORT.md) (background, history, experiments, what is verified), then pick a task from [`docs/PIPELINE.md`](docs/PIPELINE.md) (status, stage contracts, task board, submission log).

## Current status (Fri 25 Sep, 23:00 IST)

| # | Stage | Command | State | Result / check |
|---|---|---|---|---|
| 1 | Local validation split | `src/data_loader.py` | ✅ done | self-contained `local_train` / `local_val`; exact partition checked |
| 2 | Local scorer | `src/evaluate.py` | ✅ done | reproduces the leaderboard F0.5; unit tests pass |
| 3 | Cloud runner (Kaggle / Colab) | `scripts/`, `notebooks/cloud_runner.ipynb` | ✅ done | rehearsed locally; ❌ not yet run on a real Kaggle machine |
| 4 | Blocking v1 | `src/blocking.py` | ✅ done | `local_val`: **98.2% of true pairs found, F0.5 ceiling 0.994**, 38 candidates per S1; ❌ test runtime not yet measured |
| 5 | Native-script dictionary + anyascii | `src/translit.py`, `src/normalize.py` | ✅ done | blocking recall 98.2% → **98.8%**; native-script names 88.5% → 97.8% |
| 6 | Matcher (features + two-stage LightGBM + one S1 per target + expected-F0.5 decoding) | `src/features.py`, `src/match.py` | ✅ done | `local_val` out-of-fold **F0.5 0.9858** (India 0.9835, US 0.9874) |
| 7 | Full test run → Submission 1 | `scripts/run_pipeline.sh` | ⏳ running (Sat 26 Sep, 02:08) | – |
| 8 | Cross-encoder (mDeBERTa-v3, stage-2 feature) | `src/cross_encoder.py`, `notebooks/kaggle_cross_encoder.ipynb` | code done; ⏳ T4 training | – |
| 9 | Submission package (validator, docs, zip) | – | ⏳ TODO | – |

`matching.py`, `preprocess.py` and `blocking_legacy.py` are **legacy**, kept for reference only; don't build on them.

## Pipeline

```
raw TSVs ──> data_loader ──> blocking ──> features ──> LightGBM ──> cross-encoder on ──> assign each ──> threshold ──> matching_results.tsv
             (splits)        (TF-IDF       (RapidFuzz,    (all pairs)   uncertain pairs      target to its
                              top-K per     ranks within                 → stacked           best S1
                              country)      each S1)                     LightGBM
                               │
                               └──> candidate_pairs.tsv + cache/<split>/candidates.parquet
```

Design in one line each (details and evidence in `docs/REPORT.md`):
- **Countries:** block within country and compare country as a plain string. True matches never cross countries, and this handles France, which appears only in test, without special code.
- **Normalisation:** no hand-written suffix or state lists; IDF (rarity weighting) learned from the data down-weights "pvt", "llc", "sarl", …
- **One S1 per target:** every S2/S3 record matches at most one S1, so each target is assigned only to its best S1.
- **Threshold:** tuned on `local_val`, then set slightly stricter for test, which has more distractors (5.75 targets per S1 vs 4.68).

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pinned; for CPU-only torch, first:
                                       # pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Put the raw data in `../../dataset/{train,test}/`, or extract `mlc26_data.tar.zst` there.
All challenge TSVs must be read with quoting disabled (`data_loader.read_tsv`), because some fields contain literal `"` characters.

## Running (all commands from this directory)

Every stage takes `--split {local_val|local_train|train|test}`. Develop on `local_val`; run `local_train` and `test` on Kaggle.

**1. Local splits** (~1 min) → `../../dataset/splits/{local_train,local_val}/`
```bash
python src/data_loader.py
```
- `local_val` = 10% of S1 (by seeded hash) + every S2/S3 record matched to them + 10% of the records that match nothing.
- `local_train` = everything else.
- Each is self-contained, like the test set. Scoring held-out S1 against *all* train S2/S3 would be misleading: 90% of the targets would belong to S1 records outside the split.

**2. Blocking** → `../../output/<split>/candidate_pairs.tsv` + `../../cache/<split>/candidates.parquet`
```bash
python src/blocking.py --split local_val     # also prints the share of true pairs each pass finds
```
Defaults: `--k-addr 20 --k-full 30 --k-name 0 --max-df 0.02 --workers $(nproc)`. The name-trigram pass is off because it is slow and weak; lowering `--max-df` is faster but costs recall. Both were measured; see REPORT §3.

**3. Score**
```bash
python src/evaluate.py --split local_val --candidates ../../output/local_val/candidate_pairs.tsv
python src/evaluate.py --split local_val --matching   ../../output/local_val/matching_results.tsv
python tests/test_evaluate.py                          # scorer self-check (includes the official 0.714 example)
```
- **Candidate report:** share of true pairs found, reduction ratio, candidates per S1, and `f05_ceiling` (the best score a perfect matcher could reach on these candidates).
- **Matching report:** F0.5 by country, singleton accuracy, precision/recall, false-positive counts.

**4. Everything in one command** (skips finished stages; see the script header for the decisions baked in):
```bash
bash scripts/run_pipeline.sh                  # local_val (train + score) and test (predict + validate)
SHIFT=-0.5 bash scripts/run_pipeline.sh       # stricter decoding for test
```
Or stage by stage: `translit.py --split local_train` → `blocking.py` → `features.py` → `match.py --cv | --fit | --predict`, then `error_analysis.py --split local_val` for where the loss is.

**5. Before submitting**
```bash
python3 ../../utils/validate_submission.py --matching ../../output/test/matching_results.tsv \
    --candidate ../../output/test/candidate_pairs.tsv --test-dir ../../dataset/test
```

## Running on Kaggle / Colab

Same code; only paths change, through environment variables read by `src/config.py`:

| Variable | Meaning | Default | Kaggle | Colab |
|---|---|---|---|---|
| `BER_DATA_DIR` | raw + split TSVs | `<repo>/dataset` | `/tmp/data` | `/content/data` |
| `BER_OUTPUT_DIR` | submission files | `<repo>/output` | `/kaggle/working/output` | Drive |
| `BER_CACHE_DIR` | features, embeddings, checkpoints | `<repo>/cache` | `/tmp/cache` | `/tmp/cache` |

1. **Pack the data locally**, once. Don't upload the raw TSVs: Drive is slow for large and many files.
   ```bash
   bash scripts/pack_data.sh    # -> <repo>/mlc26_data.tar.zst (698 MB, zstd -19) + .sha256
   ```
2. **Upload** the archive **and** its `.sha256`: as a private Kaggle Dataset `mlc26-data` (main machine: 30 GB RAM, T4 GPU), and/or to `MyDrive/mlc26/` for Colab.
3. **Add a secret** `GH_TOKEN` (a read-only GitHub token for this repo). On Kaggle, also turn on Internet.
4. **Open `notebooks/cloud_runner.ipynb` and run all cells.** It clones the repo, then `scripts/cloud_setup.sh`:
   - copies the archive to the machine's local disk and verifies checksums;
   - installs the requirements while keeping the machine's CUDA torch;
   - regenerates the splits and checks they are **byte-identical** to the local ones.

   For long jobs on Kaggle, use *Save Version*: it keeps running after you close the browser.

## Git workflow

Branch from `main` per task (`feat/<task>`, e.g. `feat/t3-features`), open the PR against `main`, and update the status table in `docs/PIPELINE.md` when it lands. Generated files (`output/`, `cache/`, `dataset/`, `*.tar.zst`) are git-ignored.
