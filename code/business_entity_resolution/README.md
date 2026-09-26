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
| 5 | Matcher v1 (features + LightGBM + assignment + threshold) | `src/features.py`, `src/match.py` | ⏳ TODO | – |
| 6 | Cross-encoder (mDeBERTa-v3, stacked into the matcher) | `src/cross_encoder.py` | ⏳ TODO | – |
| 7 | Submission package (validator, docs, zip) | – | ⏳ TODO | – |

No leaderboard submission yet. `matching.py`, `preprocess.py` and `blocking_legacy.py` are **legacy**, kept for reference only; don't build on them.

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

## Multi-pipeline blocker (new, not yet run on real data)

`src/blocker/` adds three independent candidate generators whose lists are unioned: (A) a fine-tuned BERT bi-encoder (multilingual-e5-small, MIT), (B) a JEPA-style predictive encoder (multilingual MiniLM, Apache-2.0), and (C) TF-IDF passes plus blocking keys (reusing `blocking.py`). It comes with FAISS retrieval, a prioritised budget, a blocking checker with penalties (`src/check_blocking.py`), and a basic LightGBM matcher (`src/lgbm_matcher.py`). Everything is configured in `configs/blocking.yaml`, and `bash scripts/run_multiblock.sh` runs it end to end. Full description, flags (including `--train-fraction`), metrics and compute estimates: [`docs/BLOCKING.md`](docs/BLOCKING.md).

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

**4. Matching:** not implemented yet (tasks T3/T4/T6 in `docs/PIPELINE.md`).

**5. Before submitting**
```bash
python3 ../../utils/validate_submission.py --matching ../../output/test/matching_results.tsv \
    --candidate ../../output/test/candidate_pairs.tsv --test-dir ../../dataset/test
```

## Running on a Mac (Apple Silicon, 16 GB)

One command runs the whole pipeline, from splits to a validated `output/test/matching_results.tsv`:
```bash
bash scripts/mac_run.sh          # creates .venv from requirements-mac.txt if missing; rerun to resume
STAGES="fit eval" bash scripts/mac_run.sh      # a subset; FORCE=1 redoes finished stages
```
- **Data:** expects the raw TSVs in `<ML_Channel>/student_resource/dataset/{train,test}`; override with `BER_DATA_DIR`.
- **Training split:** `local_fit`, 15% of train Source 1 (`BER_FIT_FRACTION`). It is built like `local_val` (a closed universe), is a subset of `local_train`, and is disjoint from `local_val`. All of `local_train` (~75M pairs) does not fit in 16 GB.
- **Training:** `match.py --fit --folds 5` runs a 5-fold grouped cross-fit (grouped by Source 1) of both LightGBM stages. The 5 fold models of each stage are the final ensemble; their probabilities are averaged at inference.
- **Holdout:** `match.py --eval --split local_val` scores the saved ensemble on the untouched `local_val` and tunes `decode.json` there.
- **Saved weights:** `cache/models/stage{1,2}_fold{0-4}.txt`, `features.json` and `decode.json`; each fold is saved as soon as it is trained. The native-script dictionaries are `cache/translit_{local_train,train}.json`. `--predict` needs only these files.
- **Machine settings:** the script uses all cores (`BER_THREADS`, default `hw.ncpu`), keeps the Mac awake with `caffeinate`, and logs each stage's wall time and peak RAM to `logs/`. Blocking sizes its sparse-product chunks from the target count (`BER_BLOCK_BUDGET`) so the forked workers fit in 16 GB.
- **CPU only:** the active pipeline (sparse TF-IDF, RapidFuzz, LightGBM) runs on the CPU; the GPU/MPS is unused.

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
