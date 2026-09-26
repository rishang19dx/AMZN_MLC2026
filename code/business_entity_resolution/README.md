# Business Entity Resolution Pipeline (MLC 2026)

For each Source 1 (S1) business, find every matching S2/S3 record. Scored by F0.5, computed per S1 record and averaged, with no-match records included.
**Deadline: Sun 27 Sep 2026, 23:59 IST. Code freeze: Sun 16:00 IST.**

**Teammates:** read [`docs/REPORT.md`](docs/REPORT.md) (background, history, experiments, what is verified), then pick a task from [`docs/PIPELINE.md`](docs/PIPELINE.md) (status, stage contracts, task board, submission log).

For the architecture, data splits, models and leaderboard results, see the [main README](../../README.md). Live status and the submission log: [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pinned; for CPU-only torch, first:
                                       # pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Put the raw data in `../../dataset/{train,test}/`, or extract `mlc26_data.tar.zst` there.
Read challenge TSVs with `data_loader.read_tsv` (tab-separated, quoting disabled, every value a string). Values containing `"` are CSV-escaped (`"""ehpad Club SAS"`); reading them raw keeps a few stray quotes, which normalisation strips.

## Running (all commands from this directory)

Every stage takes `--split {local_val|local_train|scale_val|ce_train|train|test}` (what each split is for: main README, *Data splits*).

**1. Local splits** (~1 min) → `../../dataset/splits/{local_train,local_val}/`
```bash
python src/data_loader.py
```
- `local_val` = 10% of S1 (by seeded hash) + every S2/S3 record matched to them + 10% of the records that match nothing.
- `local_train` = everything else.
- Each is self-contained, like the test set. Scoring held-out S1 against *all* train S2/S3 would be misleading: 90% of the targets would belong to S1 records outside the split.

Test-scale helper splits (both use all `local_train` targets through symlinks):
```bash
python src/data_loader.py --scale-val    # scale_val: 20% of local_train Source 1 -> matcher training at test scale
python src/data_loader.py --ce-train     # ce_train: 5% of local_train Source 1 -> cross-encoder training
```

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

**Memory (15 GB laptop, test scale: 11.7M records, 65M pairs).** Measured: test features 26 min / 9.9 GB peak; test predict 36 min / 11.6 GB; test blocking 4.4 h.
```bash
python src/features.py --split test --index-only            # candidate index in its own process (37 s, 6.8 GB), cached
BER_FEATURE_CHUNK=300000 python src/features.py --split test  # smaller parts = lower peak; memory is printed on every log line
python src/match.py --split test --predict                    # also saves cache/test/pred.npz
python src/match.py --split test --redecode --shift=-0.5      # decoding variant in ~6 min -> matching_results_shift-0.50.tsv
```
Launch long jobs detached (`setsid nohup ... &`) so they survive a closed terminal; `scripts/mem_guard.sh` stops the pipeline cleanly before the machine runs out of memory.

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
