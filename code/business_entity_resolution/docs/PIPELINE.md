# Pipeline: status, contracts and task board

This is the living handoff doc. For background and the history of what was built, read [`REPORT.md`](REPORT.md) first. Then use this doc; update the status table and the submission log whenever you land something.
**Deadline: Sun 27 Sep 2026, 23:59 IST. Code freeze: Sun 16:00 IST** (after that, only the final Kaggle run, validation, docs and the zip).

## 1. Status

| Stage | File | What the problem needs | State | Verified? |
|---|---|---|---|---|
| Data I/O | `data_loader.read_tsv` | Read TSVs exactly (fields contain literal `"`) | done | ✅ 4 fields on every line, checksums |
| Local validation | `data_loader.py` | A split that behaves like test | done: self-contained `local_train` / `local_val` | ✅ exact partition. ❌ local→leaderboard gap unknown (test has 5.75 targets per S1, local has 4.68) |
| Scorer | `evaluate.py` | Leaderboard F0.5 (per-S1 average, singletons included) + blocking metrics | done | ✅ tests in `tests/test_evaluate.py` |
| Cloud | `scripts/`, `notebooks/cloud_runner.ipynb` | Run full-size / GPU jobs | done | ✅ local rehearsal. ❌ not yet run on a real Kaggle machine |
| Normalisation | `normalize.py` | Country-agnostic | v1: unidecode + lowercase + alphanumerics only; rarity weighting does the rest | ⚠️ used by blocking only |
| Blocking | `blocking.py` | High recall, bounded candidates, exact set fed to the model | v1: TF-IDF top-K per pass (addr / full), same country | ✅ `local_val`: 98.2% of true pairs found, F0.5 ceiling 0.994. ❌ test runtime unmeasured |
| Matcher (GBDT) | `features.py`, `match.py` | Precise, calibrated | **TODO** (owner B) | – |
| Cross-encoder | `cross_encoder.py` | MIT/Apache, ≤8B params | **TODO** (owner B/C, Kaggle GPU) | – |
| Assignment + threshold | `match.py` | Each target → ≤1 S1; F0.5-optimal cutoff | **TODO** | – |
| Package | `utils/validate_submission.py`, `Documentation_template.md` | Zip; outputs reproducible from data using only the package | **TODO** (owner C) | – |

Legacy code kept for reference only: `blocking_legacy.py`, `matching.py`, `preprocess.py`.

## 2. Key facts from the data (drive the design)

- Every S2/S3 record matches **at most one** S1 → assign each target only to its best-scoring S1 (a large precision lever).
- Matches **never cross countries** → block within country, comparing country as a string (so France works automatically).
- Per S1: 5.6% have no match (singletons); most have 2–5; the maximum is 11. About 26% of S2/S3 records match nothing (distractors).
- India is hard: 33% of true pairs have name Jaro-Winkler below 0.8; **18% of India target names are in native scripts** (Kannada, Malayalam, …).
- No usable postcodes (0% in S1). Generic names repeat heavily ("primary care" appears 397× in S2), so the address has to decide.
- France: test only, 15% of test S1, with no labels.

## 3. Contracts between stages

The stages communicate only through these files, so each person can work independently.

**Blocking → matcher:** `$BER_CACHE_DIR/<split>/candidates.parquet`, one row per candidate pair:

| column | type | meaning |
|---|---|---|
| `s1_id`, `cand_id` | str | the pair |
| `score_name`, `score_addr`, `score_full` | float32 / NaN | TF-IDF cosine in that pass (NaN = that pass did not retrieve it) |
| `rank_name`, `rank_addr`, `rank_full` | float / NaN | rank within the S1's list for that pass (1 = best) |

These columns are useful matcher features in their own right. **Blocking may add columns but must not rename these.**
Also written: `$BER_OUTPUT_DIR/<split>/candidate_pairs.tsv` (submission format).

**Matcher → submission:** `$BER_OUTPUT_DIR/<split>/matching_results.tsv`. Must be a subset of `candidate_pairs.tsv`; `evaluate.py` warns if not.

**Rule:** every stage runs as `python src/<stage>.py --split {local_val|local_train|test}`. Develop on `local_val` locally; run `local_train` (training data for the matcher) and `test` on Kaggle.

## 4. Blocking results (local_val)

Chosen defaults: `addr` K=20 + `full` K=30, max-df 2%, `name` pass off. Full experiment history in [`REPORT.md`](REPORT.md) §3.

| | ALL | India | US |
|---|---|---|---|
| share of true pairs found | 0.9823 | 0.9647 | 0.9941 |
| entities with every match found | 0.9486 | 0.9019 | 0.9800 |
| **F0.5 ceiling (perfect matcher)** | **0.9937** | 0.9870 | 0.9982 |
| candidates per S1 (mean / max) | 37.9 / 50 | 35.7 / 50 | 39.4 / 50 |

Weak spot: native-script names (~88% of their true pairs found, almost all through the address). Runtime on test is **not yet measured** (about 76× the `local_val` work).

## 5. Decisions (and why)

1. **Cascade matcher: LightGBM on all pairs → cross-encoder on the uncertain band → stacked LightGBM.**
   - Cross-encoders are the state of the art for pairwise entity matching, but a transformer over roughly 40M test pairs on a T4 is about 10h, beyond our one-session budget.
   - The GBDT scores everything cheaply and decides which pairs are uncertain. The cross-encoder's probability then becomes one more feature, so rank and assignment logic stays in one model.
   - The cross-encoder must handle native scripts: use `microsoft/mdeberta-v3-base` (MIT, multilingual), or transliterate first and use `deberta-v3-base` (MIT).
   - Adopt it only if `local_val` F0.5 improves by ≥0.005 over the GBDT alone.
2. **Threshold:** tuned on `local_val`, then set slightly **stricter** for test, because test has more distractors per S1. Submissions 1–2 measure the gap.
3. **France probe:** one submission with France predictions blanked vs. one filled, to measure how France performs on the public leaderboard.
4. **Reproducibility:** the package must regenerate both outputs from the provided data using only the package, so all training code ships. Public pretrained checkpoints (MIT/Apache) are downloaded at run time.

## 6. Task board (pick one, put your name on it, branch `feat/<task>`)

| # | Task | Owner | Depends on | Done when |
|---|---|---|---|---|
| T1 | Run `cloud_runner.ipynb` on Kaggle end-to-end (dataset `mlc26-data`, `GH_TOKEN` secret, Internet on) | | – | setup prints "splits identical" |
| T2 | Run `blocking.py --split local_train` and `--split test` on Kaggle; save `candidates.parquet` as Kaggle output / Dataset | | T1 | files exist; runtime + peak RAM noted here |
| T3 | `features.py`: RapidFuzz name/address similarities (ratio, token_set, partial, Jaro-Winkler), number-token overlap, legal-form agreement, blocking scores and ranks, **within-S1 relative features** (rank, gap to best, number of candidates) | | contract §3 | unit test on 10 pairs |
| T4 | `match.py`: LightGBM trained on `local_train` candidates → scores → assign each target to its best S1 → tune threshold on `local_val` → `matching_results.tsv` | | T2, T3 | `evaluate.py --split local_val` result logged |
| T5 | Submission 1 + threshold-probe Submission 2 on test | | T4 | leaderboard scores logged in §7 |
| T6 | `cross_encoder.py`: fine-tune on `local_train` hard negatives (Kaggle GPU); score the uncertain band; stack into T4 | | T4 | passes the ≥0.005 gate or is dropped |
| T7 | Error analysis on `local_val` by bucket (native script, name change, singleton false positives) → fix the biggest | | T4 | before/after numbers |
| T8 | Final: clean-clone Kaggle run on test, `validate_submission.py`, `Documentation_template.md`, zip | | freeze | zip uploaded |

## 7. Submission log

| # | Time (IST) | Commit | Change | local_val F0.5 | Public LB | Notes |
|---|---|---|---|---|---|---|
