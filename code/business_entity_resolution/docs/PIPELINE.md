# Pipeline: status, contracts and task board

This is the living handoff doc. For background and the history of what was built, read [`REPORT.md`](REPORT.md) first; for measured data facts, [`FINDINGS.md`](FINDINGS.md). Then use this doc; update the status table and the submission log whenever you land something.
**Deadline: Sun 27 Sep 2026, 23:59 IST. Code freeze: Sun 16:00 IST** (after that, only the final run, validation, docs and the zip).

**One command runs everything:** `bash scripts/run_pipeline.sh` (splits → dictionary → blocking → features → matcher → validator; finished stages are skipped).

## 1. Status (Sat 26 Sep, ~17:45 IST)

| Stage | File | What the problem needs | State | Verified? |
|---|---|---|---|---|
| Data I/O | `data_loader.read_tsv` | Read TSVs exactly (fields contain literal `"`) | done | ✅ 4 fields on every line, checksums |
| Local validation | `data_loader.py` | A split that behaves like test | done: self-contained `local_train` / `local_val` | ✅ exact partition. ❌ local→leaderboard gap unknown (test has 5.75 targets per S1, local has 4.68) |
| Scorer | `evaluate.py` | Leaderboard F0.5 (per-S1 average, singletons included) + blocking metrics | done | ✅ `tests/test_evaluate.py` |
| Cloud | `scripts/`, `notebooks/cloud_runner.ipynb` | Run full-size / GPU jobs | done; notebook now runs `run_pipeline.sh` | ✅ local rehearsal. ❌ not yet run on a real Kaggle machine |
| Normalisation | `normalize.py`, `translit.py` | Country-agnostic; native scripts | **v2:** learned native-script dictionary (from `local_train` only) + anyascii (ISC; Unidecode dropped, GPL-2) | ✅ dictionary covers 98.4% of `local_val` native-script words |
| Blocking | `blocking.py` | High recall, bounded candidates, exact set fed to the model | v1 passes (addr K=20 + full K=30); now writes per-(country, pass) parts and merges them on disk with DuckDB, so test fits in memory | ✅ `local_val`: **98.8%** of true pairs found, F0.5 ceiling **0.996**. ❌ test runtime not measured yet |
| Features | `features.py` | Pair evidence | 37 features, written in parts keyed by integer row indices. **Test-scale memory fixes** (streaming record loader, streaming TF-IDF, cached candidate index `--index-only`, compact number sets); memory printed on every log line | ✅ `local_val`: identical to before (max diff 4.8e-07). ✅ **test: 65.2M pairs, 26 min, 9.9 GB peak** on the 15 GB laptop |
| Matcher | `match.py` | Precise, calibrated | two-stage LightGBM with list and competition context. `--predict` saves `cache/<split>/pred.npz`; `--redecode --shift=X` rebuilds a decoding variant in ~6 min | ✅ cross-fit on `local_val`: **F0.5 0.9858**. ⚠️ **public LB 0.9334**: the model is trained on a 9× smaller universe than test (§4b). ✅ test predict: 36 min, 11.6 GB peak |
| Assignment + decoding | `match.py` | Each target → ≤1 S1; F0.5-optimal set per S1 | one S1 per target, per-source caps, expected-F0.5 decoding | ✅ `tests/test_match.py`; measured in §4 |
| Cross-encoder | `cross_encoder.py`, `notebooks/kaggle_cross_encoder.ipynb` | MIT/Apache, ≤8B params | mDeBERTa-v3-base trained on the `ce_train` split (disjoint from `local_val` and test), band-only scoring, auto-used by `match.py` stage 2 when `ce_scores.parquet` exists | ✅ train/score smoke-tested on CPU with a tiny model. ❌ not yet trained on the T4 |
| Test-scale split | `data_loader.py --scale-val` | Train/score the matcher with test-like crowding | `scale_val`: 377,423 `local_train` Source 1 (not `ce_train`'s) against **all 9.29M** `local_train` targets | ⏳ blocking running (Sat 17:27) |
| Package | `scripts/run_pipeline.sh`, `utils/validate_submission.py`, `Documentation_template.md` | Zip; outputs reproducible from data using only the package | end-to-end script done | ✅ full chain + validator **PASS** on a small test sample. ❌ not yet on the full test set; docs and zip TODO |

Legacy code (`blocking_legacy.py`, `matching.py`, `preprocess.py`) was removed on Sat 26 Sep; it is in git history.

## 2. Key facts from the data (drive the design)

- Every S2/S3 record matches **at most one** S1 → assign each target only to its best-scoring S1.
- Matches **never cross countries** → block within country, comparing country as a string (so France works automatically).
- Per S1: 5.6% have no match (singletons); most have 2–5; the maximum is 11. About 26% of S2/S3 records match nothing (distractors).
- India is hard: 33% of true pairs have name Jaro-Winkler below 0.8; **18% of India target names are in native scripts**, from a closed vocabulary of about 1,500 words.
- No usable postcodes. Generic names repeat heavily ("primary care" appears 397× in S2), so the address has to decide.
- France: test only, 15% of test S1, with no labels.
- More in [`FINDINGS.md`](FINDINGS.md).

## 3. Contracts between stages

The stages communicate only through these files, so each person can work independently. Every stage runs as `python src/<stage>.py --split {local_val|local_train|test}`.

| File | Written by | Contents |
|---|---|---|
| `$BER_CACHE_DIR/translit.json` | `translit.py` | `{"built_from": split, "words": {native word: latin word}}`; read by `normalize.norm` |
| `$BER_CACHE_DIR/<split>/candidates.parquet` | `blocking.py` | one row per pair: `s1_id, cand_id, score_{name,addr,full}, rank_{name,addr,full}` (NaN = that pass did not retrieve it). Add columns, never rename |
| `$BER_OUTPUT_DIR/<split>/candidate_pairs.tsv` | `blocking.py` | submission format, every S1 has a row |
| `$BER_CACHE_DIR/<split>/features/part-*.parquet` | `features.py` | `s1_id, cand_id, s1_idx, tg_idx` (row indices into the split's S1 file and S2+S3 files, in that order), the features, and `label` when ground truth exists. Each S1's list stays within one part |
| `$BER_CACHE_DIR/models/{stage1,stage2}.txt, features.json, decode.json` | `match.py --fit` / `--cv` | LightGBM models, feature order, decoding choice |
| `$BER_OUTPUT_DIR/<split>/matching_results.tsv` | `match.py` | submission format; a subset of `candidate_pairs.tsv` |

## 4. Results on `local_val`

### Blocking (addr K=20 + full K=30, max-df 2%)

| | Unidecode (v1) | dictionary + anyascii (v2) |
|---|---|---|
| share of true pairs found | 0.9823 | **0.9880** |
| India | 0.9647 | **0.9788** |
| native-script names | 0.8847 | **0.9779** |
| F0.5 ceiling (perfect matcher) | 0.9937 | **0.9961** |
| candidates per S1 | 37.9 | 37.7 |
| wall time / peak RAM (16 cores) | – | 279 s / 3.0 GB |

### Matcher (2-fold cross-fit grouped by S1; every prediction is out-of-fold)

Baseline v1 (Unidecode features): **F0.5 = 0.9827** (India 0.9752, US 0.9877).

| Component | F0.5 | What it adds |
|---|---|---|
| stage 1 only (pair features) + expected-F decoding | 0.97999 | – |
| + stage 2 (list and competition context) | **0.98265** | **+0.0027**. `t_margin` (margin over the best competing S1 for the target) carries 71% of stage-2 gain |
| best global threshold instead of expected-F decoding (t = 0.7) | 0.98233 | expected-F decoding adds +0.0003 |
| without one-S1-per-target and caps | 0.98256 | +0.0001 (stage 2 already learned the competition; kept as a free guarantee) |

- Where the loss is: precision 0.996, recall 0.960. 30.9k missed pairs vs 3.1k false matches; ~13.6k of the misses were never retrieved by blocking.
- Calibration is excellent: predicted probability matches the observed match rate within ~0.03 in every bin, which is what expected-F decoding needs.

**v2 (dictionary + anyascii features): F0.5 = 0.9858** (India **0.9835**, US 0.9874), **+0.0031** over v1.

| | v1 (Unidecode) | v2 (dictionary) |
|---|---|---|
| F0.5 ALL / India / US | 0.9827 / 0.9752 / 0.9877 | **0.9858 / 0.9835 / 0.9874** |
| precision / recall (averaged per entity) | 0.9963 / 0.9604 | 0.9965 / **0.9675** |
| singleton accuracy | 0.9775 | **0.9809** |
| missed pairs / false matches | 30,868 / 3,084 | **25,370 / 2,863** |
| stage 1 only → + stage 2 context | 0.9800 → 0.9827 | 0.9836 → 0.9858 |
| best threshold vs expected-F decoding | 0.98233 vs 0.98265 | 0.98563 vs **0.98582** |

The India gain (+0.0083) is the native-script dictionary working as intended; the US change (−0.0003) is within noise (anyascii vs Unidecode on accented Latin). Decoding choice for test: expected-F0.5, no shift (`cache/models/decode.json`).

### 4b. Test and leaderboard (Sat 26 Sep)

| | Public LB | Derived |
|---|---|---|
| Submission 1 (all countries) | **0.9334** | – |
| Submission 2 (France rows blank) | 0.8100 | US/India ≈ **0.944** (0.942–0.946), France ≈ **0.874** (0.864–0.884) |

Derivation: France is 15.0% of test Source 1; a blank France list scores ~its singleton rate (4–6%). So US/India lose ~0.042 vs `local_val` (0.986) and cost ~0.036 of the 0.052 gap; France costs ~0.010. **The gap is mostly not France.**

Why: `local_val` is a ~9× smaller universe than test, so test candidates are far closer look-alikes, and the model (trained on `local_val`) sees test negatives that look like its training positives. Feature medians, `local_val` → test sample:

| Feature | `local_val` | test | ratio |
|---|---|---|---|
| address TF-IDF cosine | 0.32 | 0.46 | 1.43 |
| name char-3-gram cosine | 0.046 | 0.069 | 1.51 |
| blocking score (full pass) | 0.28 | 0.38 | 1.37 |
| RapidFuzz address token-set (IDF-independent) | 0.57 | 0.72 | 1.28 |

The IDF-independent row shows it is genuine crowding, not only IDF drift. Fix in progress: retrain and score on `scale_val` (test-sized target pool). The `local_val` 0.9858 remains an honest number for a small universe, not a leaderboard estimate.

## 5. Decisions (and why)

1. **Cascade matcher:** LightGBM on all pairs → (cross-encoder on the uncertain band →) stacked LightGBM.
   - A pair classifier is the right *scoring* step (Ditto), but candidates compete, so stage 2 sees each S1's list and every target's competing S1s: the cheap version of the "select" strategy (Wang et al., COLING 2025).
   - Cross-encoder: `microsoft/mdeberta-v3-base` (MIT, multilingual). Its score becomes a feature; adopt only if `local_val` F0.5 improves by ≥0.005.
2. **Decoding:** per S1, choose k (0 = no match) maximising expected F0.5 (Ye et al. ICML 2012; Waegeman et al. JMLR 2014). A logit shift `--shift` (negative = stricter) is the knob for test, which has more distractors per S1.
3. **One S1 per target:** per-target argmax is the exact optimum (only the target side is constrained).
4. **Native-script dictionary learned from `local_train` only, used for every split.** Keeps the `local_val` score honest (no label leakage) and train/test features consistent; the coverage given up vs. a dictionary built from all of train is small (98.4% already).
5. **Matcher trained on `local_val` candidates** (8.3M pairs) — **revised Sat 17:30:** the leaderboard gap (§4b) comes from scale, so the matcher moves to `scale_val` (Source 1 sample against the full `local_train` target pool: test-like crowding at ~4 h of work instead of ~9 h for all of `local_train`, which also would not fit 15 GB for training).
6. **Country is not a feature** (France is unseen; a country-keyed model would route it arbitrarily). Source (S2/S3) is.
7. **anyascii instead of Unidecode** (licence: ISC vs GPL-2; also transliterates Indic scripts better).
8. **France probe:** one submission with France blanked vs. one filled.
9. **Reproducibility:** all training code ships; `run_pipeline.sh` regenerates both outputs from the data.

## 6. Task board (pick one, put your name on it, branch `feat/<task>`)

| # | Task | Owner | Depends on | State |
|---|---|---|---|---|
| T1 | Run `cloud_runner.ipynb` on Kaggle (dataset `mlc26-data`, `GH_TOKEN` secret, Internet on) | | – | open; needed for the cross-encoder |
| T2 | Blocking + features + predict on **test** | Rishang (local) | – | ✅ done (features 26 min / 9.9 GB; predict 36 min) |
| T3 | `features.py` | Rishang | – | ✅ done |
| T4 | `match.py` (two-stage LightGBM, assignment, expected-F decoding) | Rishang | – | ✅ done |
| T5 | Submission 1 + threshold/shift probe Submission 2 | Rishang | T2 | ✅ sub 1 = 0.9334; shift −0.5 / −1.0 files ready (`output/test/matching_results_shift-*.tsv`) |
| T6 | `cross_encoder.py`: fine-tune mDeBERTa-v3 on hard negatives (Kaggle T4), score the uncertain band, add as a stage-2 feature | | T1 | open |
| T7 | Error analysis on `local_val` out-of-fold predictions (`cache/local_val/oof_p2.npy`, row order = feature parts): missed pairs vs. false matches by bucket | | – | open |
| T8 | Final: clean run, `validate_submission.py`, `Documentation_template.md`, zip | | freeze | open |
| T9 | France probe submissions | Rishang | T5 | ✅ France ≈ 0.874, US/India ≈ 0.944 (§4b) |
| T10 | `scale_val`: blocking → features → `match.py --cv` → `--fit` → test `--predict` | Rishang (local) | – | ⏳ blocking running |

## 7. Submission log

| # | Time (IST) | Commit | Change | local_val F0.5 | Public LB | Notes |
|---|---|---|---|---|---|---|
| 1 | Sat 12:25 | `168e075` | v2 matcher trained on `local_val`, expected-F decoding, shift 0 | 0.9858 | **0.933414** | 5,900,990 matches; 4.7% empty lists |
| 2 | Sat 12:40 | `168e075` | Submission 1 with all France rows blank (probe) | – | 0.809967 | → France ≈ 0.874, US/India ≈ 0.944 |
| 3 | pending | `168e075` | shift −0.5 (`matching_results_shift-0.50.tsv`) | – | | 5,808,651 matches |
| 4 | pending | `168e075` | shift −1.0 (`matching_results_shift-1.00.tsv`) | – | | 5,734,256 matches |
