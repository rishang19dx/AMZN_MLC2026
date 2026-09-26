# Plan to raise the leaderboard score (Sat 26 Sep 17:50 → freeze Sun 16:00)

Status: **agreed plan**. Context and measurements: `PIPELINE.md` §4b (leaderboard gap), `FINDINGS.md` (data), `ENSEMBLE.md` (ensemble design).

## 1. What we know

| Signal | Value |
|---|---|
| Public LB, submission 1 | 0.9334 (local cross-validated F0.5 on `local_val`: 0.9858) |
| Derived from the France-blank probe | US/India ≈ **0.944**, France ≈ **0.874** |
| Pairs in the uncertain band (p between 0.02 and 0.98) | test **4.55% (2.97M)** vs `local_val` 1.06% (88k) |
| Feature medians, test vs `local_val` | address TF-IDF cosine 0.46 vs 0.32; RapidFuzz address token-set 0.72 vs 0.57 (IDF-independent) |

**Root cause:** the matcher was trained on `local_val`, a universe ~9× smaller than test. Test candidates are much closer look-alikes, so the model is far less certain there (4.3× more uncertain pairs) and loses about 0.04 on US/India. France adds about 0.01 more.

## 2. Levers, ranked by expected gain per hour

Gains are estimates until measured.

| # | Change | Why | Expected gain | Cost |
|---|---|---|---|---|
| 1 | **Retrain stages 1 + 2 on `scale_val`** (377k Source 1 against all 9.29M `local_train` targets) | Fixes the root cause: test-sized pool, IDF and look-alikes | **+0.01 to +0.03** | ~3.5 h CPU (Sat evening) |
| 2 | **Decoding tuned at scale**: shift chosen on `scale_val` cross-validation; leaderboard probes shift −0.5 / −1.0 | Calibration under test crowding | +0.002 to +0.01 | minutes |
| 3 | **mDeBERTa cross-encoder** on the uncertain band, as a stage-2 feature, with a **listwise loss per Source 1** | Complements string features: transformer + fuzzy hybrid gained 4.6 F1 in [arXiv 2509.17470](https://arxiv.org/html/2509.17470v1); listwise/pairwise objectives beat pointwise ones by about one backbone-size tier, and negative quality mattered as much as the loss ([arXiv 2603.03010](https://arxiv.org/abs/2603.03010)) | +0.003 to +0.01 | Kaggle T4: ~1 h train + ~35 min scoring |
| 4 | **Crowding features**, chosen by error analysis at scale: house number + street match, name specificity × address similarity, unit/suite, city match, S2↔S3 sibling agreement | The errors that grow with crowding | +0.002 to +0.008 | 2–3 h CPU (Sun morning) |
| 5 | **Ensemble** (`ENSEMBLE.md`): 5-seed LightGBM bagging; cross-encoder stacked in stage 2; mmBERT only if GPU time is left | Small but stable gains (+0.2–0.5 F1 in the literature) | +0.0005 to +0.002 | cheap CPU |
| 6 | **France**: French abbreviations (`R.`/`AV`/`BD` → rue/avenue/boulevard) and legal forms (SARL/SAS/EURL/SCI); stricter shift for countries not seen in training | France ≈ 0.874 on 15% of entities (reaching US/India level would add ~+0.01) | +0.003 to +0.008 | ~1.5 h |
| 7 | **Blocking recall at scale**: measure the ceiling on `scale_val`; raise K only if it dropped a lot | Recall caps everything | measured tonight | free to measure |

## 3. Decisions (Sat 17:55)

| Question | Decision |
|---|---|
| Where the cross-encoder runs | **Kaggle T4, run by a teammate** (`notebooks/kaggle_cross_encoder.ipynb`). The laptop's RTX 2050 is impractical: CPU-only torch, a CUDA install needs ~3–5 GB while 2.6 GB of disk is free, and it is ~4× slower than a T4 |
| Cross-encoder loss | **Listwise per Source 1** (`cross_encoder.py --loss listwise`, the default): BCE + −log(Σ_pos e^s / Σ_all e^s) over each Source 1 group of all its positives + its hardest negatives by p1 (max 16). `--loss bce` is kept for comparison |
| Today's 3 remaining submissions | shift −0.5, shift −1.0, then the `scale_val`-trained model (~21:15) |
| Parallel work | Teammate: Kaggle cross-encoder run. Rishang + Claude: `scale_val`, features, France |

## 4. Order of work

**Rule for the combiner:** stage 2 must be retrained **on `scale_val`** with cross-encoder scores for `scale_val`'s uncertain band. This stays out-of-sample because `scale_val` excludes `ce_train`'s Source 1 by construction.

| When | Step | Where |
|---|---|---|
| Sat ~18:45 | `scale_val` blocking done → blocking ceiling at scale (lever 7) | laptop |
| Sat ~19:15 | `scale_val` features | laptop |
| Sat ~20:00 | `match.py --split scale_val --cv` → test-scale local score, decoding choice, **error analysis** (lever 4 input) | laptop |
| Sat ~21:15 | `--fit` on `scale_val` → test `--predict` → **submission 5** | laptop |
| Sat night | `ce_train` blocking + features; `cross_encoder.py export` for `ce_train`, `scale_val`, test (band from the new stage 1) | laptop |
| Sat night / Sun early | `train` (listwise) + `score scale_val` + `score test` | **Kaggle T4 (teammate)** |
| Sun morning | stage 2 retrained with `ce` on `scale_val` → test predict → submission; France normalisation; crowding features | laptop |
| Sun 14:00 | Final choice: `scale_val` cross-validation first, leaderboard second (avoid fitting the public subset) | – |
| Sun 16:00 | Freeze: clean run, validator, methodology document, zip (check `candidate_pairs.tsv` against the 512 MB limit once zipped) | laptop |

## 5. Adoption rule

One change per submission. A change is kept only if the public leaderboard improves; for changes that cannot be measured on the leaderboard in time, `scale_val` cross-validation decides. Threshold sweeps stay on `scale_val`, never on the public board.
