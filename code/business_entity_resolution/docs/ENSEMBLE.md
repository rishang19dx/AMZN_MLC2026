# Ensemble design (to implement on top of `match.py` v1)

> **Revisions, 26 Sep 01:15 IST** (decided with Rishang; they override the sections below where they conflict):
> 1. **Hardware:** this laptop (16 cores, 15 GB RAM) + one **Kaggle T4**. There is no 24 GB GPU, so §6's estimates are ~3–4× too optimistic for the GPU pieces.
> 2. **One cross-encoder, not A/B** (§3.2). It is trained on the separate `ce_train` split (5% of `local_train` Source 1 against all `local_train` targets; `data_loader.py --ce-train`). Its scores on `local_val` and test are therefore out-of-sample by construction, so rules 1–2 hold with half the GPU time. Implemented in `src/cross_encoder.py` (export / train / score) and wired into `match.py` stage 2 (`ce`, `ce_s1_rank`, `ce_s1_gap`, `ce_t_margin`). The band rule uses out-of-fold `p1` on `local_val` (`oof_p1.npy`) and the final stage-1 model on test.
> 3. **Qwen3-4B judge (§3.5, §4.5): only if time remains** after submissions 1–4.
> 4. **Adoption rule (§5):** adopt a step when the public leaderboard improves **and** the out-of-fold `local_val` F0.5 does not fall. Leaderboard changes below 0.001 count as ties, broken by `local_val`. This guards against overfitting the public subset, since the private leaderboard decides the ranking.
> 5. **Owner:** Claude builds submission 1, the cross-encoder, the noisy-channel matcher and the error analysis.

Status: **agreed design, not yet implemented** (26 Sep 2026). `match.py` v1 already provides stage 1, stage 2, the one-Source-1-per-target assignment and expected-F0.5 decoding. This document specifies what the other members add, how they plug in, and the rules that keep every score honest. Evidence for each choice is in §7; broader background is in `FINDINGS.md`.

## 1. Structure

```
                 BASE MEMBERS (level 0)                     COMBINER (level 1)            DECISION
 ┌──────────────────────────────────────────────┐
 │ Stage 1 LightGBM on pair features            │── p1 ──┐
 │   (features.py; exists in v1)                │        │
 │ Cross-encoder A (trained on fold 0)          │─┐      │   Stage 2 LightGBM           Calibrate (isotonic,
 │ Cross-encoder B (trained on fold 1)          │─┴ ce ──┼─► = v1 context features  ──►  fitted on OOF p2)
 │ [mmBERT-small, only if the pilot is close]   │── ce2 ─┤   + ce, ce2, nc                    │
 │ Noisy-channel matcher (CPU)                  │── nc ──┘   (learned weights,               ▼
 └──────────────────────────────────────────────┘            never averaging)      One Source 1 per target
                                                                                   + per-source caps (v1)
                                                                                           │
 Qwen3-4B judge (zero-shot; only unseen-country pairs in the uncertain band) ─────►  France veto (§4.5)
                                                                                           │
                                                                                           ▼
                                                                                   Expected-F0.5 decoding (v1)
```

## 2. The rules

1. **One fold assignment for everything.** Every member that is trained on labels uses `match.py`'s rule: `fold = int(md5("mlc26-cv:" + s1_id)[:8], 16) % 2`, i.e. `Pairs.row_fold(split, 2, 'mlc26-cv')`. All pairs of a Source 1 entity, and therefore all of its targets, stay in one fold.
2. **Out-of-fold scores only.** A member's score for a training row must come from a model that did not train on that row's fold. The combiner (stage 2) only ever sees out-of-fold member scores, exactly as v1 already does for `p1`.
3. **Learned weights, never equal-weight averages** between *different kinds* of members. On `local_val`, equal-weight averaging of two blocking scores dropped AUC from 0.986 to 0.969 (§7).
4. **Calibrate after combining.** Averaging calibrated models makes them under-confident, and decoding needs calibrated probabilities.
5. **Train-only resources.** Word tables used by members (native-script dictionary, variant table, noisy-channel edit probabilities) are learned from `local_train`, which is disjoint from `local_val` and test.
6. **Country is never a feature** (as in `features.py`). France-specific behaviour is keyed on "country not present in the training split's Source 1", never on the string `France`.
7. **Parameter budget ≤ 8B in total**: two mDeBERTa (~0.28B each) + optional mmBERT-small (0.14B) + Qwen3-4B ≈ 4.7B.

**Training split.** Rules 1–2 apply to whichever labelled split `match.py --split` trains on. v1 uses `local_val` (8.4M candidate pairs; honest numbers come from its 2-fold cross-fit). Moving training to a subsample of `local_train`, with `local_val` as an untouched hold-out, is an open option (§8).

## 3. Members

### 3.1 Stage 1 LightGBM (exists)
- `match.py` stage 1 on `features.py` pair features; out-of-fold `p1` from the 2-fold cross-fit.
- **Upgrade (variance reduction):** at `--fit`, train **5 seed-bagged models** on all rows (different `seed`, `bagging_fraction≈0.8`, `feature_fraction≈0.8`) and average their predictions. This uses seeds, not more folds, so rule 1 keeps a single fold scheme.

### 3.2 Cross-encoders A and B (2-fold cross-fitting)
- **Model:** the pilot winner, expected `microsoft/mdeberta-v3-base` (MIT).
  - Input: `name | address` for each record, `max_length=128`.
  - Precision: fp16, never bf16 (DeBERTa produces NaN in bf16).
  - One epoch.
  - Training order: randomly swap which record comes first.
  - Averaged (EMA/SWA) weights.
- **Training data:** CE-A trains on fold-0 pairs and CE-B on fold-1 pairs. Each uses ~1–2M pairs sampled from its fold, weighted toward stage-1 uncertainty, and keeps plenty of native-script and hard negatives from blocking.
- **Scores:**
  - Training rows: `ce` = CE-B's score on fold-0 rows and CE-A's score on fold-1 rows (out-of-fold).
  - Test rows: `ce` = mean of CE-A and CE-B **in logit space** (limits under-confidence), converted back to a probability.
  - Check: the distribution of stage-2 predictions on test should look like the training distribution. If the averaged score shifts it noticeably, use a single cross-encoder for test.
- **Which rows get scored:** the uncertain band, `0.02 ≤ p1 ≤ 0.98` (tune from the stage-1 calibration table). Rows outside the band get `ce = NaN`, which LightGBM handles natively. The same band rule must be applied at training and test time.
- **Optional scoring in both orders:** score (S1, target) and (target, S1) and average the logits. This doubles band inference; adopt only if it helps on `local_val`.
- **Output contract:** `$BER_CACHE_DIR/<split>/ce_scores.parquet` with columns `s1_idx, tg_idx, ce` (+ `ce2` if 3.3 is adopted), keyed like `features.py`.

### 3.3 Third cross-encoder (conditional)
- `jhu-clsp/mmBERT-small` (MIT), same 2-fold cross-fitting, band only.
- Add only if the pilot puts it within **0.002 AUC** of the chosen model. Its value is diversity: a different tokenizer and pre-training, mostly for France.

### 3.4 Noisy-channel matcher (bet 1, CPU)
- Score: `nc = log P(target | S1) − log P(target | background)`. P(target | S1) is estimated from learned word-level operations: native-script dictionary, variant table (`Ave↔Avenue`, `OH↔Ohio`, `11th↔Eleventh`), a character typo model (`0↔o`, `l↔i`), inserted numbers and units, repeated words and bracketed legal forms.
- Fitted on `local_train` true pairs only (rule 5), so it is out-of-sample for `local_val` and test. No cross-fitting needed.
- Output: an `nc` column added to `features.py` output, or `$BER_CACHE_DIR/<split>/nc_scores.parquet` keyed by `s1_idx, tg_idx`.

### 3.5 Qwen3-4B judge (bet 2; France veto only)
- `Qwen/Qwen3-4B` (Apache 2.0), zero-shot, via vLLM on the 24 GB GPU.
- Prompt: the two records plus "Same business? Answer Yes or No"; read the Yes/No token probabilities.
- **Validate before use:** run it zero-shot on the US/India uncertain band of `local_val` and report precision and recall of "Yes" against labels. No training on any split.
- Runs only on unseen-country pairs inside the uncertain band (~50–100k pairs).

## 4. Combining and deciding

1. **Stage 2 (combiner):** v1's stage 2 plus the new columns `ce`, `ce2`, `nc`. Also add context features computed from `ce` the same way v1 builds them from `p1`: the cross-encoder rank within the Source 1 list and the margin over the runner-up for the target. Trained with the 2-fold cross-fit, giving out-of-fold `p2`.
2. **Calibration:** fit isotonic regression on out-of-fold `p2`, save it with the models, and apply it to test `p2`. Keep v1's `logit_shift` sweep as the decoding parameter.
3. **Assignment and caps:** unchanged from v1 (per-target argmax; S2 ≤ 5, S3 ≤ 6 per entity).
4. **Decoding:** unchanged from v1 (expected F0.5, `k = 0` catches singletons).
5. **France veto:** for pairs whose Source 1 country is not in the training split, and whose calibrated probability lies in the uncertain band (initially 0.2–0.95), set the probability to 0 when the judge answers No. Apply this **before** assignment and decoding, so the target can go to its next-best Source 1 if that one is certain.

## 5. Order of work and adoption

Each step changes one thing, and each is adopted only if the **public leaderboard** improves over the previous submission. Only the noisy-channel matcher may fall back to `local_val` if submissions run short. 5 submissions per day.

| # | Submission | New ensemble piece | Also check on `local_val` |
|---|---|---|---|
| 1 | v1 baseline, France filled | – | CV F0.5, ablations (exist) |
| 2 | Same, France blank | – (France probe) | – |
| 3 | + cross-encoders A/B in stage 2 | 3.2 | Stage-2 AUC, F0.5 gain; compare test and training prediction distributions |
| 4 | + noisy-channel `nc` | 3.4 | F0.5 gain, feature importance |
| 5 | + 5-seed stage-1 bagging, isotonic calibration | 3.1, 4.2 | Calibration table, decoding sweep |
| 6 | + France veto | 3.5, 4.5 | Judge precision/recall on the US/India band |
| 7 | + mmBERT-small (only if the pilot is close) | 3.3 | Stage-2 AUC |

**Reproducibility for the submission package:** every member is retrainable from `code/business_entity_resolution` with pinned seeds. The README lists each model's training command, run time and hardware, and gives `ce_scores.parquet` / `nc_scores.parquet` as regenerable intermediates. `candidate_pairs.tsv` stays exactly the set stage 1 scores; members only rescore those pairs.

## 6. Compute budget (estimates, to be replaced with measured numbers)

| Piece | Where | Estimate |
|---|---|---|
| CE-A + CE-B training (1–2M pairs total, 1 epoch) | 24 GB GPU | ~1 h total (4090; ~2× on 3090) |
| CE band inference, 2 models, `local_val` + test | GPU | ~30–45 min |
| mmBERT-small (if adopted) | GPU | ~30 min |
| Qwen3-4B judge, ~50–100k pairs | GPU | ~15–30 min |
| Noisy-channel tables + scoring | CPU | ~3–4 h to build; minutes to score |
| 5-seed stage-1 bagging | CPU | ~5× stage-1 fit time |

## 7. Evidence

| Source | Finding | Consequence |
|---|---|---|
| Our `local_val` candidates | Blocking passes are diverse (score correlation 0.77 on positives, −0.27 on negatives), but equal-weight averaging **hurt**: AUC 0.986 (full pass) → 0.969 (mean) / 0.974 (rank average) | Rule 3: learned weights only |
| Our `local_val` candidates | Exact name+address equality: 100% precise but ~1.4% of positives; exact name only: 84% precise in the US, 95% in India | No hard-rule members |
| Our `local_val` candidates | Raw blocking score already ranks the true Source 1 first for 96.5% of matched targets | Competition/context features (v1 stage 2) carry much of the signal |
| [Better entity matching with transformers through ensembles](https://dl.acm.org/doi/10.1016/j.knosys.2024.111678) (KBS 2024) | Transformer ensembles: about +0.2–0.5 F1 on average, inconsistent across datasets; +1.0× inference per extra model; diversity from text arrangement | Only two cross-encoders, band only; order swap optional |
| [Wu & Gales, Should Ensemble Members Be Calibrated?](https://arxiv.org/abs/2101.05397) | Averaging calibrated members gives an under-confident ensemble | Rule 4; logit-space averaging |
| Stacking practice (Kaggle) | A combiner trained on in-sample member scores learns over-confidence | Rules 1–2 |
| [Match, Compare, or Select?](https://arxiv.org/html/2405.16884) | Candidates competing ("select") beat independent pair matching by about 16 F1 | Context features in stage 2 (v1) |
| [Beyond Scale and Generation](https://arxiv.org/abs/2607.24688) | Generative matchers help mainly under distribution shift | LLM judge restricted to France |

**Rejected:** majority or intersection voting across US/India members (calibrated probabilities with decoding already act as a soft intersection); several GBDT libraries (little diversity); cross-encoders over all ~40M pairs (linear cost for small gains); fitting weights or calibration on the evaluation split; Jellyfish (cc-by-nc-4.0); JEPA-style self-supervised members (see `FINDINGS.md`).

## 8. Open items

- **Training split:** keep training on `local_val` (as v1), or move to a subsample of `local_train` and keep `local_val` as an untouched hold-out? `local_train` gives more data and a clean final check, but costs disk (8.6 GB free) and feature time (~9× `local_val`).
- **Uncertain band limits** for the cross-encoders (3.2) and the France veto (4.5): set from the stage-1 and stage-2 calibration tables.
- **Averaged vs single cross-encoder on test:** decide from the distribution check in 3.2.
- **Judge prompt and Yes-probability cutoff:** fixed on the US/India band before any France use.
