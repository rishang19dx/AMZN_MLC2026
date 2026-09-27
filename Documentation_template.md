# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** r/mak  
**Team Members:** Rishang Yadav, Mehul Sharma, Karanpreet Singh Dhaliwal  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

Two passes of per-country TF-IDF nearest-neighbour search generate about 38 candidates per Source 1 record. A two-stage LightGBM scores them: the first stage looks at each pair alone, the second at how the pair competes within its Source 1's list and against other Source 1 records for the same target. Every target is then given to at most one Source 1, and each Source 1 keeps the set of candidates that maximises *expected* F0.5. The main innovations are:
- a native-script dictionary learned from the training pairs, which replaces transliteration;
- training the matcher on a **test-scale** split, which lifted the public score from **0.9334 to 0.959**.

---

## 2. Methodology

### 2.1 Problem Analysis

All figures were measured on the training data.

- **Structure:**
  - Every S2/S3 record matches **at most one** Source 1 record (7.64M matched pairs = 7.64M distinct targets).
  - Matches **never cross countries**.
  - 5.6% of Source 1 records have no match; most have 2–5; the maximum is 11 (S2 ≤ 5, S3 ≤ 6).
  - About 26% of S2/S3 records match nothing (distractors).
- **Noise is one-sided:** Source 1 is perfectly clean (0% uppercase, brackets, native script or double spaces); all noise is in S2/S3. S2 and S3 differ in style: US addresses are uppercase in 93% of S2 and 0% of S3.
- **Names:** abbreviations, legal-form changes, typos, leetspeak (`Onc0logy`), repeated words, trade names and website-style names. India is hard: 33% of true pairs have name Jaro-Winkler below 0.8.
- **Native scripts:** 18% of India target names are written in Indic scripts (Devanagari, Kannada, Telugu, Tamil, …). These are mostly English words transliterated from a **closed 1,537-word vocabulary**, identical in train and test.
- **Addresses:** reordered components; targets add numbers in 18–23% of true pairs (units, PO boxes, leading zeros); postcodes are essentially absent.
- **Repeated names:** 667k name repeats inside the deduplicated Source 1, so the address has to decide.
- **Scale matters:** test has a ~10M-record target pool. A small validation universe (1M targets) makes candidates look far less alike (median address TF-IDF cosine 0.32 vs 0.46 on test). A matcher trained there scored 0.986 locally but only 0.933 on the leaderboard.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + structured decoding (hybrid).  
**Core Innovation:**
1. A **learned native-script dictionary**: aligning names word by word across training pairs makes 88% of native-script names identical to their Source 1 name, against ≤0.4% for off-the-shelf transliterators.
2. **Competition-aware stage-2 features** that exploit "one Source 1 per target".
3. **Expected-F0.5 decoding** per Source 1, which predicts singletons correctly.
4. **Test-scale training (`scale_val`)**: Source 1 records blocked against the full ~9.3M-record training target pool, so the matcher sees test-like crowding.

Country is compared only as a string (never a feature or a filter), so France, unseen in training, needs no special code. The one country-aware step is a decoding shift for Source 1 in countries **absent from training** (keyed on that, never on a name): −0.75 in logit space, chosen on the public leaderboard (0.958506 → 0.958662).

---

## 3. Candidate Generation (Blocking)

- **Normalisation:** anyascii transliteration, lowercase, alphanumerics only. Indic-script words go through the learned dictionary first (built from `local_train` pairs only).
- **Blocking keys used:** within each country, two TF-IDF top-K passes computed as a chunked sparse matrix product:
  - **address words, top 20**;
  - **name + address words, top 30**.
  The union is kept (≈38 candidates per Source 1, at most 50). IDF is fitted per country on the split's own records, which is unsupervised and the only use of test inputs. Words in more than 2% of targets are dropped.
- **Candidate pairs generated:** **65,247,068** on test (1,732,544 Source 1; 37.7 per record). That's a reduction ratio of 0.99999 against the within-country Cartesian product.
- **How we ensured true matches were not lost:**
  - Recall was measured against ground truth on closed-universe splits.
  - On `local_val` (1M targets): **98.8%** of true pairs kept, an F0.5 ceiling of 0.996.
  - At **test scale** (`scale_val`, 9.3M targets): **96.9%** kept (India 94.8%, US 98.3%), a ceiling of 0.989.
  - Two independent views (address, name+address) catch trade-name and name-change pairs.
  - The native-script dictionary lifted native-name recall from 88.5% to 97.8%.
  - A "rare-word shortlist" speed-up was measured and **rejected**: it lost 2.6 points of recall.

---

## 4. Matching Model

**Features used (37 per pair):**
- **Name:** RapidFuzz ratio, token-set, token-sort, partial ratio and Jaro-Winkler; exact match; TF-IDF cosine on name words and on name character trigrams; IDF-weighted coverage in both directions (Source 1 words found in the target, and target words found in Source 1). Noise is one-sided, so the two directions carry different information.
- **Address:** RapidFuzz ratio, token-set and partial; exact match; TF-IDF cosine and both coverages. Numbers: is Source 1's house number in the target, the share of Source 1 numbers found in the target, and extra target numbers (leading zeros stripped; never a hard veto).
- **Other:** blocking scores and ranks per pass; exact-duplicate target group size and duplicates within the list (92.5% of duplicate groups belong to one Source 1); how many Source 1 lists contain the target ("hub" records); list length; target source (S2/S3); missing address; non-Latin name; token counts.
- **Stage-2 context** (from out-of-fold stage-1 probabilities):
  - within the Source 1 list: rank, gap to the top, top-1, top-1 minus top-2, sum, and count above 0.5;
  - across the target's competing Source 1 records: rank, **margin over the best competing Source 1**, top-1, and count above 0.5.

**Model type:** LightGBM (MIT), two stages, binary log-loss.
- 1,904 / 630 trees; `num_leaves` 127, learning rate 0.08; early stopping on a held-out 10% of Source 1 groups.
- **Training data:** `scale_val` (377,423 training Source 1 records against all 9.29M `local_train` targets; 14.45M candidate pairs).
- Stage 2 is trained on **out-of-fold** stage-1 probabilities (2-fold, grouped by Source 1), so at test time it sees the same kind of inputs.

**Threshold selection method:**
- No global threshold. After assigning each target to its highest-probability Source 1 and capping S2 ≤ 5 / S3 ≤ 6, each Source 1 keeps the k (k = 0 means "no match") that maximises **expected F0.5** under the calibrated probabilities (Monte-Carlo).
- The only free parameter, a logit shift, was chosen on `scale_val` out-of-fold predictions: shift 0 gave 0.9670, against 0.9664 for the best global threshold.

---

## 5. Results & Error Analysis

| Setting | F0.5 (macro, per Source 1) |
|---|---|
| `local_val` out-of-fold (small universe) | 0.9858 |
| **`scale_val` out-of-fold (test-scale)** | **0.9670** (India 0.9616, US 0.9707) |
| Public LB, matcher trained on `local_val` | 0.9334 |
| **Public LB, matcher trained on `scale_val` (final)** | **0.959** |
| Probe (France blank) | 0.8100 → US/India ≈ 0.944, France ≈ 0.874 for the first model |

- **Where the loss is** (`scale_val`, out-of-fold):
  - precision 0.992 but recall 0.927: 11,437 false-positive pairs against 99,173 missed pairs;
  - about 3.1 points of the recall loss happen in blocking, the rest in the matcher;
  - singleton accuracy is 0.960.
- **Common false positives (wrong merges):** 91% of the 11,437 wrong pairs are extra targets on Source 1 records that *do* have matches, and 9% are merges on true singletons (990). About 22% have an exact name twin. Typical cases:
  - **Same name, different address:** branches or neighbours on the same street with a changed house number (`Family Partners | 526 Franklin St` vs `Family Pbrcners | #703 Ervin Rd`; `Proex PLLC | 110 Franklin St` vs `Proex Group | 121 Franklin St`).
  - **Same building or address, different business** (`Ss Products Pvt Ltd` vs `DAWN (&)` at one Ballia address).
  - **Related trade names** (`Amritsar Centre Pvt Ltd` vs `Amritsar Cuisine Pvt Ltd`).

  94% of the false pairs have p2 above 0.7, so they are confident mistakes, not threshold noise.
- **Common false negatives (missed matches):** Of the 99,173 missed pairs:
  - **41%** were never retrieved by blocking. The fixed top-K is crowded out by look-alikes at test scale, most of all in India.
  - **52%** were retrieved but fell below the decoding cut. These are mostly **renamed or trade-name targets at the same address** (`Green Learning Global LLC` → `Nexveo`, same street and number), heavily re-ordered or abbreviated names (`Interstate Excavation Inc` → `Interstate Inc Excavation`), and **targets with no address** that carry only the name.
  - **7%** went to a different Source 1 with a near-identical name (`Housing Project II` vs `Housing Project Services`, whose address is empty).

  Source 1 records with exactly one true match lose the most per record (0.089), because one miss scores zero. India's per-record loss is 1.3× the US's.

---

## 6. Conclusion

Careful blocking with learned normalisation, a competition-aware two-stage LightGBM and F0.5-optimal decoding reach 0.959 on the public leaderboard, with small, licence-clean models.

The biggest single lesson: **validation must match test scale.** The same model trained on a 9× smaller universe lost 0.026 on the leaderboard, and a test-scale split both revealed and fixed it.

Measured next steps, not done before the freeze:
- a larger blocking K (scale recall 96.9% → 97.5%, India 94.8% → 95.8%);
- the listwise mDeBERTa cross-encoder on uncertain pairs (implemented, untrained);
- French abbreviation normalisation.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (Python 3.10, pinned `requirements.txt`; `README.md` has the full runbook):

| File | Role |
|---|---|
| `src/data_loader.py` | Closed-universe splits: `local_train`/`local_val`, `scale_val` (`--scale-val`), `ce_train` (`--ce-train`) |
| `src/translit.py`, `src/normalize.py` | Native-script dictionary learned from `local_train`; normalisation |
| `src/blocking.py` | Per-country TF-IDF top-K → `output/<split>/candidate_pairs.tsv` + `cache/<split>/candidates.parquet` |
| `src/features.py` | 37 pair features, streamed in parts (test: 26 min, 9.9 GB peak) |
| `src/match.py` | Two-stage LightGBM: `--cv` (out-of-fold score + decoding choice), `--fit`, `--predict`; assignment and expected-F0.5 decoding → `matching_results.tsv` |
| `src/evaluate.py` | Local leaderboard metric + blocking recall / ceiling |
| `src/cross_encoder.py` | mDeBERTa cross-encoder with listwise loss (implemented; not used in the final submission) |
| `scripts/run_pipeline.sh` | One command, resumable |

**Reproduce the final outputs** (from `code/business_entity_resolution/`):
```bash
python src/data_loader.py && python src/data_loader.py --ce-train && python src/data_loader.py --scale-val
python src/translit.py --split local_train
python src/blocking.py --split scale_val && python src/features.py --split scale_val
python src/match.py --split scale_val --cv && python src/match.py --split scale_val --fit
python src/blocking.py --split test && python src/features.py --split test
python src/match.py --split test --predict
python src/match.py --split test --redecode --unseen-shift=-0.75   # final file: stricter decoding for countries unseen in training
```
The submitted `output/matching_results.tsv` is `output/test/matching_results_unseen-0.75.tsv` from the last command.
Outputs: `output/test/matching_results.tsv` and `output/test/candidate_pairs.tsv`.

Measured on a 16-core / 15 GB laptop:
- test blocking 4.4 h;
- `scale_val` blocking 56 min and features 26 min;
- `--cv` 56 min, `--fit` 3.5 h;
- test features 26 min;
- test predict about 1.5 h.

### B. Additional Results

| Blocking recall vs ground truth | `local_val` | `scale_val` (K 20/30, final) | `scale_val` (larger K) |
|---|---|---|---|
| True pairs kept | 98.8% | 96.9% | 97.5% |
| India / US | 97.9% / 99.4% | 94.8% / 98.3% | 95.8% / 98.6% |
| F0.5 ceiling | 0.996 | 0.989 | – |

| Ablation (`local_val`, out-of-fold) | F0.5 |
|---|---|
| Stage 1 only + expected-F0.5 decoding | 0.9836 |
| + stage 2 context | 0.9858 |
| Unidecode instead of the learned dictionary | 0.9827 |
