# Progress report: Business Entity Resolution (MLC 2026)

_Written Fri 25 Sep 2026, ~23:00 IST. Deadline Sun 27 Sep 2026, 23:59 IST (code freeze Sun 16:00 IST)._
_Read this first, then use [`PIPELINE.md`](PIPELINE.md): the status table, stage contracts and task board you pick work from._

## TL;DR

- We have a **trustworthy local evaluation setup**:
  - a validation split that behaves like the test set;
  - a scorer that reproduces the leaderboard F0.5;
  - cloud setup for Kaggle/Colab.
- **Blocking v1 works on `local_val`:** it finds **98.2% of true pairs** with 38 candidates per S1 record. A perfect matcher on these candidates would score **F0.5 = 0.994**, so blocking is not our bottleneck.
- **Not built yet:** the matcher (features + LightGBM + cross-encoder), assigning each target to one S1, threshold tuning, and the submission package. **No leaderboard submission yet.**
- **Biggest open risks:**
  1. Blocking runtime on the full test set (about 76× bigger than `local_val`).
  2. India names in native scripts (only ~88% of those true pairs are found).
  3. France is in test but not in train, so we cannot measure it locally.

---

## 1. The problem in one paragraph

Source 1 (S1) is a deduplicated reference list of businesses (name, address, country). For each S1 record we must list every S2/S3 record that is the same business.
- **Score:** F0.5, computed per S1 record and averaged.
- **Singletons count:** an S1 record with no true match scores 1.0 if we predict nothing and 0.0 if we predict anything.
- **Two output files:** `matching_results.tsv` (scored) and `candidate_pairs.tsv` (the exact candidate set our model scored; audited for blocking quality).
- **Model rules:** any model must be MIT or Apache-2.0 licensed and ≤8B parameters.
- **Reproducibility:** the zip must regenerate both outputs from the provided data using only what is in the zip.
- **Train vs test:** training covers US and India; test adds France.

## 2. What the data told us (these facts drive every design choice)

All numbers come from full-size queries on the training data.

| Fact | Number | Consequence |
|---|---|---|
| Each S2/S3 record matches **at most one** S1 record | 7.64M matched pairs = 7.64M distinct targets | Assign each target only to its best S1: a big, free precision gain |
| Matches never cross countries | 100% | Block within country; compare country as a plain string, so France needs no special code |
| Matches per S1 record | 5.6% have none; most have 2–5; the maximum is 11 | Recall matters too: about 3.5 true matches per record on average |
| Distractors | ~26% of S2/S3 records match nothing | The matcher must be able to say "no" |
| India names are noisy | 33% of true pairs have name Jaro-Winkler below 0.8; 40% differ in the first 4 characters | Exact-key and prefix blocking fail |
| Native-script names | **18% of India target names** (Kannada, Malayalam, …) | Need transliteration or a multilingual model |
| Postcodes | essentially absent | Postcode blocking is useless |
| Name collisions | "primary care" appears 397× in S2; 667k repeated name+country combinations even in the "deduplicated" S1 | The name alone cannot decide; the address matters |
| Literal `"` inside fields | e.g. `""NIAGARA"" BUILDING` | **Always read with `quoting=csv.QUOTE_NONE`** (`data_loader.read_tsv`) |
| Test is bigger relative to S1 | 5.75 targets per S1 in test vs 4.68 in train | A threshold tuned locally is probably too loose for test; probe it on the leaderboard |

## 3. Chronology: what was built, in order, and how it was checked

### Step 0: Review of the starting code (Fri evening)
Existing code: `blocking.py` (exact name / prefix / sorted-token keys in Python dicts), `matching.py` (off-the-shelf NLI DeBERTa, never fine-tuned, fixed 0.85 threshold), `preprocess.py` (hand-written US/India dictionaries). Problems found:
- Blocking would likely run out of memory at full size (about 10M records held as Python dicts on a 15 GB laptop).
- Its 4-character prefix key misses about 40% of India true pairs.
- An NLI "entailment" score is not a "same business" score.
- `requirements.txt` was not pinned, and `unidecode` was missing from the venv, so `preprocess.py` could not even be imported.

These files are kept as `blocking_legacy.py`, `matching.py` and `preprocess.py` **for reference only**.

### Step 1: Validation split + scorer: PR #1 (merged)
- **`data_loader.py`** splits the training data into two self-contained universes:
  - `local_val` = 10% of S1 (chosen by a seeded hash of the ID) + every S2/S3 record matched to them + 10% of the records that match nothing.
  - `local_train` = everything else.
- **Why:** the old split scored held-out S1 against *all* train S2/S3. There, 90% of the targets belonged to S1 records outside the split, so the distractor mix was wrong and the one-target-one-S1 rule could not be tested.
- Output: `dataset/splits/{local_train,local_val}/`, in the same file layout as `dataset/train/`. Takes about 1 minute.
- **`evaluate.py`** reproduces the leaderboard metric exactly. It reports:
  - scores by country and for singletons, plus precision/recall and false-positive counts;
  - for blocking: pair recall, reduction ratio, candidate counts, and the **F0.5 ceiling** (the best a perfect matcher could score on those candidates).
- **Checked:**
  - unit tests (`tests/test_evaluate.py`), including the 0.714 example from the problem statement;
  - the split divides S1, S2 and S3 exactly, with no overlaps;
  - predicting nothing scores 0.055 (that's the singleton share) and the ground truth scores 1.0.

| split | S1 | targets | targets per S1 | share of targets matched | singletons |
|---|---|---|---|---|---|
| local_val | 221,025 | 1,034,103 | 4.68 | 74.1% | 5.5% |
| local_train | 1,985,796 | 9,286,116 | 4.68 | 74.0% | 5.6% |

### Step 2: Cloud setup: PR #2 (open)
- **`scripts/pack_data.sh`** packs the raw train and test files into **one** zstd `-19` archive: 2.5 GB → 698 MB. It also writes sha256 checksum files for the raw data and for the local splits.
- **`scripts/cloud_setup.sh`**, on Kaggle or Colab:
  1. copies the archive to the machine's local disk (reading through the Drive mount is slow);
  2. verifies the checksums and extracts;
  3. installs the pinned requirements while keeping the machine's CUDA build of torch;
  4. regenerates the splits and **checks they are byte-identical** to the ones made locally;
  5. runs the scorer tests.
- **`notebooks/cloud_runner.ipynb`** detects Kaggle or Colab, clones this private repo using a `GH_TOKEN` secret, and runs the setup script.
- **`config.py`**: paths can be overridden with `BER_DATA_DIR`, `BER_OUTPUT_DIR` and `BER_CACHE_DIR`.
- **Checked:** a full local rehearsal on a small copy of the data. Checksums and splits matched, a rerun skipped extraction, and a corrupted file forced a fresh extract. **Not yet run on a real Kaggle machine** (task T1).

### Step 3: Blocking v1: PR #3 (this PR)
- **`normalize.py`**: transliterate to ASCII with unidecode, lowercase, `&` → `and`, collapse punctuation to spaces. There are deliberately **no hand-written suffix or state lists**; common words like "pvt", "llc" and "sarl" are down-weighted automatically by IDF (rarity) learned from the data, which also covers France.
- **`blocking.py`**: for each S1 record, take the top-K most similar targets **within the same country** by TF-IDF cosine, in several passes, and keep the union:

  | pass | text | features | default K |
  |---|---|---|---|
  | `addr` | normalised address | word tokens (house numbers included) | 20 |
  | `full` | name + address | word tokens | 30 |
  | `name` | normalised name | character 3-grams | **0 = off** (see experiments) |

  - IDF is fitted per country on S1 + targets, using only the provided files.
  - Tokens found in more than 2% of targets are dropped (`--max-df`).
  - Top-K is found with a chunked sparse matrix product, parallelised across worker processes.

- **Outputs:**
  - `$BER_OUTPUT_DIR/<split>/candidate_pairs.tsv` (submission format);
  - `$BER_CACHE_DIR/<split>/candidates.parquet`, one row per pair with each pass's score and rank. This file is the contract with the matcher; see `PIPELINE.md` §3.

#### Experiments on `local_val` (all measured; this is why the defaults are what they are)

**Run A:** all three passes, K=20 each, max-df 2%, single process. Wall time 17 min, peak RAM 3.9 GB.

| share of true pairs found | name | addr | full | union |
|---|---|---|---|---|
| All | 0.710 | 0.904 | 0.976 | 0.984 |
| US | 0.765 | 0.919 | 0.991 | 0.995 |
| India | 0.628 | 0.880 | 0.953 | 0.969 |
| non-Latin names | **0.010** | 0.879 | 0.844 | 0.885 |

→ The name-trigram pass was the slowest (370 s for US alone) and the weakest. It is useless on native scripts: unidecode's transliteration looks nothing like the English spelling. **Decision: disable it.**

**Run B, frequency-cap sweep** (addr + full passes; `max_df` is the share of targets a token may appear in before it is dropped):

| max_df | K addr / full | full pass | union | candidates per S1 |
|---|---|---|---|---|
| 0.001 | 20 / 20 | 0.888 | 0.903 | 28.8 |
| 0.001 | 30 / 50 | 0.915 | 0.922 | 56.2 |
| 0.005 | 20 / 20 | 0.959 | 0.965 | 30.5 |
| **0.02** | 20 / 20 (Run A) | 0.976 | – | – |

→ Dropping common tokens is the main speed lever: a 0.1% cap makes the matrix product about 11× cheaper on a timing sample. But it **destroys recall**. Common tokens like the city or "road" look useless alone, yet they rank the candidates that share a rare token, and raising K does not bring that back. **Decision: keep max-df 2%; get speed from parallelism instead.**

**Run C, chosen configuration:** addr K=20 + full K=30, max-df 2%, parallel. Scored with `evaluate.py --candidates`:

| | ALL | India | US |
|---|---|---|---|
| candidate pairs | 8,382,935 | 3,171,710 | 5,211,225 |
| candidates per S1 (mean / p99 / max) | 37.9 / 48 / 50 | 35.7 / 47 / 50 | 39.4 / 48 / 50 |
| **share of true pairs found** | **0.9823** | 0.9647 | 0.9941 |
| entities with every match found | 0.9486 | 0.9019 | 0.9800 |
| **F0.5 ceiling (perfect matcher)** | **0.9937** | 0.9870 | 0.9982 |
| reduction ratio | 0.99993 | 0.99991 | 0.99994 |

## 4. What is verified and what is not

| Claim | Status |
|---|---|
| The scorer equals the leaderboard formula | ✅ unit tests, including the official example |
| `local_val` has the same structure as train | ✅ partition checks |
| `local_val` scores predict leaderboard scores | ❌ **unknown**: test has more distractors and includes France. Submissions 1–2 will measure the gap |
| The cloud setup works on a real Kaggle machine | ❌ rehearsed locally only (T1) |
| Blocking recall / ceiling on `local_val` | ✅ Run C above |
| Blocking fits the runtime budget on test | ❌ **not measured**. Test is 7.8× the S1 records and 9.7× the targets, and cost grows with their product, so roughly 76× the `local_val` work. Run C's wall time was lost (the session dropped). Re-time it before launching test (T2) |
| Blocking quality on France | ❌ cannot be measured locally (no labels) |

## 5. Decisions we made (and why)

1. **Cascade matcher:** LightGBM on all pairs → cross-encoder on the uncertain band → stacked LightGBM.
   - Cross-encoders are state of the art for pairwise matching, but a transformer over ~40M test pairs on a T4 GPU takes about 10h, beyond our budget of one Kaggle GPU session.
   - The cross-encoder's score becomes a feature, and we keep it only if `local_val` F0.5 improves by ≥0.005.
   - Base model: `microsoft/mdeberta-v3-base` (MIT, multilingual, so native scripts are covered).
2. **Assignment:** each target goes to at most one S1 record: the one with the highest match probability.
3. **Threshold:** tune on `local_val`, then set slightly **stricter** for test, because test has more distractors.
4. **France probe:** submit once with France predictions blanked and once filled, to measure France on the public leaderboard.
5. **Reproducibility:** all training code ships in the zip, and public MIT/Apache checkpoints are downloaded at run time.
6. **Where things run:** develop on `local_val` locally; run `local_train` and test on Kaggle (30 GB RAM, T4 GPU; *Save Version* keeps running after you close the browser). Colab Free is the backup GPU.

## 6. How to get going (new teammate)

```bash
git clone https://github.com/rishang19dx/AMZN_MLC2026.git && cd AMZN_MLC2026/code/business_entity_resolution
python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# put the raw data in <repo>/dataset/{train,test}/ (or extract mlc26_data.tar.zst there)
python src/data_loader.py                                   # splits, ~1 min
python tests/test_evaluate.py                               # scorer self-check
python src/blocking.py --split local_val                    # candidates + per-pass recall
python src/evaluate.py --split local_val --candidates ../../output/local_val/candidate_pairs.tsv
```
**Cloud:** ask Rishang for the Kaggle dataset `mlc26-data`, add your own `GH_TOKEN` secret, open `notebooks/cloud_runner.ipynb`.

## 7. What's next (pick from the board in `PIPELINE.md` §6)

Priority order for Sat 26 Sep:
1. **T1/T2:** first real Kaggle run; re-time blocking; produce `candidates.parquet` for `local_train` and test.
2. **T3/T4:** `features.py` + `match.py` (RapidFuzz features, ranking features within each S1's candidates, LightGBM, assignment, threshold) → **Submission 1 by Sat midday**, then the threshold-probe Submission 2.
3. **T6:** cross-encoder fine-tuning on the Kaggle GPU, in parallel with T7 (error analysis: native-script names first).

Known improvement ideas for blocking (only if error analysis says blocking is the bottleneck, which, with a 0.994 ceiling, it probably isn't):
- better transliteration or multilingual embeddings for native-script names;
- pruning from the target side (keep each target in at most m S1 lists) to cut candidate count.
