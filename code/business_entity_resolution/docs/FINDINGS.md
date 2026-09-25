# Findings: metric, data structure, pre/post-processing and model choices

The agreed ensemble design (members, fold rules, combiner, France veto) is in [`ENSEMBLE.md`](ENSEMBLE.md).

All numbers were measured on the challenge data on 25 Sep 2026, unless marked as an estimate. Train = `dataset/train`; `local_val` / `local_train` = the closed-universe splits made by `src/data_loader.py`. This complements `PIPELINE.md` (status, file contracts between stages, task board).

## 0. Ground rules we follow

- **Learned resources come from train only:** dictionaries, variant tables, spell-correction vocabulary and model weights. Test *inputs* (never labels; test has none) are used only to score candidates and for the one-Source-1-per-target assignment. No self-training on test.
- **One exception:** unlabeled word-frequency statistics (IDF) may be computed per country on test inputs. France has no training records, so without this French blocking would have no rarity weighting. State this in the methodology document.
- **No external data or lookups** (problem statement, "Fair Play").
- **Country is an open-set string.** Never one-hot it or hard-code `France`. Any special handling for France is keyed on "country not seen in training".
- **`candidate_pairs.tsv` must be exactly the set the model scored.** Post-processing only ever chooses *among* scored candidates, so matches stay a subset of candidates.

## 1. The metric

Per Source 1 entity, with β² = 0.25:

```
F0.5 = 1.25·TP / (1.25·TP + 0.25·FN + FP)
```

- **A false positive costs 4× a missed match.** Adding any candidate increases the denominator by exactly 1, whether it is right or wrong; the numerator grows by 1.25 only if it is right.
- **Decision rule:** a candidate with calibrated match probability `p` is worth adding iff **p > F_current / 1.25**, where F_current is the entity's F0.5 before adding it. Near F = 0.9 the bar is about 0.72, and it rises as the entity improves.
- **Missed matches are cheap on large entities.** 5 true matches with 3 predicted, all correct: F0.5 = 0.88. Never pad.
- **Singletons (5.6% of train Source 1):** an empty prediction scores 1.0; any prediction scores 0. Choosing between empty and the best candidate needs P(entity has no match).
- **Baseline:** predicting nothing scores 0.055 on `local_val` (singletons only).

## 2. Structure in the data

| Finding | Measured (train) | How we use it |
|---|---|---|
| Each target matches at most one Source 1 entity | 7.64M pairs = 7.64M distinct targets | Assign each target only to its best Source 1; feature: margin over the second-best Source 1 |
| **Source 1 is perfectly clean** | 0.000 on every noise indicator: uppercase, brackets, native script, websites, double spaces | Noise is one-sided: normalise targets toward Source 1's vocabulary; use asymmetric features (Source 1 words missing from target vs extra target words) |
| Exact-duplicate targets (normalised name+address) | 131k groups: **92.5%** same Source 1, 6.5% split across entities, 0.3% all unmatched | Duplicate-matched feature and soft propagation, not a hard rule |
| S2 and S3 have different styles | US addresses uppercase: **S2 93%, S3 0%**; native-script India names: S2 24%, S3 13% (test) | Source is a feature; calibrate per source |
| Address numbers change | Target adds a number in 18–23% of true pairs (`#121`, `PO BOX 7915`, `009291`); all Source 1 numbers kept in only 63–83% | Never veto on a number mismatch; strip leading zeros; test whether Source 1's house number appears in the target |
| Matches per source | S2 ≤ 5, S3 ≤ 6 per entity; the two counts are independent (corr −0.02); 85% of entities have both | Cap at 5 (S2) / 6 (S3); independence helps estimate P(singleton) |
| Match counts | 0: 5.6%, 1: 5.4%, 2: 17%, 3: 24%, 4: 22%, 5: 15%, 6+: 11%; max 11 | Prior for decoding |
| Singletons often share a name with a target | 36–42% of singletons have an exact name twin among targets (71–76% for matched entities) | The name alone is a trap; the address and the assignment decide |
| Distractors (26% of targets match nothing) | Exact name appears in Source 1 for only 5.5% of distractors (vs 26.6% of true targets) | Most distractors are not near-copies of a Source 1 record |
| Matches never cross countries | 100% same country | Block within country |
| No postcodes | 0% of Source 1 has a PIN; about 1% of India targets | Postcode blocking and libpostal aren't worth it |
| Test differs from train | 5.75 targets per Source 1 in test vs 4.68 in train | Thresholds tuned on `local_val` may be slightly loose for test |
| **No leak** | File row order and entity-ID correlation between Source 1 and its matches: 0.0001 | Nothing to exploit or guard against |
| Quote characters | Fields containing `"` use standard CSV escaping (`"""ehpad Club SAS"`) | Reading with quoting disabled keeps stray `"`; normalisation strips them, so no impact |

## 3. Pre-processing (all learned from train)

1. **Native-script dictionary (names and addresses).** Indic-script words come from a closed vocabulary of **1,537 words**; test's is 100% covered by train's. About 1.5M train and 1.76M test target records contain native script. The dictionary is built by aligning words by position in (native target name, Latin Source 1 name) pairs with equal word counts. Results on 55,105 `local_val` native-script true pairs:

   | Method | Licence | Token-set similarity to Source 1 name | Exact after normalising |
   |---|---|---|---|
   | Unidecode (current `normalize.py`) | GPL-2 | 67.0 | 0.1% |
   | indic_transliteration (IAST) | MIT | 68.1 | 0.2% |
   | anyascii | ISC | 73.5 | 0.4% |
   | **Learned dictionary + fallback** | our data | **98.4** | **88.3%** |

   The dictionary covers 97.4% of `local_val` native words. Scripts seen: Devanagari, Kannada, Telugu, Tamil, Bengali, Gujarati, Malayalam, Oriya. A neural transliterator (IndicXlit) isn't needed. anyascii is the fallback; switch from Unidecode only after blocking parameters are frozen.

2. **Variant table mined from train pairs**, using the same alignment: `Ave` → `Avenue`, `OH` → `Ohio`, `Eleventh` → `11th`, `Pvt` → `Private`, `[Inc]` → `Inc`, etc.
3. **Reverse the generator's artifacts:** leetspeak (`Onc0logy`, `lnvestment`), repeated words (`Mohan Mohan`, `Socienny Socienny`), bracketed legal forms, honorific prefixes (`Smt`, `Shri`), truncation (`(Limite`), website-style names (`tamikod.com`).
4. **Spell-correct target words to train Source 1's vocabulary**, within the same country (`MONTGMERY` → `montgomery`). France has no train vocabulary, so its words pass through unchanged.
5. **Numbers:** strip leading zeros; extract the house number; compare as "contained in" rather than "equal".

## 4. Post-processing

1. **Greedy expected-F0.5 decoding per entity:** sort by calibrated probability and add while p > F_current / 1.25.
2. **Empty vs top-1** for every entity, using P(no match), to catch singletons.
3. **One Source 1 per target:** a target goes to its highest-scoring Source 1 only.
4. **Duplicate-target propagation** as a feature or soft boost, never a forced match.
5. **Per-source caps:** at most 5 from S2 and 6 from S3.
6. **Unseen countries (France):** leaderboard probe comparing France blank vs filled before deciding; any stricter threshold is keyed on "country not in train".

## 5. Model choices

**Constraints:** MIT/Apache licence and ≤ 8B parameters. GPU: a 24 GB RTX 3090/4090, 6–15 GPU hours before the freeze, run over SSH.

| Role | Choice | Why |
|---|---|---|
| Main matcher | **LightGBM** (MIT) | Scores all ~40M pairs cheaply. Features: RapidFuzz, blocking TF-IDF scores/ranks, number containment, Monge-Elkan / soft TF-IDF, competition margin, source, duplicate signal |
| Cross-encoder | **Pilot:** `cross-encoder/nli-deberta-v3-base` vs `microsoft/mdeberta-v3-base` vs `jhu-clsp/mmBERT-small` | See below |
| Possible upgrade | `microsoft/deberta-v3-large` (MIT, 435M) | About 3× the cost of base; fits in 24 GB |
| Rejected | Qwen3-8B / LLM matchers | Throughput far too low for about 40M pairs; their advantage is few-shot and distribution shift, and we have 7.6M labelled positive pairs |
| Rejected | bge-reranker-v2-m3, gte-multilingual-reranker-base | Trained for query-to-passage relevance and 3–4× the compute; gte needs `trust_remote_code` |
| Rejected | MiniLM rerankers | English vocabulary; loses all native-script text (0% round-trip) |
| Rejected | libpostal | Heavy C dependency, and its main benefit is postcodes, which we don't have |

**Tokenizer measurements** (3,000 random + 1,000 native-script `local_val` true pairs, serialised as `name | address`):

| Tokenizer | Tokens per pair (mean / 95th pct) | Native script: unknown tokens / round-trips intact |
|---|---|---|
| deberta-v3-base | 46 / 83 | 0.1% / 94% |
| mdeberta-v3-base | 49 / 84 | 0.1% / 97% |
| mmBERT-small | 52 / 88 | 0.0% / 100% |
| XLM-R (bge / gte) | 54 / 88 | 0.0% / 92% |
| ms-marco-MiniLM-L6 | 47 / 82 | 3.2% / 0% |

`max_length = 128` is enough.

**Notes on `nli-deberta-v3-base`:**
- Apache 2.0, about 184M parameters, fine-tuned on English SNLI/MNLI only. Labels are `contradiction, entailment, neutral`.
- Zero-shot "entailment" is not "same business", so it must be fine-tuned.
- The NLI head start helps with small labelled sets, not with 7.6M positive pairs.

**Why a multilingual model may still matter:** the native-script dictionary largely solves India, so the case now rests on France. Fine-tuned matchers can lose 22–61% F1 on unseen entities (Peeters et al., arXiv 2310.11244). The pilot's train-on-US, test-on-India run is our stand-in for France.

**Training settings:**
- **Negatives:** hard negatives from blocking's `local_train` candidates.
- **Data size:** 1–2M pairs, one epoch.
- **Precision:** DeBERTa in **fp16, never bf16** (NaN in disentangled attention); mmBERT may use bf16 with FlashAttention 2.
- **Augmentation:** randomly swap the two records' order (matching is symmetric), and apply light token dropout.
- **How the score is used:** it becomes a LightGBM feature rather than a direct threshold.
- **Adoption gate:** adopt only if `local_val` F0.5 improves by ≥ 0.005.

**Throughput (estimate, unmeasured):** deberta-v3-base in fp16 on a 4090 does about 4–6k pairs/s for inference, so about 2–3 h for all test candidates (about 2× on a 3090). mmBERT-small with FlashAttention 2 is roughly 3× faster.

## 6. Corrections to `research.pdf`

- **"Escalate uncertain pairs to an LLM that retrieves external context":** prohibited (external lookup), grounds for disqualification.
- **"Use libpostal postal codes as blocking keys":** our data has essentially no postcodes.
- **"NLI-initialised DeBERTa consistently beats the base model":** that evidence is from small labelled sets; ours has 7.6M positive pairs.

## 7. Open items

- **`train_source1.tsv` timestamp:** it is newer than the other raw files (25 Sep vs 18 Sep). The content looks consistent (row count matches ground truth; CSV quoting pattern matches test). Compare its checksum with the packed Kaggle copy.
- **Local dependencies not yet in `requirements.txt`:** `sentencepiece==0.2.2` and `protobuf==7.36.2`, required by DeBERTa tokenizers.
- **Planned France probe:** 2 leaderboard submissions, identical except France is blank in one.

## Sources

- [cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) · [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base) · [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) · [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base)
- Peeters et al., [Entity Matching using Large Language Models](https://arxiv.org/html/2310.11244) · [Match, Compare, or Select?](https://arxiv.org/html/2405.16884) · [Beyond Scale and Generation](https://arxiv.org/abs/2607.24688)
- [transformers PR #24116 (mDeBERTa fp16 overflow fix)](https://github.com/huggingface/transformers/pull/24116)
