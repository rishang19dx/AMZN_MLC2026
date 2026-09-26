# Multi-pipeline blocking / candidate generation

For every Source 1 (S1) entity, produce a small, high-recall list of plausible Source 2/3 records.
Three independent pipelines each propose candidates. Their lists are **unioned** (never intersected), deduplicated and, only if a list is too long, cut by priority.

```
                   Source 1 record
        ┌──────────────────┼────────────────────┐
   Pipeline A          Pipeline B            Pipeline C
   fine-tuned BERT     JEPA-style encoder    TF-IDF passes + blocking keys
   bi-encoder          (asymmetric)          (sparse top-K)
   FAISS top-K/S2,S3   FAISS top-K/S2,S3     top-K per pass
        └──────────────────┼────────────────────┘
          union → dedup → priority → budget
                           ↓
     candidate_pairs.tsv  +  debug_candidate_scores/  +  blocking_report
                           ↓
     lgbm_matcher.py  →  matching_results.tsv  (subset of the candidates)
```

> **Status:** implemented but **not yet trained or run on the real data**. No recall numbers for this blocker exist yet: they are written to `blocking_report.txt` by the first run on `local_val`. The only measured numbers in the repo are for the v1 TF-IDF blocker (`blocking.py`, which Pipeline C reuses): 98.2% pair recall at 37.9 candidates per S1 on `local_val` (REPORT §3).

## Files

| Path | What it does |
|---|---|
| `configs/blocking.yaml` | Every setting, and the defaults. `--config my.yaml` is deep-merged over it; `--set a.b=c` overrides one value |
| `src/train_blocker.py` | Trains Pipelines A and B (optionally on a random subset), then validates the whole blocker |
| `src/generate_candidates.py` | Inference on any directory → `candidate_pairs.tsv` (+ report if labelled) |
| `src/check_blocking.py` | Scores any `candidate_pairs.tsv` against ground truth, with penalties for wasted candidates |
| `src/lgbm_matcher.py` | Basic LightGBM matcher: `fit` on the holdout candidates, `predict` on test |
| `scripts/run_multiblock.sh` | All stages, end to end, up to the official validator |
| `src/blocker/data.py` | Directory loading, entity-level closed-universe holdout, random entity subsets |
| `src/blocker/normalization.py` | `normalize_name` / `normalize_address` / `normalize_country`, `name_core`, postal codes, blocking keys |
| `src/blocker/models/` | `encoder.py` (shared encoder + batched inference), `bert_encoder.py` (A), `jepa_encoder.py` (B) |
| `src/blocker/training/` | `pairs.py`, `losses.py`, `contrastive_training.py` (loop + A), `jepa_training.py` (B), `hard_negative_mining.py` |
| `src/blocker/pipelines/` | `classical_blocker.py` (C), `embedding_blocker.py` (A/B retrieval), `ann_index.py`, `candidate_union.py`, `engine.py`, `output.py` |
| `src/blocker/evaluation/blocking_metrics.py` | Recall, candidate stats, pair quality, penalties, per-pipeline Venn, K and budget sweeps |
| `tests/test_blocker.py` | Unit and integration tests on a synthetic directory, with tiny local models |

The older modules are not changed, except for one refactor: `translit.py` gained `learn_table()` so the dictionary can be learned from an in-memory split. `blocking.py` (the v1 blocker, still used by `scripts/mac_run.sh`), `features.py` and `match.py` still work as before. The new code imports `topk_sparse`/`within_rank` from `blocking.py` and `assign_and_cap`/`group_stats` from `match.py`.

## How to run

All commands run from `code/business_entity_resolution/`. `DATA` is the directory holding `train/`, `test/` and `splits/`.

```bash
pip install -r requirements.txt                 # adds faiss-cpu, PyYAML, pytest
python -m pytest tests/test_blocker.py -q       # synthetic data, tiny models, ~1-2 min

bash scripts/run_multiblock.sh                  # everything (DATA=... ART=... OUT=... to relocate)
TRAIN_FRACTION=0.1 EPOCHS=1 bash scripts/run_multiblock.sh   # fast first pass
PIPELINES=classical bash scripts/run_multiblock.sh           # no neural training at all
```

Step by step (this is what the script does):

```bash
python src/data_loader.py                                          # splits/local_train, splits/local_val
python src/train_blocker.py --train-dir $DATA/splits/local_train --val-dir $DATA/splits/local_val \
    --output-dir $ART --train-fraction 0.2 --no-eval                # add --device cuda, --epochs 1, --pipelines bert
python src/generate_candidates.py --data-dir $DATA/splits/local_val --artifacts-dir $ART \
    --output-dir $OUT/local_val --k-sweep                           # report + recall-vs-K
python src/check_blocking.py --candidates $OUT/local_val/candidate_pairs.tsv --data-dir $DATA/splits/local_val \
    --debug $OUT/local_val/debug_candidate_scores
python src/lgbm_matcher.py fit --data-dir $DATA/splits/local_val --candidates-dir $OUT/local_val --artifacts-dir $ART
python src/generate_candidates.py --data-dir $DATA/test --artifacts-dir $ART --output-dir $OUT/test
python src/lgbm_matcher.py predict --data-dir $DATA/test --candidates-dir $OUT/test --artifacts-dir $ART --output-dir $OUT/test
python3 ../../utils/validate_submission.py --matching $OUT/test/matching_results.tsv \
    --candidate $OUT/test/candidate_pairs.tsv --test-dir $DATA/test
```

`train_blocker.py --train-dir $DATA/train` (without `--val-dir`) makes the holdout itself. It uses the same hash as `data_loader.py`, so the holdout equals `local_val`. It then also runs the validation blocker and writes it to `$ART/validation/`.

### On Kaggle

Use `notebooks/kaggle_multiblock.ipynb`. It clones this branch and prepares the VM (see below), then runs `run_multiblock.sh` with:
- `DATA=/tmp/data`;
- `OUT=/tmp/out` for the large intermediates;
- `ART=/kaggle/working/artifacts`, which *Save Version* keeps.

Deliverables are copied to `/kaggle/working/output/`.

**How it prepares the VM** (`scripts/cloud_setup.sh`):
- It links the attached data in. That can be the official zip as a Dataset, or `mlc26_data.tar.zst`.
- It installs the pinned requirements but keeps Kaggle's CUDA torch.
- It builds `local_train` / `local_val`.

**Settings:**
- `MODE='classical'` runs on CPU.
- `MODE='full'` needs *GPU T4 x2*. Both GPUs are used through `DataParallel` (`runtime.multi_gpu`), and fp16 AMP is used because T4s have no fast bf16.
- `STAGES` splits the work across sessions, and `PREV_ARTIFACTS` reuses the encoders from an earlier run.

The notebook's first cell has the full plan.

### Subset flags (fast training)

| Flag | Effect |
|---|---|
| `train_blocker.py --train-fraction F` | Train the encoders on a random F of the training S1 entities, together with all their targets and F of the distractors |
| `train_blocker.py --val-subsample F` | Final validation report on F of the holdout entities |
| `--set models.bert.max_pairs_per_epoch=N` | Cap on positive pairs per epoch; a new random sample each epoch |
| `lgbm_matcher.py fit --train-fraction F` | Matcher trained on F of the holdout S1 entities |
| `generate_candidates.py --subsample F` | Block only F of the entities, for quick experiments |

All sampling is at the **entity** level and seeded (`data.subset_seed`, `seed`).

## 1. Normalisation (`src/blocker/normalization.py`)

Built on `normalize.py`: anyascii transliteration, a native-script → Latin word dictionary learned from **training** pairs only, lowercasing, `&`→`and`, and punctuation → space. On top of that, the new module adds these token rules:
- **Acronyms:** runs of single letters are joined (`S.A.S` → `sas`, `C.I.T.` → `cit`).
- **Numbers:** ordinals and their typos lose the suffix (`2nd`, `45th`, `45nd` → `2`, `45`), and leading zeros are dropped.
- **Placeholders:** `null`, `none`, `nan`, `nil` are removed.
- **Names:** legal-form and business abbreviations are unified (`pvt`→`private`, `ltd`→`limited`, `corp`→`corporation`, `et`→`and`, …).
- **`name_core`:** the name with legal forms and function words removed. It is used by the keys pass and the char-n-gram pass.
- **Addresses:** street types are unified in English and French (`rd`→`road`, `st`→`street`, `r`→`rue`, `bd`→`boulevard`, `imp`→`impasse`, …).
- **Postal codes:** 5- and 6-digit tokens are extracted as blocking keys.
- **Country:** normalised as an **open-set** string. Only spelling variants of one label are unified (`USA`→`us`), and any other value passes through unchanged.

Raw fields are never overwritten. The normalised columns are added beside them: `name_n`, `addr_n`, `country_n`, `name_core`, `full_n`, `keys`.

## 2. Pipeline A: fine-tuned BERT bi-encoder

- **Base model:** `intfloat/multilingual-e5-small` (MIT, 117.7M parameters). It is multilingual, so Indic-script and French text is covered.
- **Architecture:** mean pooling → linear projection to 256 dimensions → L2 normalisation. Each record is embedded independently, so there is no pairwise model inference.
- **Input:** `query: name: <name_n> | address: <addr_n> | country: <country_n>`, built from the normalised fields.
- **Objective:** symmetric InfoNCE (Multiple Negatives Ranking loss) between S1 and its matching S2/S3 records, temperature 0.05. Negatives are:
  - in-batch negatives, from batches that each hold one country, so the negatives are plausible look-alikes;
  - mined hard negatives (see §6).
- **Masking:** another true match of the same S1 entity is masked out of the negatives, so true matches are never pushed apart.
- **Retrieval:** a FAISS index per (country, source). Each S1 gets its top `top_k_source2` from S2 and its top `top_k_source3` from S3.

## 3. Pipeline B: JEPA-style predictive encoder

**Why no real JEPA:** no JEPA checkpoint fits the rules. Meta's I-JEPA and V-JEPA are vision models with non-commercial licences, and there is no text JEPA release. So the JEPA objective is implemented on an Apache-2.0 BERT-family model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (117.7M parameters).

**Components:**
- **Context encoder** f_θ (trained): reads a noisy observation, i.e. an S2/S3 record or a masked S1 record.
- **Target encoder** f_ξ = EMA(f_θ): reads the clean S1 reference. It gets no gradients (stop-grad, as in I-JEPA/BYOL). Its momentum is annealed from 0.996 to 1.0.
- **Predictor** g (MLP): predicts, in latent space, the target encoder's embedding of the reference record.

**Loss:**
- InfoNCE between f_ξ(S1) and g(f_θ(target)), with in-batch and hard negatives;
- \+ 1 − cos (latent regression);
- \+ a VICReg variance hinge, which prevents collapse;
- \+ a self-prediction term: a masked S1 view (token dropout, address dropped) must predict the full S1 view.

**Retrieval is asymmetric:** S1 is embedded with f_ξ, and S2/S3 with g∘f_θ.

**How it differs from A,** so the two make different mistakes:
- **Input view:** the **raw text in its original script** (NFKC, case-folded), not the transliterated template;
- **Backbone:** a different pretrained model;
- **Objective:** latent prediction, masked views and a variance regulariser;
- **Architecture:** asymmetric (A is symmetric).

**Parameter budget:** the context and target encoders **share a frozen token-embedding matrix**. That brings the whole model to about 139M unique parameters (96M shared embeddings + 2 × 21M transformer + predictor), below the 200M limit. The count is checked at build and load time (`utils.check_param_budget`), and loading fails if a model is at or above `models.max_params`.

## 4. Pipeline C: classical blocking

Each pass is a sparse TF-IDF top-K retrieval within a country, using the chunked, forked sparse product from `blocking.py`:

| Pass | Text | Default K |
|---|---|---|
| `addr` | Address words | 20 |
| `full` | Name + address words | 30 |
| `keys` | Blocking keys | 15 |
| `name` | char 3-grams of `name_core` | off (slow and weak in REPORT §3) |

**The keys:**
- first significant name token;
- 4-character name prefix;
- sorted pair of name tokens;
- last name token;
- house number + street word;
- address token pair;
- postal code.

Retrieval scores the IDF-weighted overlap of the keys, so **no single key needs to match exactly**. Features found in more than `max_df` of a country's targets are dropped, but never below `max_df_floor`. IDF is fitted on the data being blocked (unlabelled statistics only).

## 5. ANN retrieval (`pipelines/ann_index.py`)

- **Index types:** `auto` uses exact `IndexFlatIP` up to 200k vectors, and above that `IndexIVFScalarQuantizer` (8-bit) with nlist = 4√N and nprobe 48. `flat`, `ivf` and `hnsw` can be chosen in the config.
- **Memory:** embeddings are stored as float16 memmaps on disk (~5 GB for the 10M test targets at 256 dimensions) and added to the index in batches.
- **Isolation:** `ann.isolate_process: auto` runs the build and search in a spawned subprocess that never imports torch, whenever torch is loaded on macOS (see *macOS note*).
- **Saving indices:** `ann.save_indices: true` saves the FAISS indices under the cache directory.

## 6. Hard-negative mining

This step is configurable under `training.hard_negatives`. After the epochs listed in `mine_after_epochs`, the current model:
1. embeds up to `max_anchors` training S1 records and up to `max_targets` training targets (the anchors' own targets plus random ones);
2. retrieves each anchor's `top_m` nearest targets in its own country;
3. drops the anchor's true matches, skips the first `skip_top` of what is left, and keeps `per_anchor` of them.

The next epochs attach `per_batch_item` of these negatives to every pair. This is how "ABC Medical Store" and "ABC Medical Centre" end up in the same softmax. Only training data is used.

## 7. Union, priority and budget (`pipelines/candidate_union.py`)

`C_final = C_bert ∪ C_jepa ∪ C_classical`, deduplicated on (S1 row, target row). For every pair the union keeps:
- `score_<pass>` and `rank_<pass>` (NaN if that pass did not retrieve it);
- `hit_<family>`, `n_pipelines` and `n_passes`;
- RapidFuzz `rf_name` / `rf_addr` scores;
- `priority`.

**Prioritisers** are swappable (`candidate_generation.prioritizer`):
- `votes_rank` (default): `n_pipelines` + the mean over families of each family's best normalised rank. A pair found by two pipelines always ranks above one found by only one.
- `votes_score`
- `rrf`

**Budget:** a list is cut only if it is longer than `max_candidates_per_source1` (default 100), lowest priority first.
- `protect_min_pipelines: 2` never cuts a consensus candidate.
- `hard_budget: false` disables cutting altogether.
- `target_max_lists` (off by default) keeps each target only in its best m lists.

**Empty lists are allowed:** nothing is forced. Use `pipelines.*.min_similarity` to let dense pipelines return nothing for weak matches.

## 8. Outputs

| File | Content |
|---|---|
| `candidate_pairs.tsv` | `source1_entity_id \t candidate_entity_ids`, one row per S1 in file order, comma-separated S2/S3 ids. Ordered by priority, then id, so the output is deterministic |
| `debug_candidate_scores/part-*.parquet` | Provenance of every final pair, as listed in §7, plus ids and row indices |
| `candidate_manifest.json` | Row-order contract (checked by the matcher), build stats, the configuration used |
| `blocking_report.{txt,json}` | Only for labelled data: every metric below |

## 9. Validation methodology and metrics

**Holdout:** an entity-level closed universe (`local_val`): 10% of S1, with every target matched to those entities and 10% of the distractors.
- Holdout entities and their targets never produce training pairs.
- The transliteration dictionary is learned without them.
- Model selection uses recall@k on a sub-universe of the holdout (`training.eval_max_entities`). This slightly favours the holdout, but only for choosing the epoch.

**The report** (`blocking_metrics.py`; `check_blocking.py` computes the same for any candidate file) contains:
- **Candidate counts:** mean, median, p95 and max per S1.
- **Recall:**
  - pair recall;
  - macro recall per S1;
  - share of entities with every match found;
  - **F0.5 ceiling**: the leaderboard score of a perfect matcher on these candidates.
- **Pair quality** (share of candidate pairs that are real) and **reduction ratio**, both against the full product and against the within-country product.
- **Zero-candidate entities**, and how many true pairs they lose.
- **Singleton pollution:** singletons that still get candidates.
- **Per-pipeline view:**
  - recall of each pipeline alone and of each classical pass;
  - an exclusive Venn split of the true pairs (BERT only, JEPA only, classical only, each pair of pipelines, all three, missed by all).
- **Recall vs K** (`--k-sweep`, K = 10/20/30/50/100) and **recall @ candidate budget**.
- **Penalised scores**, so a blocker cannot win by returning everything (λ and the reference budget are configurable):
  - `penalized_recall` = PC − λ·(false candidates per S1)/ref_budget
  - its macro version
  - F1(PC, PQ)
  - H(PC, RR)

The default K values (20 per source for A/B; 20/30/15 for C) and the budget of 100 are **starting points**. Pick the final ones from the K sweep of your first `local_val` run.

## 10. Basic LightGBM matcher (`src/lgbm_matcher.py`)

- **Training data:** the blocker's candidates on `local_val`, which the encoders never trained on.
- **Evaluation:** grouped K-fold by S1 gives out-of-fold probabilities, and the threshold is tuned on them for F0.5 (singletons included).
- **Features:**
  - blocker provenance: every score and rank, `n_pipelines`, `priority`;
  - RapidFuzz similarity of names and addresses;
  - exact flags for name, `name_core`, address, house number and postal code;
  - list context: rank and gap within the S1 list, list length, target hub count;
  - record shape.
- **Country is not a feature:** France is unseen in training.
- **Decoding:** each target goes to its best S1 (`match.assign_and_cap`), then the threshold is applied. Matches are therefore always a subset of the candidates.

For a stronger matcher later, `features.py` + `match.py` (the two-stage model with expected-F0.5 decoding) can be pointed at these candidates.

## 11. Models, licences, parameter counts

| Use | Model | Licence | Parameters | Source |
|---|---|---|---|---|
| Pipeline A | `intfloat/multilingual-e5-small` + 384→256 projection | MIT | 117.7M (+0.1M) | huggingface.co/intfloat/multilingual-e5-small |
| Pipeline B | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, context + EMA target (shared embeddings) + predictor | Apache-2.0 | ≈139M unique | huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Matcher | LightGBM | MIT | trees | – |

- **Where the numbers come from:** licences and base parameter counts were read from the Hugging Face API (Sep 2026). The exact trained counts are written to `$ART/model_card.json`.
- **Data:** only the challenge files are used for training. Test inputs are only embedded and indexed; the IDF statistics are unsupervised.
- **External lookups:** none.

## 12. Compute requirements (estimates; nothing has been timed yet)

- **Encoding** is the dominant cost: each learned pipeline embeds every record once.
  - Test is ~12.3M records (1.7M S1 + 10M targets); `local_val` is ~1.25M.
  - Expect roughly 3–6k records/s on a T4 in fp16 → **~40–70 min per pipeline for test**.
  - On an Apple-silicon GPU (MPS) expect several times slower.
  - On CPU only it is impractical for test.
- **Training**, with the defaults (2 epochs × 400k pairs, batch 64, one mining round): ~12.5k steps per pipeline, i.e. **~1–2 h per pipeline on a T4**. `--train-fraction` and `max_pairs_per_epoch` scale this down linearly.
- **RAM:** ~8–10 GB for test is comfortable. Records and normalised text take ~5 GB; embeddings live on disk (float16 memmaps, ~2 × 5 GB of disk); an 8-bit IVF index for the largest country takes ~1.2 GB.
- **Candidate union:** processed in chunks of 100k S1 rows.
- **Matcher:** streams parquet parts, so memory is bounded.
- **Classical pipeline:** CPU only. Previously measured at 1 m 42 s for `local_fit` (330k S1) on a 10-core M4. Test is ~35× that work.

**macOS note:** the macOS wheels of torch, faiss and LightGBM each ship their own OpenMP. With all three in one process, multi-threaded faiss segfaults and LightGBM deadlocks. The code avoids this:
- faiss runs in a torch-free subprocess (`ann.isolate_process: auto`);
- `lgbm_matcher.py` never imports torch or faiss.

On Linux, Kaggle or Colab none of this applies.
