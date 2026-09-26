# Business Entity Resolution Multi-Pipeline Blocking System

A production-quality, modular candidate generation and entity resolution system for large-scale multi-source commercial entity matching.

---

## Architecture Overview

```text
Raw Source TSVs (S1, S2, S3)
             ↓
Deterministic Normalization (Name, Address, Open-Set Country, PIN/Postal)
             ↓
┌───────────────────────────┬───────────────────────────┬───────────────────────────┐
│        Pipeline A         │        Pipeline B         │        Pipeline C         │
│     BERT Dual Encoder     │    JEPA Latent Blocker    │ Multi-Key Classical Block │
│  (MiniLM-L6-v2, <200M)    │ (Context/Target/Predictor)│ (Inverted Keys & Jaccard) │
│            ↓              │            ↓              │            ↓              │
│ ANN Vector Index (Top-K)  │ Indep. Vector Index (Top-K│ Key Bucketing & Overlap   │
└─────────────┬─────────────┴─────────────┬─────────────┴─────────────┬─────────────┘
              │                           │                           │
              └─────────────────────► Union ◄─────────────────────────┘
                               (C_final = C_A ∪ C_B ∪ C_C)
                                          ↓
                         Candidate Budget Prioritization
                               (Consensus + Score)
                                          ↓
                             output/candidate_pairs.tsv
                                          ↓
                       (Optional) LightGBM Precision Filter
                                          ↓
                            output/matching_results.tsv
```

### Key Highlights
1. **Three Independent Blocking Families:**
   - **Pipeline A (Fine-tuned BERT Dual Encoder):** Dense semantic representation (<200M parameters, Apache 2.0) trained via Multiple Negatives Ranking Loss (InfoNCE).
   - **Pipeline B (Second Learned Representation / JEPA):** Joint-Embedding Predictive Architecture with Context Encoder, EMA Target Encoder, and Predictor MLP head.
   - **Pipeline C (Classical Non-Neural Blocker):** Multi-key inverted index using character 3-grams, token Jaccard, address signatures, and PIN/postal code extraction.
2. **Strict Union (Never Intersection):** Guarantees high recall by combining candidates without loss.
3. **No Quadratic Comparison:** Employs ANN vector indices and inverted key buckets ($O(N \log N)$).
4. **Open-Set Country Support:** Gracefully handles US, India, France, and any unseen country strings.
5. **No Data Leakage:** Entity-level train/validation split strictly isolates Source 1 entities and their positive matches.

---

## Pretrained Models & Licensing Compliance

| Pipeline | Model Name | Parameter Count | License | Source |
| :--- | :--- | :--- | :--- | :--- |
| **Pipeline A (BERT)** | `sentence-transformers/all-MiniLM-L6-v2` | **22,713,216** (~22.7M) | Apache 2.0 | HuggingFace / UKP |
| **Pipeline B (JEPA / Learned)** | `google/electra-small-discriminator` | **13,483,008** (~13.5M) | Apache 2.0 | HuggingFace / Google |

Both models are strictly below the **200M parameter ceiling** and comply fully with challenge licensing requirements.

---

## Quickstart & Environment Setup

### 1. Create Virtual Environment and Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA GPU acceleration
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Training (Explicitly Invoked)
Training is **never run automatically** and is invoked via CLI:
```bash
# Full training
python -m business_entity_resolution.src.train_blocker \
    --train-dir dataset/train \
    --output-dir artifacts \
    --config configs/blocking.yaml

# Fast subset training (e.g. 10% sample)
python -m business_entity_resolution.src.train_blocker \
    --train-dir dataset/train \
    --output-dir artifacts \
    --config configs/blocking.yaml \
    --train-subset-ratio 0.1
```

### 2. Candidate Generation (Blocking)
Generates `output/candidate_pairs.tsv` and `output/debug_candidate_scores.tsv`:
```bash
python -m business_entity_resolution.src.blocking \
    --test-dir dataset/test \
    --artifacts-dir artifacts \
    --output-dir output \
    --config configs/blocking.yaml
```

### 3. Evaluate Candidate Quality Against Ground Truth
Runs candidate recall, reduction ratio, candidate precision, and penalty scoring:
```bash
python -m business_entity_resolution.src.evaluation.check_blocking \
    --candidates output/candidate_pairs.tsv \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --provenance output/debug_candidate_scores.tsv
```

### 4. Official Submission Validation
Validates formatting against official competition rules:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Running Unit and Integration Tests
```bash
python -m pytest tests/test_blocking_system.py -v
```
All 16 test suites verify:
- TSV parsing & normalization
- Leak-free train/val splits
- Positive & negative pair construction
- Embedding normalization & ANN top-K retrieval
- Classical inverted index blocking
- Candidate union, deduplication, and budget prioritization
- Output formatting & official constraint compliance
- LightGBM feature engineering & Macro $F_{0.5}$ evaluation
