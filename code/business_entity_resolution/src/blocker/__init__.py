"""
Multi-pipeline blocking / candidate generation.

  normalization  ->  Pipeline A (fine-tuned BERT bi-encoder, FAISS)
                 ->  Pipeline B (JEPA-style predictive encoder, FAISS)
                 ->  Pipeline C (TF-IDF passes + blocking keys, sparse top-K)
                 ->  union + dedup + prioritised budget  ->  candidate_pairs.tsv

Entry points live one level up (src/train_blocker.py, src/generate_candidates.py,
src/check_blocking.py, src/lgbm_matcher.py); docs in docs/BLOCKING.md.
"""

import os
import sys

# The package builds on the flat modules next to it (normalize, blocking,
# translit, evaluate, ...), which import each other by bare name.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# faiss, torch and lightgbm each ship their own OpenMP runtime in the macOS
# wheels. Loading two in one process aborts unless duplicates are allowed, and
# even then only torch + single-threaded faiss is stable (multi-threaded faiss
# or LightGBM next to torch segfaults / deadlocks). Hence: ANN search runs in a
# spawned torch-free subprocess (pipelines/ann_index.search_jobs), and
# lgbm_matcher.py never imports torch or faiss. Forked sparse-retrieval workers
# also need the ObjC fork-safety check off on macOS.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'YES')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

