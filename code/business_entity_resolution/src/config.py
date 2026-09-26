import os

# Paths default to the repo layout (<repo>/dataset, <repo>/output) and can be
# overridden per machine with environment variables, so the same code runs
# locally and on Colab (see README "Running on Colab"):
#   BER_DATA_DIR    raw + split TSVs         (Colab: fast local disk, /content/...)
#   BER_OUTPUT_DIR  submission files         (Colab: on Drive, survives disconnects)
#   BER_CACHE_DIR   intermediate artifacts   (features, embeddings, checkpoints)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get('BER_DATA_DIR', os.path.join(BASE_DIR, 'dataset'))
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
CACHE_DIR = os.environ.get('BER_CACHE_DIR', os.path.join(BASE_DIR, 'cache'))

IN_COLAB = 'COLAB_RELEASE_TAG' in os.environ
IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

TRAIN_S1 = os.path.join(DATA_DIR, 'train', 'train_source1.tsv')
TRAIN_S2 = os.path.join(DATA_DIR, 'train', 'train_source2.tsv')
TRAIN_S3 = os.path.join(DATA_DIR, 'train', 'train_source3.tsv')
TRAIN_GT = os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv')

TEST_S1 = os.path.join(DATA_DIR, 'test', 'test_source1.tsv')
TEST_S2 = os.path.join(DATA_DIR, 'test', 'test_source2.tsv')
TEST_S3 = os.path.join(DATA_DIR, 'test', 'test_source3.tsv')

OUTPUT_DIR = os.environ.get('BER_OUTPUT_DIR', os.path.join(BASE_DIR, 'output'))
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

# Local validation: fraction of S1 entities held out, and the seed that makes
# the hash-based assignment reproducible (see data_loader.py).
VAL_FRACTION = 0.10
SPLIT_SEED = 'mlc26'

SPLIT_NAMES = ('train', 'test', 'local_train', 'local_val', 'ce_train', 'scale_val')
# scale_val: ~20% of local_train's Source 1 (not ce_train's) against ALL
# local_train targets (data_loader.py --scale-val). Test-sized target pool, so
# the matcher is trained/scored with test-like crowding and IDF.
# ce_train: ~5% of local_train's Source 1 against ALL local_train targets
# (data_loader.py --ce-train). Training data for the cross-encoder that is
# disjoint from local_val and test, so its scores there are out-of-sample.


def split_paths(name):
    """
    Paths for a named split. Every split uses the same layout as the official
    train/test folders, so the pipeline can run on any of them unchanged:
      <dir>/<name>_source{1,2,3}.tsv  and  <dir>/<name>_ground_truth.tsv
    ('test' has no ground truth file.)
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}, expected one of {SPLIT_NAMES}")
    d = os.path.join(DATA_DIR, name) if name in ('train', 'test') else os.path.join(SPLITS_DIR, name)
    return {
        'dir': d,
        's1': os.path.join(d, f'{name}_source1.tsv'),
        's2': os.path.join(d, f'{name}_source2.tsv'),
        's3': os.path.join(d, f'{name}_source3.tsv'),
        'gt': os.path.join(d, f'{name}_ground_truth.tsv'),
    }
