import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

TRAIN_S1 = os.path.join(DATA_DIR, 'train', 'train_source1.tsv')
TRAIN_S2 = os.path.join(DATA_DIR, 'train', 'train_source2.tsv')
TRAIN_S3 = os.path.join(DATA_DIR, 'train', 'train_source3.tsv')
TRAIN_GT = os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv')

TEST_S1 = os.path.join(DATA_DIR, 'test', 'test_source1.tsv')
TEST_S2 = os.path.join(DATA_DIR, 'test', 'test_source2.tsv')
TEST_S3 = os.path.join(DATA_DIR, 'test', 'test_source3.tsv')

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
