"""Checks the local scorer against the rules in the problem statement.
Run with:  python tests/test_evaluate.py   (or pytest)"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from evaluate import f_beta, score_candidates, score_matching


def test_problem_statement_example():
    pred = {'S2-00047', 'S2-00193', 'S3-00812'}
    truth = {'S2-00047', 'S3-00812'}
    assert round(f_beta(pred, truth), 3) == 0.714


def test_singletons():
    assert f_beta(set(), set()) == 1.0
    assert f_beta({'S2-1'}, set()) == 0.0


def test_no_correct_prediction_scores_zero():
    assert f_beta(set(), {'S2-1'}) == 0.0
    assert f_beta({'S2-2'}, {'S2-1'}) == 0.0


def test_perfect():
    assert f_beta({'S2-1', 'S3-2'}, {'S2-1', 'S3-2'}) == 1.0


def test_macro_average_and_missing_rows():
    gt = {'S1-a': {'S2-1'}, 'S1-b': set(), 'S1-c': {'S2-2', 'S3-3'}}
    preds = {'S1-a': {'S2-1'}, 'S1-b': {'S2-9'}}  # S1-c missing -> empty -> 0
    rep = score_matching(gt, preds)['ALL']
    assert abs(rep['f05'] - 1 / 3) < 1e-12
    assert rep['singleton_acc'] == 0.0
    assert rep['entities_with_fp'] == 1


def test_candidate_ceiling():
    gt = {'S1-a': {'S2-1', 'S2-2'}, 'S1-b': set()}
    cands = {'S1-a': {'S2-1', 'S2-7'}, 'S1-b': {'S2-8'}}
    rep = score_candidates(gt, cands)['ALL']
    # a: oracle keeps {S2-1}: P=1, R=0.5 -> F0.5 = 1.25*0.5/(0.25+0.5) = 0.8333; b: oracle predicts empty -> 1
    assert abs(rep['f05_ceiling'] - (0.8333333333333334 + 1) / 2) < 1e-12
    assert rep['pair_recall'] == 0.5
    assert rep['entities_full_recall'] == 0.0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'ok  {name}')
