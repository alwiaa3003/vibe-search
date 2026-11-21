"""Metric maths on tiny hand-made examples (values worked out by hand)."""
import math

import pytest

from metrics import dcg, mean, ndcg_at_k, recall_at_k, reciprocal_rank

LABELS = {"a": 2, "b": 0, "c": 1, "d": 2}  # relevant (>=1): a, c, d


def test_recall_counts_relevant_in_top_k():
    assert recall_at_k(["a", "b", "c"], LABELS, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b", "c"], LABELS, k=1) == pytest.approx(1 / 3)


def test_recall_with_strict_threshold():
    # only label 2 counts: a and d; top 3 contains a only
    assert recall_at_k(["a", "b", "c"], LABELS, k=3, min_rel=2) == pytest.approx(1 / 2)


def test_recall_none_when_no_relevant_items():
    assert recall_at_k(["x"], {"x": 0}, k=10) is None


def test_unlabelled_items_count_as_not_relevant():
    assert recall_at_k(["zzz", "a"], LABELS, k=1) == 0.0
    assert reciprocal_rank(["zzz", "b", "c"], LABELS) == pytest.approx(1 / 3)


def test_reciprocal_rank():
    assert reciprocal_rank(["b", "a"], LABELS) == pytest.approx(0.5)
    assert reciprocal_rank(["a"], LABELS) == 1.0
    assert reciprocal_rank(["b", "x"], LABELS) == 0.0
    assert reciprocal_rank(["b", "b2", "a"], LABELS, k=2) == 0.0  # beyond cut-off


def test_dcg_by_hand():
    # gains 2,0,1 -> (3/log2(2)) + 0 + (1/log2(4)) = 3 + 0.5
    assert dcg([2, 0, 1]) == pytest.approx(3.5)


def test_ndcg_perfect_and_worst():
    assert ndcg_at_k(["a", "d", "c", "b"], LABELS, k=4) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "x", "y"], LABELS, k=3) == 0.0


def test_ndcg_by_hand():
    ranked = ["b", "a", "c"]          # gains 0, 2, 1
    got = ndcg_at_k(ranked, LABELS, k=3)
    actual = 3 / math.log2(3) + 1 / math.log2(4)
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)  # 2, 2, 1
    assert got == pytest.approx(actual / ideal)


def test_ndcg_none_when_nothing_relevant():
    assert ndcg_at_k(["a"], {"a": 0}) is None


def test_mean_skips_none():
    assert mean([1.0, None, 0.0]) == pytest.approx(0.5)
    assert mean([None]) is None
