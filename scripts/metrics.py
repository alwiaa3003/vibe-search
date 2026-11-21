"""Ranking metrics: Recall@k, MRR and nDCG@k.

`ranked` is the list of item ids a method returned, best first.
`labels` maps item id -> graded label (0 = not relevant, 1 = partly fits,
2 = fits well). Items with no label count as not relevant (0).
For Recall and MRR an item is "relevant" if its label >= `min_rel` (default 1).
"""
import math


def recall_at_k(ranked, labels, k=10, min_rel=1):
    """Share of all known relevant items that appear in the top k.

    Returns None when the query has no relevant items at all (it can't be scored).
    """
    relevant = {i for i, g in labels.items() if g >= min_rel}
    if not relevant:
        return None
    found = sum(1 for i in ranked[:k] if i in relevant)
    return found / len(relevant)


def reciprocal_rank(ranked, labels, k=None, min_rel=1):
    """1 / rank of the first relevant item (0 if none in the list / top k)."""
    for rank, i in enumerate(ranked[:k] if k else ranked, start=1):
        if labels.get(i, 0) >= min_rel:
            return 1.0 / rank
    return 0.0


def dcg(gains):
    """Discounted cumulative gain with gain (2^g - 1) and log2(rank + 1) discount."""
    return sum((2 ** g - 1) / math.log2(rank + 1) for rank, g in enumerate(gains, 1))


def ndcg_at_k(ranked, labels, k=10):
    """DCG of the top k divided by the best possible DCG (ideal ordering).

    Returns None when no item has a label > 0 (nothing to rank well).
    """
    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    if ideal == 0:
        return None
    return dcg([labels.get(i, 0) for i in ranked[:k]]) / ideal


def mean(values):
    """Mean of the non-None values (None = query could not be scored)."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
