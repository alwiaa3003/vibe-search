"""Retrieval, fusion, filters, 'more like this' and 'why' on the toy collection."""
import numpy as np
import pytest

from engine import METHODS, RRF_K, tokenize


def ids(engine, results):
    return [engine.films[i]["id"] for i, _ in results]


@pytest.mark.parametrize("method", METHODS)
def test_every_method_finds_the_obvious_film(toy_engine, method):
    res = toy_engine.search("haunted mansion ghost", method=method, k=3)
    assert ids(toy_engine, res)[0] == "F1"


def test_scores_are_sorted_best_first(toy_engine):
    for method in METHODS:
        scores = [s for _, s in toy_engine.search("love wedding", method=method, k=5)]
        assert scores == sorted(scores, reverse=True)


def test_bm25_needs_shared_words(toy_engine):
    res = toy_engine.search("cricket", method="bm25", k=5)
    assert ids(toy_engine, res)[0] == "F3"
    assert res[0][1] > 0


def test_chunk_score_is_best_chunk(toy_engine):
    q = toy_engine.encode(["robot space"])[0]
    best = toy_engine.scores_chunks(q)
    i = toy_engine.id_to_idx["F4"]
    own = toy_engine.chunks[toy_engine.owner == i] @ q
    assert best[i] == pytest.approx(own.max())


def test_hybrid_is_reciprocal_rank_fusion(toy_engine):
    q = "detective murder rain"
    res = toy_engine.search(q, method="hybrid", k=5)
    # recompute RRF by hand from the three lists
    m = np.ones(len(toy_engine.films), dtype=bool)
    qv = toy_engine.encode([q])[0]
    lists = [toy_engine.top(toy_engine.scores_bm25(q), m, 100),
             toy_engine.top(toy_engine.scores_short(qv), m, 100),
             toy_engine.top(toy_engine.scores_chunks(qv), m, 100)]
    expected = {}
    for ranked in lists:
        for r, i in enumerate(ranked, 1):
            expected[i] = expected.get(i, 0) + 1 / (RRF_K + r)
    top_i, top_s = res[0]
    assert top_s == pytest.approx(max(expected.values()))
    assert toy_engine.films[top_i]["id"] == "F5"


def test_rerank_reorders_hybrid_candidates(toy_engine):
    res = toy_engine.search("cricket match team", method="hybrid_rerank", k=2)
    assert ids(toy_engine, res)[0] == "F3"


def test_filters_year_language_genre(toy_engine):
    e = toy_engine
    for method in METHODS:
        got = [e.films[i] for i, _ in e.search("love family", method=method, k=10,
                                                year_min=2016, year_max=2020)]
        assert got and all(2016 <= f["year"] <= 2020 for f in got)
        got = [e.films[i] for i, _ in e.search("love", method=method, k=10, languages=["en"])]
        assert got and all(f["language"] == "en" for f in got)
        got = [e.films[i] for i, _ in e.search("fear", method=method, k=10, genres=["mystery"])]
        assert got and all("mystery" in f["genres"] for f in got)


def test_filter_with_no_matches_returns_empty(toy_engine):
    assert toy_engine.search("ghost", method="hybrid", year_min=2030) == []


def test_k_larger_than_collection(toy_engine):
    assert len(toy_engine.search("love", method="emb_short", k=50)) == len(toy_engine.films)


def test_unknown_method_raises(toy_engine):
    with pytest.raises(ValueError):
        toy_engine.search("x", method="magic")


def test_more_like_this_excludes_itself(toy_engine):
    res = toy_engine.more_like_this("F2", k=4)
    assert "F2" not in ids(toy_engine, res)
    assert len(res) == 4


def test_why_finds_best_sentence_and_shared_words(toy_engine):
    i = toy_engine.id_to_idx["F3"]
    w = toy_engine.why("rain on the cricket match", i)
    assert "rain" in w["best_sentence"] or "match" in w["best_sentence"]
    assert {"rain", "cricket", "match"} <= set(w["shared_words"])


def test_why_candidates_come_from_best_chunk(toy_engine):
    i = toy_engine.id_to_idx["F1"]
    qv = toy_engine.encode(["ghost fear"])[0]
    cands = toy_engine._why_candidates(i, qv)
    assert toy_engine.films[i]["first_sentence"] not in cands
    assert any("ghost" in s for s in cands)
    assert not any(s.startswith("Bhoot Bungalow (") for s in cands)  # prefix removed


def test_why_matching_genre(toy_engine):
    w = toy_engine.why("a scary horror night", toy_engine.id_to_idx["F1"])
    assert w["matching_genres"] == ["horror"]


def test_tokenize_drops_stopwords():
    assert tokenize("A film about the Ghost in a mansion") == ["ghost", "mansion"]
