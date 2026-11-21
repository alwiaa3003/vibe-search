"""Checks against the REAL data and index. Skipped if they haven't been built
(run fetch_wikidata.py, fetch_wikipedia.py, clean_data.py, build_index.py)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "data/index/meta.json").exists(),
                                reason="real index not built")


@pytest.fixture(scope="module")
def real():
    from engine import load_engine
    return load_engine(ROOT)


@pytest.mark.parametrize("method", ["bm25", "emb_short", "emb_chunks", "hybrid"])
def test_real_search_returns_10(real, method):
    assert len(real.search("cosy mystery for a rainy evening", method=method, k=10)) == 10


def test_real_filters(real):
    res = real.search("family drama", method="hybrid", k=10, languages=["ml"],
                      year_min=2015, year_max=2018)
    assert res
    for i, _ in res:
        f = real.films[i]
        assert f["language"] == "ml" and 2015 <= f["year"] <= 2018


def test_real_chunks_start_with_prefix(real):
    from text_utils import chunk_prefix
    for k, owner in enumerate(real.owner):
        assert real.chunk_texts[k].startswith(chunk_prefix(real.films[owner]) + ": ")


def test_real_vectors_are_unit_length(real):
    import numpy as np
    assert np.allclose(np.linalg.norm(real.short[:50], axis=1), 1, atol=1e-3)
    assert len(real.short) == len(real.films)
