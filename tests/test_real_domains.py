"""Checks against the REAL books and songs indexes. Each test is skipped if
that domain hasn't been built (see README: fetch, clean, build_index)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def built(domain):
    from domains import DOMAINS
    return (ROOT / DOMAINS[domain]["index"] / "meta.json").exists()


@pytest.fixture(scope="module")
def engines():
    from engine import load_engine
    return {d: load_engine(ROOT, d) for d in ("books", "songs") if built(d)}


@pytest.mark.parametrize("domain,query", [("books", "a gripping crime story"),
                                          ("songs", "romantic songs for a long drive")])
def test_real_domain_search(engines, domain, query):
    if domain not in engines:
        pytest.skip(f"{domain} index not built")
    e = engines[domain]
    res = e.search(query, method="hybrid", k=10)
    assert len(res) == 10
    for i, _ in res:
        f = e.films[i]
        assert f["title"] and f["creator"] and 2000 <= f["year"] <= 2026


def test_real_songs_at_most_two_per_film(engines):
    if "songs" not in engines:
        pytest.skip("songs index not built")
    e = engines["songs"]
    res = e.search("sad songs after a breakup", k=20)
    groups = [e.films[i].get("group_id") for i, _ in res if e.films[i].get("group_id")]
    assert all(groups.count(g) <= 2 for g in groups)


def test_real_book_groups(engines):
    if "books" not in engines:
        pytest.skip("books index not built")
    e = engines["books"]
    assert {f["language"] for f in e.films} == {"in", "intl"}
    res = e.search("family drama", k=10, languages=["in"])
    assert res and all(e.films[i]["indian"] for i, _ in res)
