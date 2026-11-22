"""Multi-domain features: songs capped per film, 'more like this' skips the same
film, and the API's domain parameter (toy data, no model needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FILMS, build_toy_engine, fake_encode
from engine import Engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.app import create_app  # noqa: E402


def song_engine(cap=2):
    """Three 'songs' share film G1's ghost story, one comes from film G2."""
    songs = []
    for n, (group, text) in enumerate([("G1", "ghost haunted mansion fear"),
                                       ("G1", "ghost haunted mansion fear"),
                                       ("G1", "ghost haunted mansion fear"),
                                       ("G2", "ghost haunted village")]):
        songs.append({**FILMS[0], "id": f"S{n}", "title": f"Song {n}", "group_id": group,
                      "language": "hi", "creator": "Singer", "plot": text, "lead": text,
                      "first_sentence": text})
    vecs = fake_encode([s["plot"] for s in songs])
    return Engine(songs, vecs, vecs, np.arange(len(songs), dtype=np.int32), fake_encode,
                  short_texts=[s["plot"] for s in songs], chunk_texts=[s["plot"] for s in songs],
                  max_per_group=cap)


def test_songs_are_capped_per_film():
    e = song_engine(cap=2)
    res = e.search("ghost haunted mansion", method="emb_short", k=4)
    groups = [e.films[i]["group_id"] for i, _ in res]
    assert groups.count("G1") == 2 and "G2" in groups


def test_more_like_this_skips_same_film():
    e = song_engine()
    res = e.more_like_this("S0", k=3)
    assert [e.films[i]["group_id"] for i, _ in res] == ["G2"]


def test_card_has_creator_and_subtitle():
    e = song_engine()
    e.films[0]["subtitle"] = "From the film X"
    card = e.film_card(0)
    assert card["creator"] == "Singer" and card["subtitle"] == "From the film X"


@pytest.fixture(scope="module")
def client():
    app = create_app(lambda: {"movies": build_toy_engine(), "songs": song_engine()})
    with TestClient(app) as c:
        yield c


def test_domains_endpoint(client):
    assert client.get("/api/domains").json()["built"] == ["movies", "songs"]


def test_unbuilt_and_unknown_domain(client):
    assert client.get("/api/search", params={"q": "x", "domain": "books"}).status_code == 404
    assert client.get("/api/search", params={"q": "x", "domain": "games"}).status_code == 400


def test_songs_search_and_options(client):
    r = client.get("/api/search", params={"q": "ghost", "domain": "songs", "k": 3}).json()
    assert r["domain"] == "songs" and r["results"][0]["creator"] == "Singer"
    o = client.get("/api/options", params={"domain": "songs"}).json()
    assert o["group_label"] == "Language" and o["groups"] == [{"code": "hi", "label": "Hindi"}]


def test_similar_with_domain(client):
    r = client.get("/api/similar/S0", params={"domain": "songs", "k": 3}).json()
    assert r["film"]["id"] == "S0" and all(x["id"] != "S0" for x in r["results"])


# ---------- artist / author names and "Show more" ----------
from engine import norm_name  # noqa: E402


def named_engine():
    e = song_engine()
    e.films[3]["creator"] = "A. R. Rahman, Shreya Ghoshal"
    e.films[1]["creator"] = "KK"
    return Engine(e.films, e.short, e.chunks, e.owner, fake_encode,
                  short_texts=e.short_texts, chunk_texts=e.chunk_texts, max_per_group=2)


def test_norm_name_joins_initials():
    assert norm_name("A. R. Rahman") == norm_name("AR Rahman") == "ar rahman"


def test_artist_name_puts_their_songs_first():
    e = named_engine()
    assert e.people_names("ar rahman ghost songs") == ["A. R. Rahman"]
    assert e.films[e.search("ar rahman ghost songs", method="emb_short", k=3)[0][0]]["id"] == "S3"
    assert e.films[e.search("shreya ghoshal", k=3)[0][0]]["id"] == "S3"   # co-singer counts


def test_one_word_artist_needs_the_whole_query():
    e = named_engine()
    assert e.people_names("kk songs") == ["KK"]
    assert e.people_names("kk ghost mansion") == []


def test_show_more_pages_do_not_repeat(client):
    p = {"q": "ghost", "domain": "movies", "method": "bm25", "k": 2}
    first = client.get("/api/search", params=p).json()
    second = client.get("/api/search", params={**p, "offset": 2}).json()
    assert first["offset"] == 0 and second["offset"] == 2
    ids1, ids2 = {r["id"] for r in first["results"]}, {r["id"] for r in second["results"]}
    assert ids1 and not ids1 & ids2
