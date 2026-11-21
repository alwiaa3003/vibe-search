"""API tests with the toy engine (no model, no data needed)."""
import pytest
from fastapi.testclient import TestClient

from conftest import build_toy_engine

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(build_toy_engine)) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["films"] == 5


def test_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Vibe search" in r.text


def test_options(client):
    o = client.get("/api/options").json()
    assert "hybrid" in o["methods"] and o["year_min"] == 2012 and "hi" in o["languages"]


def test_search_returns_cards(client):
    r = client.get("/api/search", params={"q": "haunted ghost", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["id"] == "F1"
    assert {"title", "year", "language", "wiki_url", "score"} <= set(body["results"][0])


def test_search_with_filters_and_why(client):
    r = client.get("/api/search", params=[("q", "love"), ("lang", "en"), ("lang", "ta"),
                                          ("year_min", 2010), ("why", "true")])
    res = r.json()["results"]
    assert res and all(x["language"] in ("en", "ta") for x in res)
    assert "best_sentence" in res[0]["why"]


@pytest.mark.parametrize("params,code", [
    ({"q": ""}, 422), ({"q": "x", "k": 0}, 422), ({"q": "x", "k": 99}, 422),
    ({"q": "x", "method": "magic"}, 400), ({}, 422), ({"q": "x" * 301}, 422)])
def test_search_validation(client, params, code):
    assert client.get("/api/search", params=params).status_code == code


def test_similar(client):
    r = client.get("/api/similar/F2", params={"k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["film"]["id"] == "F2" and len(body["results"]) == 2
    assert all(x["id"] != "F2" for x in body["results"])


def test_unknown_film_is_404(client):
    assert client.get("/api/similar/NOPE").status_code == 404
    assert client.get("/api/film/NOPE").status_code == 404


def test_film_detail(client):
    f = client.get("/api/film/F3").json()
    assert f["title"] == "Last Over" and "cricket" in f["plot"]
