"""FastAPI backend + the single HTML search page.

Run from the project folder:
  python -m uvicorn api.app:app --port 8000
Then open http://127.0.0.1:8000 in a browser.
Endpoints (domain = movies | books | songs, default movies):
  GET /api/search?domain=movies&q=...&method=hybrid&k=10&offset=0&year_min=&year_max=&lang=hi&genre=comedy&why=true
  GET /api/similar/{item_id}?domain=movies&k=10&offset=0   (offset = "show more")
  GET /api/film/{item_id}?domain=movies      (full text of one item)
  GET /api/options?domain=movies             (groups, genres, methods, year range)
  GET /api/domains                           (which domains are built)
  GET /api/health
"""
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))  # engine.py and friends live there

from domains import DOMAINS  # noqa: E402
from engine import METHODS  # noqa: E402


MAX_OFFSET = 300  # "Show more" stops after this many results
RANK_ALL = MAX_OFFSET + 50  # always rank this many, so pages never overlap


def create_app(engine_factory):
    """Build the app. `engine_factory()` returns {domain: Engine}, or a single
    Engine, which is then used as "movies" (the tests pass a fake one)."""
    state = {}

    def engines():
        if "engines" not in state:  # load once, at startup (or first request)
            t = time.time()
            e = engine_factory()
            state["engines"] = e if isinstance(e, dict) else {"movies": e}
            state["load_seconds"] = round(time.time() - t, 2)
        return state["engines"]

    def eng(domain):
        if domain not in DOMAINS:
            raise HTTPException(400, f"domain must be one of {list(DOMAINS)}")
        if domain not in engines():
            raise HTTPException(404, f"the {domain} section isn't built yet")
        return engines()[domain]

    @asynccontextmanager
    async def lifespan(app):
        engines()  # load the model and vectors before serving requests
        yield

    app = FastAPI(title="Vibe search", version="0.2", lifespan=lifespan)

    @app.get("/")
    def page():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/health")
    def health():
        e = engines()
        return {"ok": True, "films": len(e["movies"].films) if "movies" in e else 0,
                "items": {d: len(x.films) for d, x in e.items()},
                "load_seconds": state.get("load_seconds")}

    @app.get("/api/domains")
    def domains():
        return {"built": list(engines()), "all": list(DOMAINS)}

    @app.get("/api/options")
    def options(domain: str = "movies"):
        e = eng(domain)
        years = [f["year"] for f in e.films]
        conf = DOMAINS[domain]
        present = sorted({f["language"] for f in e.films},
                         key=lambda c: list(conf["groups"]).index(c)
                         if c in conf["groups"] else 99)
        return {"domain": domain, "methods": METHODS,
                "languages": present,
                "groups": [{"code": c, "label": conf["groups"].get(c, c)} for c in present],
                "group_label": conf["group_label"],
                "genres": sorted({g for f in e.films for g in f["genres"]}),
                "year_min": min(years), "year_max": max(years), "count": len(e.films)}

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=1, max_length=300),
               domain: str = "movies",
               method: str = "hybrid", k: int = Query(10, ge=1, le=50),
               offset: int = Query(0, ge=0, le=MAX_OFFSET),
               year_min: int | None = None, year_max: int | None = None,
               lang: list[str] | None = Query(None), genre: list[str] | None = Query(None),
               why: bool = False):
        if method not in METHODS:
            raise HTTPException(400, f"method must be one of {METHODS}")
        e = eng(domain)
        t = time.time()
        # "Show more": always rank the same RANK_ALL items (so page 2 can't
        # repeat page 1), then return one page. Ranking is cheap; the "why"
        # sentences are only made for this page.
        results = e.search(q, method=method, k=RANK_ALL, year_min=year_min, year_max=year_max,
                           languages=lang, genres=genre)[offset:offset + k]
        cards = [e.film_card(i, s) for i, s in results]
        if why and results:
            for card, w in zip(cards, e.why_many(q, [i for i, _ in results])):
                card["why"] = w
        return {"query": q, "domain": domain, "method": method, "count": len(cards),
                "offset": offset, "more": len(cards) == k and offset + k <= MAX_OFFSET,
                "people": e.people_names(q),
                "took_ms": round((time.time() - t) * 1000), "results": cards}

    @app.get("/api/similar/{film_id}")
    def similar(film_id: str, domain: str = "movies", k: int = Query(10, ge=1, le=50),
                offset: int = Query(0, ge=0, le=MAX_OFFSET)):
        e = eng(domain)
        if film_id not in e.id_to_idx:
            raise HTTPException(404, "unknown id")
        src = e.film_card(e.id_to_idx[film_id])
        cards = [e.film_card(i, s) for i, s in e.more_like_this(film_id, k=RANK_ALL)[offset:offset + k]]
        return {"film": src, "offset": offset, "count": len(cards),
                "more": len(cards) == k and offset + k <= MAX_OFFSET, "results": cards}

    @app.get("/api/film/{film_id}")
    def film(film_id: str, domain: str = "movies"):
        e = eng(domain)
        if film_id not in e.id_to_idx:
            raise HTTPException(404, "unknown id")
        f = e.films[e.id_to_idx[film_id]]
        return {**e.film_card(e.id_to_idx[film_id]), "lead": f["lead"], "plot": f["plot"]}

    return app


def _real_engines():
    from engine import load_engines
    return load_engines(ROOT)


app = create_app(_real_engines)
