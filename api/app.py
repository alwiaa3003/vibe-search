"""Phase 6: FastAPI backend + the single HTML search page.

Run from the project folder:
  uvicorn api.app:app --port 8000
Then open http://127.0.0.1:8000 in a browser.
Endpoints:
  GET /api/search?q=...&method=hybrid&k=10&year_min=&year_max=&lang=hi&lang=ta&genre=comedy&why=true
  GET /api/similar/{film_id}?k=10
  GET /api/film/{film_id}
  GET /api/options     (languages, genres, methods, year range for the page)
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

from engine import METHODS  # noqa: E402


def create_app(engine_factory):
    """Build the app. `engine_factory()` returns an Engine (tests pass a fake one)."""
    state = {}

    def eng():
        if "engine" not in state:  # load once, at startup (or first request)
            t = time.time()
            state["engine"] = engine_factory()
            state["load_seconds"] = round(time.time() - t, 2)
        return state["engine"]

    @asynccontextmanager
    async def lifespan(app):
        eng()  # load the model and vectors before serving requests
        yield

    app = FastAPI(title="Vibe search", version="0.1", lifespan=lifespan)

    @app.get("/")
    def page():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "films": len(eng().films),
                "load_seconds": state.get("load_seconds")}

    @app.get("/api/options")
    def options():
        e = eng()
        years = [f["year"] for f in e.films]
        return {"methods": METHODS,
                "languages": sorted({f["language"] for f in e.films}),
                "genres": sorted({g for f in e.films for g in f["genres"]}),
                "year_min": min(years), "year_max": max(years)}

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=1, max_length=300),
               method: str = "hybrid", k: int = Query(10, ge=1, le=50),
               year_min: int | None = None, year_max: int | None = None,
               lang: list[str] | None = Query(None), genre: list[str] | None = Query(None),
               why: bool = False):
        if method not in METHODS:
            raise HTTPException(400, f"method must be one of {METHODS}")
        e = eng()
        t = time.time()
        results = e.search(q, method=method, k=k, year_min=year_min, year_max=year_max,
                           languages=lang, genres=genre)
        cards = [e.film_card(i, s) for i, s in results]
        if why and results:
            for card, w in zip(cards, e.why_many(q, [i for i, _ in results])):
                card["why"] = w
        return {"query": q, "method": method, "count": len(cards),
                "took_ms": round((time.time() - t) * 1000), "results": cards}

    @app.get("/api/similar/{film_id}")
    def similar(film_id: str, k: int = Query(10, ge=1, le=50)):
        e = eng()
        if film_id not in e.id_to_idx:
            raise HTTPException(404, "unknown film id")
        src = e.film_card(e.id_to_idx[film_id])
        return {"film": src, "results": [e.film_card(i, s)
                                         for i, s in e.more_like_this(film_id, k=k)]}

    @app.get("/api/film/{film_id}")
    def film(film_id: str):
        e = eng()
        if film_id not in e.id_to_idx:
            raise HTTPException(404, "unknown film id")
        f = e.films[e.id_to_idx[film_id]]
        return {**e.film_card(e.id_to_idx[film_id]), "lead": f["lead"], "plot": f["plot"]}

    return app


def _real_engine():
    from engine import load_engine
    return load_engine(ROOT)


app = create_app(_real_engine)
