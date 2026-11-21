# Vibe search

Describe a mood or a situation ("cosy mystery for a rainy evening", "something to watch
with my grandparents") and get films that fit, instead of exact keyword matches.

The collection is about 8,200 films from 2010-2026: roughly 80% Indian (Hindi, Tamil, Telugu,
Malayalam, Kannada, Bengali, Marathi, Punjabi) and 20% English. All data comes from free, openly
licensed sources (Wikidata and Wikipedia). Everything runs on a laptop CPU.

> **Demo video:** _TODO: add link_

## What it does

- **Five search methods**, so they can be compared:
  - `bm25`: classic keyword search.
  - `emb_short`: one sentence-embedding per film (title + genres + first lead sentence + plot start).
  - `emb_chunks`: the plot is split into chunks of about 128 tokens, and a film scores as its best chunk.
  - `hybrid` (default): reciprocal rank fusion of the three lists above.
  - `hybrid_rerank`: a small cross-encoder re-scores the top 30 hybrid results (optional, slower).
- **Filters:** year range, original language, genre.
- **More like this:** nearest films to a given film.
- **Why it matched:** the plot sentence closest to the query, shared words, and matching genres.
- **Evaluation tooling:** blind pooled labelling, then Recall@10, MRR and nDCG@10 per method, plus a
  failure analysis.
- **FastAPI backend** and a single HTML page with no build step.

## Architecture

```
 Wikidata SPARQL ──► fetch_wikidata.py ──► film list (title, year, language, genres, article)
                                                │
 Wikipedia API ───► fetch_wikipedia.py ──► lead + Plot text (50 articles/request, cached)
                                                │
                    clean_data.py ──────► data/clean/films.jsonl  (+ coverage report)
                                                │
                    build_index.py ─────► data/index/  short.npy (1 vector per film)
                    (all-MiniLM-L6-v2)               chunks.npy (5.3 vectors per film)
                                                │
                    engine.py  ── BM25 (rank-bm25) + cosine search + RRF + optional cross-encoder
                       │          + filters + more-like-this + why-it-matched
          ┌────────────┼──────────────┬───────────────┐
    search_cli.py   api/app.py     pool.py         evaluate.py
    (terminal)      (FastAPI +     (blind           (metrics +
                    static page)   labelling sheet)  failure analysis)
```

Design notes:
- Text sent to the model is cut **only at sentence boundaries**, and its length is measured with the
  model's own tokenizer (254 tokens plus 2 special tokens = the 256 limit). Nothing is silently truncated.
- Hybrid uses rank fusion, not score addition: BM25 and cosine scores aren't on the same scale.
- Filters are applied before ranking, so a filtered search still returns k results.
- The engine takes an `encode` function, so tests run with a fake 5-film engine in under a second.

## Data, licences and attribution

| Source | Used for | Licence |
|---|---|---|
| [Wikidata](https://www.wikidata.org) | film list, year, original language, genres, article link | CC0 1.0 |
| [English Wikipedia](https://en.wikipedia.org) | article lead and Plot section | CC BY-SA 4.0, © Wikipedia contributors |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | embeddings | Apache 2.0 |
| [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | optional re-ranker | Apache 2.0 (trained on MS MARCO, whose data terms are non-commercial) |

Film descriptions are text from Wikipedia, written by Wikipedia contributors and licensed CC BY-SA 4.0.
Every result links to its Wikipedia article, where the full author history is available. See
[CREDITS.md](CREDITS.md).

**This repository contains no downloaded data**, only the scripts that fetch it. Data, vectors and the
labelling sheet stay on your machine (`data/`, `eval/labelling_sheet.csv`).

The fetch scripts follow Wikimedia's
[User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy)
and [Robot policy](https://wikitech.wikimedia.org/wiki/Robot_policy):
- a descriptive User-Agent with a contact URL;
- one request at a time, at least 1 s apart, with 5 s after any slow request;
- gzip, `maxlag`, and respecting Retry-After;
- 50 titles per request;
- every response cached locally.

## How to run it from scratch (Windows, Git Bash)

Requirements: Python 3.13, about 2 GB of free disk, and internet access for the one-time download
(about 25-45 minutes in total, mostly waiting politely on Wikimedia).

```bash
cd /c/Users/<you>/VibeSearch           # keep the project outside OneDrive
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt        # on Windows, PyPI's torch wheel is CPU-only

# 1. Tell Wikimedia who you are: set CONTACT in scripts/config.py
#    (a URL such as your GitHub profile, or "User:YourWikiName")

# 2. Fetch and prepare the data (cached, so reruns are fast)
python scripts/lookup_ids.py           # checks the Wikidata IDs used
python scripts/fetch_wikidata.py       # ~20 min: 153 small SPARQL queries
python scripts/fetch_wikipedia.py      # ~25 min: ~180 batched requests
python scripts/clean_data.py           # 2 s: prints the coverage report

# 3. Build the vectors (downloads the 90 MB model once)
python scripts/build_index.py          # ~17 min on a laptop CPU

# 4. Search
python scripts/search_cli.py "cosy mystery for a rainy evening" --why
python scripts/search_cli.py "village revenge drama" --lang ta te --year 2015-2020
python scripts/search_cli.py --like Q15401703          # more like Drishyam (2013)

# 5. Web page + API
python -m uvicorn api.app:app --port 8000   # wait for "Application startup complete" (~25 s)
# open http://127.0.0.1:8000

# 6. Tests
python -m pytest -q
```

API endpoints: `GET /api/search?q=...&method=hybrid&k=10&lang=hi&genre=comedy&year_min=2015&why=true`,
`GET /api/similar/{wikidata_id}`, `GET /api/film/{wikidata_id}`, `GET /api/options`, `GET /api/health`.

## Evaluation

> **TODO: no results yet.** The table below will be filled in from real human labels. No numbers have
> been made up, estimated or generated by a model.

How to produce it (details in [LABELLING.md](LABELLING.md)):
1. Choose about 30 queries in `eval/candidate_queries.csv` (40 candidates: mood, situation, vibe,
   Indian-context, hard).
2. `python scripts/pool.py` merges the top 10 of all 5 methods per query into a **blind, shuffled**
   labelling sheet. Method names aren't shown.
3. Label every row 0 / 1 / 2 (doesn't fit / partly / fits well).
4. `python scripts/evaluate.py` writes `eval/results.md`: the table, results by query type, per-query
   scores, a failure analysis and the limits.

| method | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|
| bm25 | TODO | TODO | TODO |
| emb_short | TODO | TODO | TODO |
| emb_chunks | TODO | TODO | TODO |
| hybrid | TODO | TODO | TODO |
| hybrid_rerank | TODO | TODO | TODO |

Known limits of this evaluation: one labeller, about 30 queries, pooled from the same methods (so
Recall is relative to the pool), labels judge Wikipedia descriptions, and the data is a 2026-11-21 snapshot.

## Resource use (measured on the development laptop, CPU only)

| | measured |
|---|---|
| API startup (load model + vectors + BM25) | 26 s |
| API memory (resident), without re-ranker | 876 MB |
| Re-ranker | +about 100 MB RAM; about 1.5 s per query |
| Search latency (hybrid, warm) | 40-60 ms; about 0.5 s with "why it matched" |
| Disk: `data/index` / `data/clean` | 103 MB / 28 MB |
| Disk: raw cache + interim (only needed to rebuild) | 218 MB + 27 MB |
| Disk: Python packages (`.venv`, mostly PyTorch) | 1.2 GB |
| Model weights (calculated: 22.7M parameters × 4 bytes; each model) | about 91 MB |
| Building the index | 16.5 min |

**Hosting:** no host has been chosen and nothing is deployed. Notes for later:
- A host would need roughly 1 GB of RAM, mostly PyTorch plus the model.
- Ways to shrink that (an ONNX or quantised model, or precomputing more) are untested.
- A live demo would serve Wikipedia text from a server. That's allowed under CC BY-SA 4.0 with
  attribution and the same licence for the text, but the obligations should be re-checked before going
  live. Docker has **not** been built or tested.

## Limitations

- **Unequal text quality:**
  - The median Indian plot is about 300 words, against about 640 for English.
  - 19% of Indian films have no Plot section, only the lead.
  - 10% have no genre at all.
  - Search probably works better for English films (the evaluation should confirm or refute this).
- **English sample is "famous films":** the top 100 per year by number of Wikipedia language editions.
- **Coverage drops after 2019**, and only 83 Indian films are from 2026.
- **The English-only model** sees Indian films only through their English Wikipedia text.
- **Seen in example queries:**
  - Title words over-match (e.g. "feel-good… bad day" returns *Happy Death Day*).
  - Negation ("nothing too violent") isn't understood.
  - "More like this" mostly returns remakes and sequels.
- Films whose Wikidata item has no original-language entry are missed: 443 Indian-origin films,
  measured on 2026-11-21.
- Text conversion drops wiki templates, so occasional sentences lose a word.

## Project layout

```
scripts/   fetch_*.py, clean_data.py, build_index.py, engine.py, search_cli.py,
           demo_queries.py, pool.py, evaluate.py, metrics.py, text_utils.py, wiki_client.py, config.py
api/       app.py (FastAPI)
static/    index.html (the search page)
tests/     pytest suite (toy engine + optional real-index checks)
eval/      candidate_queries.csv (+ local labelling sheet and results)
data/      local only: raw cache, interim, clean data, index
```

If you put this in git, keep `data/`, `.venv/`, `.env` and `eval/labelling_sheet*.csv` (it contains
Wikipedia text) out of the repository, e.g. with a `.gitignore`. `eval/results.md` holds only titles
and scores, so it can be committed.
