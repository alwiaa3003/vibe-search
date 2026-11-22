# Vibe search

Describe a mood or a situation ("cosy mystery for a rainy evening", "romantic songs for a long drive",
"a gripping crime story I can't put down") and get **movies, books or songs** that fit, instead of
exact keyword matches.

| Section | What's in it (fetched 2026-11-22) |
|---|---|
| **Movies** | 5,201 films from 2000-2026: 2,515 Bollywood (Hindi) and 2,686 Hollywood (American, the 100 best-known per year) |
| **Books** | 1,614 books from 2000-2026: 338 Indian books and 1,276 well-known English-language novels from many countries |
| **Songs** | 8,399 Indian songs from 2000-2026: 8,132 film songs from 1,320 films (7,031 Hindi, then Punjabi, Urdu, English, Marathi, Bengali) and 267 non-film singles |

All data comes from free, openly licensed sources (Wikidata, Wikipedia and MusicBrainz). Everything runs
on a laptop CPU. Only song titles and artists are shown: no lyrics and no audio.
South Indian films and songs (Tamil, Telugu, Malayalam, Kannada) are left out on purpose: this includes
Hindi films that also list one of those languages (for example Hindi + Telugu releases), 43 films in all.

> **Demo video:** _TODO: add link_

## What it does

- **Five search methods**, so they can be compared:
  - `bm25`: classic keyword search.
  - `emb_short`: one sentence-embedding per item (title + genres + first sentence + start of the plot).
  - `emb_chunks`: the plot is split into chunks of about 128 tokens, and an item scores as its best chunk.
  - `hybrid` (default): reciprocal rank fusion of the three lists above.
  - `hybrid_rerank`: a small cross-encoder re-scores the top 30 hybrid results (optional, slower).
- **Filters:** year range, genre, and a group (movies: Bollywood / Hollywood; books: Indian /
  international; songs: language).
- **More like this:** nearest items to a given item (for songs: songs from *other* films).
- **Artist and author names:** a query that names a singer or author ("Arijit Singh sad songs",
  "Amitav Ghosh") puts their items first, ranked by the rest of the query. Song credits like
  "Udit Narayan, Alka Yagnik" count for each person; one-word names ("KK", "Shaan") only count when the
  query is just that name, so a mood word can't trigger an artist by accident.
- **Show more:** results come 12 at a time, up to 300; every page is cut from the same ranking, so
  pages never repeat.
- **Why it matched:** the sentence closest to the query, shared words, and matching genres.
- **Evaluation tooling** (movies): blind pooled labelling, then Recall@10, MRR and nDCG@10 per method,
  plus a failure analysis.
- **FastAPI backend** and a single HTML page (Movies / Books / Songs tabs, mood suggestions, light and
  dark mode) with no build step.

## How songs are matched (and why)

MusicBrainz has song titles, artists, dates and languages for Indian music, but almost no mood tags for
film songs. So:
- **Film songs** are linked to their film (soundtrack title + year matched to a film we have Wikipedia
  text for) and are searched through **the film's story**. "Sad songs after a breakup" therefore finds
  songs from films about heartbreak; it can't tell which song *in* that film is the sad one. Results show
  at most 2 songs per film.
- **Non-film singles** only have MusicBrainz tags (pop, hip hop, sufi, ...), so they match style words
  better than moods.

## Architecture

```
 Wikidata SPARQL ──► fetch_wikidata.py / fetch_books.py ──► lists (title, year, language, genres, author)
 Wikipedia API ───► fetch_wikipedia.py / fetch_books.py ──► lead + Plot/Synopsis text (50 per request)
 MusicBrainz API ─► fetch_songs.py ──► Indian soundtracks + track lists, tagged singles
                         │
     clean_data.py / clean_books.py / clean_songs.py ──► data/clean/{films,books,songs}.jsonl
                         │                                 (one shared record format, see domains.py)
     build_index.py [movies|books|songs] ──► data/index*/  short.npy (1 vector per item)
     (all-MiniLM-L6-v2)                                    chunks.npy (several per item)
                         │
     engine.py ── BM25 + cosine search + RRF + optional cross-encoder + filters
                  + more-like-this + why-it-matched   (one Engine per domain, one shared model)
          ┌──────────────┼──────────────┬───────────────┐
    search_cli.py     api/app.py      pool.py         evaluate.py
    (terminal)        (FastAPI +      (blind           (metrics +
                      static page)    labelling sheet)  failure analysis)
```

Design notes:
- Text sent to the model is cut **only at sentence boundaries**, and its length is measured with the
  model's own tokenizer (254 tokens plus 2 special tokens = the 256 limit).
- Hybrid uses rank fusion, not score addition: BM25 and cosine scores aren't on the same scale.
- Filters are applied before ranking, so a filtered search still returns k results.
- Movies, books and songs use the same record format, so one engine and one page serve all three.
- The engine takes an `encode` function, so tests run with a fake 5-item engine in under a second.

## Data, licences and attribution

| Source | Used for | Licence |
|---|---|---|
| [Wikidata](https://www.wikidata.org) | movie and book lists, years, languages, genres, authors | CC0 1.0 |
| [English Wikipedia](https://en.wikipedia.org) | article leads and Plot/Synopsis sections (movies, books, and film songs via their film) | CC BY-SA 4.0, © Wikipedia contributors |
| [MusicBrainz](https://musicbrainz.org) | song titles, artists, dates, languages, soundtrack track lists | core data CC0 1.0 |
| MusicBrainz tags | style words for non-film singles | CC BY-NC-SA 3.0 (non-commercial, share-alike) |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | embeddings | Apache 2.0 |
| [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | optional re-ranker | Apache 2.0 (trained on MS MARCO, whose data terms are non-commercial) |

Descriptions are text from Wikipedia, written by Wikipedia contributors and licensed CC BY-SA 4.0. Every
movie and book links to its Wikipedia article; every song links to MusicBrainz (and film songs to their
film's article). Because MusicBrainz tags are CC BY-NC-SA, **the songs part is non-commercial**. See
[CREDITS.md](CREDITS.md).

**This repository contains no downloaded data**, only the scripts that fetch it. Data, vectors and the
labelling sheet stay on your machine (`data/`, `eval/labelling_sheet.csv`).

The fetch scripts are polite:
- Wikimedia ([User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy),
  [Robot policy](https://wikitech.wikimedia.org/wiki/Robot_policy)): a descriptive User-Agent with a
  contact URL, one request at a time at least 1 s apart (5 s after a slow one), gzip, `maxlag`,
  Retry-After, 50 titles per request.
- MusicBrainz ([rate limiting](https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting)): the same
  User-Agent, at most one request every 1.1 s, waiting on HTTP 503.
- Every response is cached locally, so nothing is downloaded twice.

## How to run it from scratch (Windows, Git Bash)

Requirements: Python 3.13, about 2.5 GB of free disk, and internet access for the one-time download
(about 2 hours in total, mostly waiting politely on the servers).

```bash
cd /c/Users/<you>/VibeSearch           # keep the project outside OneDrive
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt        # on Windows, PyPI's torch wheel is CPU-only

# 1. Say who you are: set CONTACT in scripts/config.py
#    (a URL such as your GitHub profile, or "User:YourWikiName")

# 2. Movies
python scripts/lookup_ids.py           # checks the Wikidata IDs used
python scripts/fetch_wikidata.py       # ~10 min: 54 small SPARQL queries
python scripts/fetch_wikipedia.py      # ~12 min: ~110 batched requests
python scripts/clean_data.py           # seconds: prints the coverage report
python scripts/build_index.py          # ~12 min on a laptop CPU (downloads the 90 MB model once)

# 3. Books
python scripts/fetch_books.py          # ~15 min: 81 SPARQL queries + ~35 batched requests
python scripts/clean_books.py
python scripts/build_index.py books

# 4. Songs (needs the movie data: film songs borrow their film's text)
python scripts/fetch_songs.py          # ~75 min: ~2,300 MusicBrainz requests
python scripts/clean_songs.py
python scripts/build_index.py songs

# 5. Search in the terminal
python scripts/search_cli.py "cosy mystery for a rainy evening" --why
python scripts/search_cli.py "a gripping crime story" --domain books
python scripts/search_cli.py "romantic songs for a long drive" --domain songs

# 6. Web page + API
python -m uvicorn api.app:app --port 8000   # wait for "Application startup complete"
# open http://127.0.0.1:8000

# 7. Tests
python -m pytest -q
```

API endpoints (`domain` = `movies`, `books` or `songs`; default `movies`):
`GET /api/search?domain=books&q=...&method=hybrid&k=10&lang=in&genre=fantasy&year_min=2010&why=true`,
`GET /api/similar/{id}?domain=...`, `GET /api/film/{id}?domain=...`, `GET /api/options?domain=...`,
`GET /api/domains`, `GET /api/health`.

## Evaluation (movies)

> **TODO: no results yet.** The table below will be filled in from real human labels. No numbers have
> been made up, estimated or generated by a model.

How to produce it (details in [LABELLING.md](LABELLING.md)):
1. Choose the queries in `eval/candidate_queries.csv` (40 candidates: mood, situation, vibe,
   Indian-context, hard).
2. `python scripts/pool.py` merges the top 10 of all 5 methods per query into a **blind, shuffled**
   labelling sheet. Method names aren't shown. (Current sheet: 38 queries, 1,065 rows.)
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

Known limits of this evaluation: one labeller, about 30-38 queries, pooled from the same methods (so
Recall is relative to the pool), labels judge Wikipedia descriptions, movies only, and the data is a
2026-11-22 snapshot. Books and songs have no evaluation yet.

## Resource use (measured on the development laptop, CPU only)

| | measured on 2026-11-22 |
|---|---|
| API startup (load model + vectors + BM25 for all three sections) | 12.7 s |
| API memory (working set), all three sections, without re-ranker | about 1.2 GB |
| Re-ranker | +about 100 MB RAM; about 1.5 s per query (measured on the earlier data) |
| Search latency (hybrid, warm) | 16-35 ms; 0.4-0.55 s with "why it matched" |
| Building the indexes (laptop CPU) | movies 12.4 min, books 3.8 min, songs 18.8 min |
| Disk: indexes (movies / books / songs) | 84 MB / 25 MB / 108 MB |
| Disk: clean data / interim / raw cache (only needed to rebuild) | 59 MB / 60 MB / 509 MB |
| Disk: Python packages (`.venv`, mostly PyTorch) | about 1.2 GB |
| Model weights (22.7M parameters x 4 bytes; each model) | about 91 MB |

**Hosting:** no host has been chosen and nothing is deployed. A host would need roughly 1.5 GB of RAM. A live demo would serve Wikipedia text
(allowed under CC BY-SA 4.0 with attribution and the same licence) and MusicBrainz tags (non-commercial
only); re-check both before going live. Docker has **not** been built or tested.

## Limitations

- **Movies:**
  - The Hollywood set is "famous films" (top 100 per year by number of Wikipedia language editions).
  - Hindi films have shorter texts: median plot about 380 words against about 640 for American films,
    and 10% of Hindi films have no Plot section. Fewer Hindi films after 2019 (28 from 2026).
  - Seen in example queries: title words over-match ("feel-good… bad day" returns *A Good Day to Die
    Hard*), negation ("nothing too violent") isn't understood, and "more like this" often returns
    sequels.
- **Books:**
  - International books are **novels** only (asking Wikidata for every kind of English book timed out).
  - Indian books are few (338) and 53% have no genre on Wikidata; 41% have no plot/synopsis section.
  - 17 Wikidata book items that linked to an author's or character's article were dropped.
  - Fewer books in recent years (Wikipedia articles appear later).
- **Songs:**
  - Film songs are found through their **film's** Wikipedia text, not the song itself, so all songs of
    one film look the same to the search (results show at most 2 per film). The song's own title helps
    only when it contains the query words ("Long Drive", "The Breakup Song").
  - Only soundtracks that could be matched to a film with Wikipedia text, and that aren't South Indian,
    are included: 1,320 films (from 7,860 Indian soundtrack releases). For Punjabi, Marathi and Bengali,
    the films come from the earlier 2010-2026 data, so those film songs are mostly from 2010 onwards.
  - MusicBrainz search returns at most 500 results per query: one query (2022, released in India) had
    563, so 63 soundtrack releases were not seen.
  - Artist credits are as MusicBrainz lists them, so they can include lyricists and composers, not only
    singers.
  - Name search only knows the names MusicBrainz credits (or Wikidata's author); 31 books have no
    author on Wikidata (for example some Salman Rushdie books), so their author's name doesn't find them.
  - Non-film singles (267) only have tags; self-promotion tags ("<artist> official", "best of ...") and
    tag-spam releases were removed, but tag quality is uneven.
  - A soundtrack is matched by title and year (within 1 year), so a few may be matched to the wrong film
    with the same name.
- The English-only model sees everything through English text.
- Text conversion drops wiki templates, so occasional sentences lose a word.

## Project layout

```
scripts/   fetch_*.py, clean_*.py, build_index.py, engine.py, domains.py, search_cli.py,
           demo_queries.py, pool.py, evaluate.py, metrics.py, text_utils.py,
           wiki_client.py, mb_client.py, config.py
api/       app.py (FastAPI)
static/    index.html (the search page)
tests/     pytest suite (toy engine + optional real-index checks)
eval/      candidate_queries.csv (+ local labelling sheet and results)
data/      local only: raw cache, interim, clean data, indexes
```
