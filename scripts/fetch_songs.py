"""Songs, step 1: find Indian songs (2000+) on MusicBrainz.

Two kinds of songs:
  1. FILM SONGS. Indian soundtrack releases (country India or an Indian language,
     release date 2000+). A soundtrack is kept only if it can be matched to a
     film we have Wikipedia text for (same title, year within 1). We then fetch
     its track list. The film's text gives the song its "mood" (MusicBrainz has
     almost no mood tags for Indian soundtracks).
  2. NON-FILM SINGLES. Indian singles (2000+) that have at least one MusicBrainz
     tag (e.g. "pop", "sufi", "hip hop"). Their only searchable text is title,
     artist and tags, so they match genre words better than moods.
Only titles, artists, dates, languages and tags are used: no lyrics, no audio.
Licences: MusicBrainz core data CC0; tags CC BY-NC-SA 3.0 (see CREDITS.md).
Output: data/interim/songs_raw.json (local only, never committed).
Run (after the movie data is built):  python scripts/fetch_songs.py
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from mb_client import mb_get

INDIAN_LANGS = {"hin": "hi", "pan": "pa", "tam": "ta", "tel": "te", "mal": "ml",
                "kan": "kn", "ben": "bn", "mar": "mr", "urd": "ur", "guj": "gu"}
# "Indian" = released in India OR in an Indian language.
PLACES = ["country:IN"] + [f"lang:{c}" for c in INDIAN_LANGS]
Q_SOUNDTRACK = "{where} AND secondarytype:soundtrack AND date:[{y} TO {y}]"
Q_SINGLES = ("{where} AND primarytype:single AND NOT secondarytype:soundtrack "
             "AND tag:* AND date:[{y} TO {y}]")
YEARS = range(2000, 2027)
MAX_RESULTS = 500  # MusicBrainz search returns at most 500 results per query
FILM_FILES = [Path("data/clean/films.jsonl"),               # Hindi + American, 2000+
              Path("data/interim/films_2010_9lang.jsonl")]  # older set: 9 Indian langs, 2010+
OUT = Path("data/interim/songs_raw.json")


def search_pages(query):
    """Page through ONE release search (100 per request, max 500 in total)."""
    out, offset, total = [], 0, None
    while (total is None or offset < total) and offset < MAX_RESULTS:
        d = mb_get("release", {"query": query, "limit": 100, "offset": offset})
        total = d["count"]
        out += d["releases"]
        offset += 100
        if not d["releases"]:
            break
    return out, total


def search_all(template):
    """Run the search per year; if a year has > 500 hits, split it by place."""
    found, capped = {}, []
    for y in YEARS:
        q = template.format(where="(" + " OR ".join(PLACES) + ")", y=y)
        rows, total = search_pages(q)
        if total > MAX_RESULTS:  # too many: one query per place instead
            rows = []
            for place in PLACES:
                part, t = search_pages(template.format(where=place, y=y))
                rows += part
                if t > MAX_RESULTS:
                    capped.append((y, place, t))
        for r in rows:
            found[r["id"]] = r
    print(f"  {len(found)} releases; queries cut at 500 results: {capped or 'none'}",
          flush=True)
    return list(found.values())


def norm_title(t):
    """'Kill Dil (Original Motion Picture Soundtrack)' -> 'kill dil'."""
    t = t.lower()
    t = re.sub(r"\((?:[^)]*(soundtrack|ost|motion picture|hindi|tamil|telugu|"
               r"malayalam|kannada|film|from)[^)]*)\)", " ", t)
    t = re.sub(r"[:\-–]\s*(original )?(motion picture )?(soundtrack|ost|music from).*$", " ", t)
    t = re.sub(r"\b(original motion picture soundtrack|soundtrack|ost)\b", " ", t)
    t = re.sub(r"\(\s*\d{4}\s*(film)?\s*\)", " ", t)
    return " ".join(re.findall(r"[a-z0-9]+", t))


def year_of(date):
    return int(date[:4]) if date and date[:4].isdigit() else None


def artists(credit):
    return ", ".join(a.get("name") or a["artist"]["name"] for a in credit or [])


def load_films():
    """Films we have Wikipedia text for, indexed by normalised title."""
    by_title, seen = defaultdict(list), set()
    for path in FILM_FILES:
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            f = json.loads(line)
            if f["id"] in seen:
                continue
            seen.add(f["id"])
            by_title[norm_title(f["title"])].append(f)
    return by_title


def match_film(release, films_by_title):
    """Return the film this soundtrack belongs to, or None."""
    y = year_of(release.get("date"))
    lang = INDIAN_LANGS.get(release.get("text-representation", {}).get("language"))
    best = None
    for f in films_by_title.get(norm_title(release["title"]), []):
        if f.get("language") == "us" or not f.get("indian", True):
            continue  # American films: not Indian songs
        if y is None or abs(f["year"] - y) > 1:
            continue
        score = (lang is not None and lang in (f.get("all_languages") or [f["language"]]),
                 -abs(f["year"] - y))
        if best is None or score > best[0]:
            best = (score, f)
    return best[1] if best else None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    print("Searching MusicBrainz (1 request per second, cached)...")
    soundtracks = search_all(Q_SOUNDTRACK)
    singles = search_all(Q_SINGLES)

    # 1) film songs: match each soundtrack to a film, one release per film
    films_by_title = load_films()
    print(f"Films with text to match against: {sum(len(v) for v in films_by_title.values())}")
    per_film = {}
    for r in soundtracks:
        f = match_film(r, films_by_title)
        if f is None:
            continue
        key = (r.get("country") == "IN", r.get("track-count") or 0)
        if f["id"] not in per_film or key > per_film[f["id"]][0]:
            per_film[f["id"]] = (key, r, f)
    print(f"Soundtracks matched to a film: {len(per_film)} films "
          f"(of {len(soundtracks)} soundtrack releases)", flush=True)

    film_songs, failed = [], 0
    for n, (fid, (_, r, f)) in enumerate(sorted(per_film.items()), 1):
        try:
            full = mb_get("release/" + r["id"], {"inc": "recordings+artist-credits"})
        except RuntimeError:
            failed += 1
            if failed >= 5:
                raise SystemExit("5 failed lookups: stopping (rerun resumes from the cache).")
            continue
        if full is None:
            continue
        for medium in full.get("media", []):
            for t in medium.get("tracks", []):
                film_songs.append({
                    "recording_id": t["recording"]["id"], "release_id": r["id"],
                    "title": t["title"], "artist": artists(t.get("artist-credit")),
                    "date": full.get("date") or r.get("date"),
                    "language": (full.get("text-representation") or {}).get("language"),
                    "film_id": fid, "length_ms": t.get("length")})
        if n % 200 == 0:
            print(f"  {n}/{len(per_film)} track lists ({time.time() - t0:.0f}s)", flush=True)

    # 2) non-film singles with tags (the search result already has what we need)
    single_rows = [{
        "release_id": r["id"], "title": r["title"],
        "artist": artists(r.get("artist-credit")), "date": r.get("date"),
        "language": (r.get("text-representation") or {}).get("language"),
        "country": r.get("country"),
        "tags": [t["name"] for t in r.get("tags", [])]} for r in singles]

    OUT.write_text(json.dumps({
        "fetched": time.strftime("%Y-%m-%d"), "queries": [Q_SOUNDTRACK, Q_SINGLES],
        "places": PLACES,
        "soundtrack_releases": len(soundtracks), "films_matched": len(per_film),
        "failed_lookups": failed, "film_songs": film_songs, "singles": single_rows},
        ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(film_songs)} film-song tracks and {len(single_rows)} singles "
          f"to {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
