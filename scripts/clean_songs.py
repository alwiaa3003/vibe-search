"""Songs, step 2: turn the MusicBrainz data into searchable song records.

FILM SONGS borrow their film's Wikipedia text (lead + plot), because
MusicBrainz has almost no mood tags for Indian soundtracks. So a film song is
found when its FILM fits the mood; songs of one film look alike to the search,
which is why results show at most 2 songs per film (domains.MAX_PER_GROUP).
NON-FILM SINGLES only have title, artist and tags (genre words).
Dropped: instrumentals, karaoke versions, dialogue tracks, repeated versions
of the same song in one film (remix/reprise/...), tag-spam singles.
Output: data/clean/songs.jsonl (local only, never committed).
Run:  python scripts/clean_songs.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

from domains import DOMAINS
from text_utils import GENRE_RULES, coarse_genres, sentences

OUT = Path("data/clean/songs.jsonl")
FILM_FILES = [Path("data/clean/films.jsonl"), Path("data/interim/films_2010_9lang.jsonl")]
LANG = {"hin": "hi", "pan": "pa", "tam": "ta", "tel": "te", "mal": "ml", "kan": "kn",
        "ben": "bn", "mar": "mr", "urd": "ur", "guj": "gu", "eng": "en"}
SKIP = re.compile(r"instrumental|karaoke|dialogue|jukebox|\btheme\b|\bbgm\b|"
                  r"background score|mashup|medley", re.I)
VERSION = re.compile(r"\s*[\(\[][^()\[\]]*[\)\]]\s*|\s+-\s+.*$")  # "(Reprise)", " - Remix"
SONG_GENRE_RULES = GENRE_RULES + [
    ("pop", "pop"), ("rock", "rock"), ("metal", "rock"), ("hip hop", "hip hop"),
    ("rap", "hip hop"), ("sufi", "sufi"), ("qawwali", "sufi"), ("ghazal", "ghazal"),
    ("devotional", "devotional"), ("bhajan", "devotional"), ("bhangra", "bhangra"),
    ("punjabi", "punjabi"), ("electronic", "electronic"), ("edm", "electronic"),
    ("house", "electronic"), ("classical", "classical"), ("carnatic", "classical"),
    ("hindustani", "classical"), ("folk", "folk"), ("indie", "indie"), ("love", "romance"),
    ("filmi", "filmi"), ("bollywood", "filmi"), ("jazz", "jazz"), ("r&b", "r&b"),
    ("soul", "r&b"), ("ambient", "electronic"), ("acoustic", "acoustic"),
]
MAX_TAGS = 25  # more tags than this on one single looks like tag spam


def load_films():
    films = {}
    for path in FILM_FILES:
        if path.exists():
            for line in path.open(encoding="utf-8"):
                f = json.loads(line)
                films.setdefault(f["id"], f)
    return films


def lang_name(code):
    return DOMAINS["songs"]["groups"].get(code, "")


def film_song_rows(raw, films):
    rows, seen, dropped = [], set(), Counter()
    for s in raw["film_songs"]:
        f = films.get(s["film_id"])
        title = (s["title"] or "").strip()
        if not f or not title:
            dropped["no film/title"] += 1
            continue
        if SKIP.search(title):
            dropped["instrumental/karaoke/dialogue"] += 1
            continue
        base = VERSION.sub(" ", title).strip().lower()
        if (f["id"], base) in seen or s["recording_id"] in seen:
            dropped["another version of the same song"] += 1
            continue
        seen.update({(f["id"], base), s["recording_id"]})
        lang = LANG.get(s["language"] or "") or f["language"]
        if lang not in DOMAINS["songs"]["groups"]:
            lang = "other"
        artist = s["artist"] or "unknown artist"
        first = (f"“{title}” is a song from the {f['year']} {lang_name(lang)} film "
                 f"{f['title']}, credited to {artist}.").replace("  ", " ")
        film_lead = " ".join(sentences(f["lead"])[:3])
        rows.append({
            "id": "S" + s["recording_id"], "title": title, "year": f["year"],
            "language": lang, "indian": True,
            "genres": f["genres"], "genres_fine": f["genres_fine"],
            "first_sentence": first,
            "lead": first + " About the film: " + film_lead,
            "plot": f["plot"] or " ".join(sentences(f["lead"])[3:]),
            "creator": artist, "subtitle": f"From the film {f['title']} ({f['year']})",
            "group_id": f["id"], "kind": "film song",
            "wiki_url": "https://musicbrainz.org/recording/" + s["recording_id"],
            "link_label": "Open on MusicBrainz",
            "extra_url": f["wiki_url"], "extra_label": "The film on Wikipedia",
        })
    return rows, dropped


def clean_tags(tags, title, artist):
    names = {title.lower(), artist.lower()}
    out = [t.lower().strip() for t in tags
           if 2 <= len(t) <= 30 and re.search(r"[a-z]", t.lower()) and t.lower() not in names]
    return list(dict.fromkeys(out))[:8]


def single_rows(raw):
    rows, seen, dropped = [], set(), Counter()
    for s in raw["singles"]:
        title, artist = (s["title"] or "").strip(), (s["artist"] or "").strip()
        year = int(s["date"][:4]) if s.get("date") and s["date"][:4].isdigit() else None
        if not title or not year or year < 2000:
            dropped["no title/year"] += 1
            continue
        # Indian = released in India, or sung in an Indian language. Urdu alone
        # isn't enough (it also matches Pakistani releases).
        if s.get("country") != "IN" and LANG.get(s.get("language") or "") in (None, "ur", "en"):
            dropped["not Indian (released elsewhere, non-Indian language)"] += 1
            continue
        if len(s["tags"]) > MAX_TAGS:
            dropped["tag spam"] += 1
            continue
        tags = clean_tags(s["tags"], title, artist)
        if not tags:
            dropped["no usable tags"] += 1
            continue
        key = (VERSION.sub(" ", title).strip().lower(), artist.lower())
        if key in seen or SKIP.search(title):
            dropped["duplicate/instrumental"] += 1
            continue
        seen.add(key)
        lang = LANG.get(s["language"] or "", "other")
        first = f"“{title}” is a {year} single by {artist}."
        rows.append({
            "id": "S" + s["release_id"], "title": title, "year": year,
            "language": lang, "indian": True,
            "genres": coarse_genres(tags, SONG_GENRE_RULES), "genres_fine": tags,
            "first_sentence": first,
            "lead": first + f" Style: {', '.join(tags)}.", "plot": "",
            "creator": artist, "subtitle": "Single (not from a film)", "kind": "single",
            "wiki_url": "https://musicbrainz.org/release/" + s["release_id"],
            "link_label": "Open on MusicBrainz",
        })
    return rows, dropped


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raw = json.loads(Path("data/interim/songs_raw.json").read_text(encoding="utf-8"))
    films = load_films()
    a, da = film_song_rows(raw, films)
    b, db = single_rows(raw)
    rows = a + b
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"MusicBrainz fetched {raw['fetched']}: {raw['soundtrack_releases']} soundtrack "
          f"releases, {raw['films_matched']} matched to a film with Wikipedia text, "
          f"{raw['failed_lookups']} failed lookups")
    print(f"Film songs kept {len(a)} from {len(raw['film_songs'])} tracks; dropped {dict(da)}")
    print(f"Singles kept {len(b)} from {len(raw['singles'])}; dropped {dict(db)}")
    print("By language:", dict(Counter(r["language"] for r in rows).most_common()))
    print("Films with songs:", len({r['group_id'] for r in a}),
          "| songs per film median:",
          sorted(Counter(r["group_id"] for r in a).values())[len(a) and len({r['group_id'] for r in a}) // 2]
          if a else 0)
    print("By year:", dict(sorted(Counter(r["year"] for r in rows).items())))
    print("Single genres:", dict(Counter(g for r in b for g in r["genres"]).most_common(15)))
    for r in a[:3] + b[:3]:
        print(f"- {r['title']} | {r['creator'][:50]} | {r['subtitle']} | {r['genres_fine'][:4]}")
    print(f"Saved {len(rows)} songs to {OUT}")
