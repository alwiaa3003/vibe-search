"""Books, step 2: clean the fetched books and report honest coverage.

Same rules as the movies: a book is kept if its Wikipedia article has a
plot/synopsis section of >= 40 words, or a lead of >= 50 words.
Output: data/clean/books.jsonl (local only, never committed).
Run:  python scripts/clean_books.py
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from text_utils import (BOOK_GENRE_RULES, coarse_genres, sentences, tidy, words)

MIN_PLOT_WORDS, MIN_LEAD_WORDS = 40, 50
OUT = Path("data/clean/books.jsonl")


def book_title(wiki_title):
    """'The White Tiger (novel)' -> 'The White Tiger'."""
    return re.sub(r"\s*\([^()]*\b(novel|book|memoir|novella|collection|series|poetry|"
                  r"poem|play|comics?|manga|autobiography)\)$", "", wiki_title).strip()


def book_genre(label):
    """'science fiction novel' -> 'science fiction'."""
    label = label.lower().strip()
    return re.sub(r"\s+(novel|novels|book|books|literature)$", "", label) or label


def clean_all(wd, wp):
    by_qid = {b["qid"]: b for b in wd["books"]}
    rows, reason, seen = [], {}, set()
    for qid in wp["selected"]:
        b, t = by_qid[qid], wp["texts"].get(qid)
        if qid in wp["failed"]:
            reason[qid] = "fetch_failed"
            continue
        if t is None:
            reason[qid] = "missing_article"
            continue
        if t["title"] in seen:
            reason[qid] = "duplicate_article"
            continue
        lead, plot = tidy(t["lead"]), tidy(t["plot"])
        if words(plot) < MIN_PLOT_WORDS:
            plot = ""
        if plot:
            reason[qid] = "plot"
        elif words(lead) >= MIN_LEAD_WORDS:
            reason[qid] = "lead_only"
        else:
            reason[qid] = "too_short"
            continue
        seen.add(t["title"])
        first = (sentences(lead) or [""])[0]
        genres = coarse_genres(b["genres_raw"], BOOK_GENRE_RULES)
        if not genres:  # "... is a 2008 dystopian novel by ..." -> sci-fi
            genres = coarse_genres([first.split(" by ")[0]], BOOK_GENRE_RULES)
        countries = [c for c in b["author_countries"]][:3]
        rows.append({
            "id": qid, "title": book_title(t["title"]), "wiki_title": t["title"],
            "wiki_url": "https://en.wikipedia.org/wiki/" + t["title"].replace(" ", "_"),
            "year": b["year"], "language": "in" if b["indian"] else "intl",
            "all_languages": b["languages"], "indian": b["indian"],
            "genres": genres,
            "genres_fine": sorted({book_genre(g) for g in b["genres_raw"]}) or genres,
            "first_sentence": first, "lead": lead, "plot": plot,
            "text_status": reason[qid], "sitelinks": b["sitelinks"],
            "creator": ", ".join(b["authors"][:3]),
            "subtitle": ("Author from " + ", ".join(countries)) if countries else "",
            "link_label": "Read on Wikipedia",
        })
    return rows, reason


def report(wd, wp, rows, reason):
    print(f"Wikidata fetched {wd['fetched']}; failed queries: {wd['failed_queries'] or 'none'}")
    print(f"Candidates {len(wd['books'])}, selected {len(wp['selected'])}, kept {len(rows)}")
    print("Outcomes:", dict(Counter(reason.values())))
    for name, group in (("Indian", [r for r in rows if r["indian"]]),
                        ("International", [r for r in rows if not r["indian"]])):
        plots = [words(r["plot"]) for r in group if r["plot"]]
        print(f"\n{name}: {len(group)} kept, with plot/synopsis {len(plots)}"
              + (f", plot median {statistics.median(plots):.0f} words" if plots else "")
              + f", no genre {sum(not r['genres'] for r in group)}")
        print("  author countries:", ", ".join(f"{c} {n}" for c, n in Counter(
            c for r in group for c in r["subtitle"].replace("Author from ", "").split(", ")
            if c).most_common(10)))
    print("\nPer year (Indian / international):")
    by = Counter((r["year"], r["indian"]) for r in rows)
    for y in sorted({r["year"] for r in rows}):
        print(f"  {y}: {by[(y, True)]:4} / {by[(y, False)]:4}")
    print("\nGenres:", ", ".join(f"{g} {n}" for g, n in Counter(
        g for r in rows for g in r["genres"]).most_common()))
    print("\nExamples:")
    for r in rows[:3] + rows[-3:]:
        print(f"- {r['title']} ({r['year']}) by {r['creator']} [{r['language']}] "
              f"{r['genres']} :: {r['first_sentence'][:140]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    wd = json.loads(Path("data/interim/wikidata_books.json").read_text(encoding="utf-8"))
    wp = json.loads(Path("data/interim/wikipedia_books.json").read_text(encoding="utf-8"))
    rows, reason = clean_all(wd, wp)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(wd, wp, rows, reason)
    print(f"\nSaved {len(rows)} books to {OUT}")
