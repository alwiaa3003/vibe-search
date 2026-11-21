"""Books, step 1: list books on Wikidata, then fetch their Wikipedia text.

Which books (user's choice, 2026-11-22): published 2000-2026, with an English
Wikipedia article, instance of novel / literary work / book / written work.
  - Indian: the author is a citizen of India, OR the book's language is an
    Indian language. All of them are kept.
  - International: NOVELS in English (form of creative work P7937 = novel,
    language of work P407 = English) whose author is not Indian. Per year, the
    BOOKS_PER_YEAR with the most Wikipedia language editions (sitelinks, a rough
    fame signal) are kept. (Asking for every English "book"/"literary work" per
    year timed out on Wikidata; starting from "form: novel" takes ~10 s a year
    and, measured for 2005, misses only 2 memoir/non-fiction items.)
Same polite, cached client as the movies (one request at a time).
Output: data/interim/wikidata_books.json + data/interim/wikipedia_books.json
Run:  python scripts/fetch_books.py
"""
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import unquote

import requests

from text_utils import BOOK_HEADINGS, split_wikitext
from wiki_client import sparql, wikipedia_wikitext_batch

TYPES = "wd:Q8261 wd:Q7725634 wd:Q571 wd:Q47461344"  # novel, literary work, book, written work
INDIA = "wd:Q668"
INDIAN_LANGS = "wd:Q1568 wd:Q5885 wd:Q8097 wd:Q36236 wd:Q33673 wd:Q9610 wd:Q1571 wd:Q58635 wd:Q1617"
#              Hindi     Tamil     Telugu    Malayalam  Kannada    Bengali   Marathi   Punjabi    Urdu
ENGLISH = "wd:Q1860"
YEARS = range(2000, 2027)
BOOKS_PER_YEAR = 60
MIN_SITELINKS_INTL = 3   # keeps the English query small; the top 60 have far more
BATCH = 50
OUT_WD = Path("data/interim/wikidata_books.json")
OUT_WP = Path("data/interim/wikipedia_books.json")

# Details for a set of books chosen by the inner query (%(inner)s).
DETAILS = """
SELECT ?book ?article ?sitelinks (MIN(YEAR(?d2)) AS ?year)
  (GROUP_CONCAT(DISTINCT ?genreLabel; separator="|") AS ?genres)
  (GROUP_CONCAT(DISTINCT ?authorLabel; separator="|") AS ?authors)
  (GROUP_CONCAT(DISTINCT ?citLabel; separator="|") AS ?countries)
  (GROUP_CONCAT(DISTINCT ?code; separator="|") AS ?langs) WHERE {
  { %(inner)s }
  ?article schema:about ?book; schema:isPartOf <https://en.wikipedia.org/> .
  ?book wikibase:sitelinks ?sitelinks; wdt:P577 ?d2 .
  OPTIONAL { ?book wdt:P136 ?g . ?g rdfs:label ?genreLabel . FILTER(LANG(?genreLabel) = "en") }
  OPTIONAL { ?book wdt:P50 ?a . ?a rdfs:label ?authorLabel . FILTER(LANG(?authorLabel) = "en")
             OPTIONAL { ?a wdt:P27 ?c . ?c rdfs:label ?citLabel . FILTER(LANG(?citLabel) = "en") } }
  OPTIONAL { ?book wdt:P407 ?l . ?l wdt:P218 ?code . }
} GROUP BY ?book ?article ?sitelinks
"""
YEAR_FILTER = ('?book wdt:P577 ?d . FILTER(?d >= "%(y)d-01-01T00:00:00Z"^^xsd:dateTime && '
               '?d < "%(y1)d-01-01T00:00:00Z"^^xsd:dateTime)')
INNER_INDIAN_AUTHOR = ("SELECT DISTINCT ?book WHERE { hint:Query hint:optimizer \"None\" . "
                       "?a wdt:P27 " + INDIA + " . ?book wdt:P50 ?a . "
                       "VALUES ?t { " + TYPES + " } ?book wdt:P31 ?t . " + YEAR_FILTER + " }")
INNER_INDIAN_LANG = ("SELECT DISTINCT ?book WHERE { hint:Query hint:optimizer \"None\" . "
                     "VALUES ?lang { " + INDIAN_LANGS + " } ?book wdt:P407 ?lang . "
                     "VALUES ?t { " + TYPES + " } ?book wdt:P31 ?t . " + YEAR_FILTER + " }")
INNER_ENGLISH = ("SELECT DISTINCT ?book WHERE { hint:Query hint:optimizer \"None\" . "
                 "?book wdt:P7937 wd:Q8261 . " + YEAR_FILTER +
                 " ?book wikibase:sitelinks ?sl . FILTER(?sl >= %(min_sl)d) "
                 "?book wdt:P407 " + ENGLISH + " . }")


def run(name, inner, y, books, failed):
    q = DETAILS % {"inner": inner % {"y": y, "y1": y + 1, "min_sl": MIN_SITELINKS_INTL}}
    try:
        rows = sparql(q)
    except (requests.RequestException, RuntimeError) as e:
        failed.append((name, y, type(e).__name__))
        print(f"  FAILED {name} {y}: {type(e).__name__}", flush=True)
        if len(failed) >= 3 and all(f[0] == name for f in failed[-3:]):
            raise SystemExit("3 failures in a row: stopping (finished queries are cached).")
        return
    for r in rows:
        if int(r["year"]["value"]) != y:  # earliest date is in another year
            continue
        qid = r["book"]["value"].rsplit("/", 1)[-1]
        split = lambda k: sorted(filter(None, r[k]["value"].split("|")))
        b = books.setdefault(qid, {
            "qid": qid, "year": y, "sitelinks": int(r["sitelinks"]["value"]),
            "wiki_title": unquote(r["article"]["value"].rsplit("/wiki/", 1)[-1]).replace("_", " "),
            "genres_raw": split("genres"), "authors": split("authors"),
            "author_countries": split("countries"), "languages": split("langs"),
            "found_via": set()})
        b["found_via"].add(name)


def select(books):
    indian = [b for b in books if b["indian"]]
    by_year = defaultdict(list)
    for b in books:
        if not b["indian"]:
            by_year[b["year"]].append(b)
    intl = []
    for y, group in sorted(by_year.items()):
        group.sort(key=lambda b: (-b["sitelinks"], b["qid"]))
        intl += group[:BOOKS_PER_YEAR]
    return indian + intl


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0, books, failed = time.time(), {}, []
    for name, inner in (("indian_author", INNER_INDIAN_AUTHOR),
                        ("indian_language", INNER_INDIAN_LANG),
                        ("english", INNER_ENGLISH)):
        for y in YEARS:
            run(name, inner, y, books, failed)
        print(f"{name}: running total {len(books)} books ({time.time() - t0:.0f}s)", flush=True)
    for b in books.values():
        b["indian"] = bool(b["found_via"] & {"indian_author", "indian_language"}) \
            or "India" in b["author_countries"]
        b["found_via"] = sorted(b["found_via"])
    books = list(books.values())
    OUT_WD.write_text(json.dumps({"fetched": date.today().isoformat(), "failed_queries": failed,
                                  "books_per_year": BOOKS_PER_YEAR, "books": books},
                                 ensure_ascii=False, indent=1), encoding="utf-8")

    chosen = select(books)
    print(f"Candidates {len(books)} (Indian {sum(b['indian'] for b in books)}); "
          f"selected {len(chosen)}", flush=True)
    texts, failed_wp = {}, []
    for i in range(0, len(chosen), BATCH):
        batch = chosen[i:i + BATCH]
        titles = list(dict.fromkeys(b["wiki_title"] for b in batch))
        try:
            result = wikipedia_wikitext_batch(titles)
        except (requests.RequestException, RuntimeError, KeyError) as e:
            failed_wp += [b["qid"] for b in batch]
            print(f"  FAILED batch at {i} ({type(e).__name__})")
            continue
        for b in batch:
            page = result.get(b["wiki_title"])
            if page is None:
                texts[b["qid"]] = None
                continue
            lead, plot = split_wikitext(page[1], BOOK_HEADINGS)
            texts[b["qid"]] = {"title": page[0], "lead": lead, "plot": plot}
    OUT_WP.write_text(json.dumps({"selected": [b["qid"] for b in chosen],
                                  "failed": failed_wp, "texts": texts},
                                 ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(texts)} book texts. Failed Wikidata queries: {failed or 'none'}; "
          f"failed Wikipedia: {len(failed_wp)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
