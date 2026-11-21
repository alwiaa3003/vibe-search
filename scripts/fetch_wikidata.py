"""Phase 1, step 1: list candidate films from Wikidata.

One SPARQL query per (language, year), so every query stays small and fast.
A film counts for the year of its EARLIEST publication date (P577); films whose
earliest date is before 2010 are dropped.
English films are far more numerous (~1,500 a year with an English article), and
fetching all their details times out, so for English the query keeps only films
with >= 15 Wikipedia language editions ("sitelinks", a rough fame signal).
The query planner is told to follow our written order (start from the language),
otherwise it picks a slow plan and the query times out.
Output: data/interim/wikidata_films.json (local only, never committed).
Run:  python scripts/fetch_wikidata.py
"""
import json
import time
from datetime import date
from pathlib import Path
from urllib.parse import unquote

import requests

from wiki_client import sparql

# Checked with scripts/lookup_ids.py (ISO 639-1 code -> Wikidata item).
LANGS = {"hi": "Q1568", "ta": "Q5885", "te": "Q8097", "ml": "Q36236",
         "kn": "Q33673", "bn": "Q9610", "mr": "Q1571", "pa": "Q58635",
         "en": "Q1860"}
YEARS = range(2010, 2027)
MIN_SITELINKS = {"en": 15}  # other languages: no minimum
OUT = Path("data/interim/wikidata_films.json")

QUERY = """
SELECT ?film ?article ?sitelinks (MIN(YEAR(?d2)) AS ?year)
  (GROUP_CONCAT(DISTINCT ?genreLabel; separator="|") AS ?genres)
  (GROUP_CONCAT(DISTINCT ?code; separator="|") AS ?langs) WHERE {
  { SELECT DISTINCT ?film WHERE {
      hint:Query hint:optimizer "None" .
      ?film wdt:P364 wd:%(lang)s .
      ?film wdt:P31 wd:Q11424 .
      ?film wdt:P577 ?d .
      FILTER(?d >= "%(y)d-01-01T00:00:00Z"^^xsd:dateTime &&
             ?d <  "%(y1)d-01-01T00:00:00Z"^^xsd:dateTime)
      ?film wikibase:sitelinks ?sl . FILTER(?sl >= %(min_sl)d)
  } }
  ?article schema:about ?film; schema:isPartOf <https://en.wikipedia.org/> .
  ?film wikibase:sitelinks ?sitelinks; wdt:P577 ?d2 .
  OPTIONAL { ?film wdt:P136 ?g . ?g rdfs:label ?genreLabel .
             FILTER(LANG(?genreLabel) = "en") }
  OPTIONAL { ?film wdt:P364 ?l . ?l wdt:P218 ?code . }
} GROUP BY ?film ?article ?sitelinks
"""

films, failed = {}, []
started = time.time()
for code, qid in LANGS.items():
    kept = 0
    for y in YEARS:
        try:
            rows = sparql(QUERY % {"lang": qid, "y": y, "y1": y + 1,
                                   "min_sl": MIN_SITELINKS.get(code, 0)})
        except (requests.RequestException, RuntimeError) as e:
            failed.append((code, y, type(e).__name__))
            print(f"  FAILED {code} {y}: {type(e).__name__}")
            # Don't keep hitting the server if something is clearly wrong.
            if len(failed) >= 3 and [f[:1] for f in failed[-3:]] == [(code,)] * 3:
                raise SystemExit("3 failures in a row: stopping. Rerun later "
                                 "(finished queries are cached).")
            continue
        for r in rows:
            year = int(r["year"]["value"])
            if year != y:  # earliest date is in another year: counted there
                continue
            fid = r["film"]["value"].rsplit("/", 1)[-1]
            if fid in films:  # bilingual film already found via another language
                continue
            article = r["article"]["value"]
            films[fid] = {
                "qid": fid,
                "wiki_title": unquote(article.rsplit("/wiki/", 1)[-1]).replace("_", " "),
                "wiki_url": article,
                "year": year,
                "found_via": code,
                "languages": sorted(filter(None, r["langs"]["value"].split("|"))),
                "genres_raw": sorted(filter(None, r["genres"]["value"].split("|"))),
                "sitelinks": int(r["sitelinks"]["value"]),
            }
            kept += 1
    print(f"{code}: {kept} films  (running total {len(films)}, "
          f"{time.time() - started:.0f}s)", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"fetched": date.today().isoformat(),
                           "failed_queries": failed,
                           "min_sitelinks": MIN_SITELINKS,
                           "films": list(films.values())},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved {len(films)} films to {OUT}. Failed queries: {failed or 'none'}")
