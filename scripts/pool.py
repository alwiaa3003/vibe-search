"""Phase 5: build the BLIND labelling sheet.

For every query marked use=yes in eval/candidate_queries.csv, take the top
POOL_DEPTH results of EVERY method, merge them (each film once), shuffle them,
and write eval/labelling_sheet.csv. The sheet does not say which method found
a film, or at what rank, so the labels can't favour a method.

Safe to rerun: labels you already filled in are kept (matched on query_id +
film_id), and the old sheet is backed up first.
Run:  python scripts/pool.py
"""
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

from engine import METHODS, load_engine
from text_utils import sentences, take_sentences

POOL_DEPTH = 10
QUERIES = Path("eval/candidate_queries.csv")
SHEET = Path("eval/labelling_sheet.csv")
FIELDS = ["query_id", "query", "film_id", "title", "year", "language", "genres",
          "description", "wiki_url", "label", "comment"]


def read_queries(path=QUERIES):
    with path.open(encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["use"].strip().lower() == "yes"]


def describe(film, max_words=90):
    """First lead sentence + start of the plot, whole sentences only."""
    first = film["first_sentence"]
    rest = take_sentences(sentences(film["plot"] or film["lead"])[0 if film["plot"] else 1:],
                          max_words - len(first.split()))
    return " ".join([first] + rest)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    queries = read_queries()
    eng = load_engine(with_rerank=True)
    old = {}
    if SHEET.exists():
        with SHEET.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                old[(r["query_id"], r["film_id"])] = (r["label"], r["comment"])
        backup = SHEET.with_name(f"labelling_sheet.backup-{time.strftime('%Y%m%d-%H%M%S')}.csv")
        shutil.copy(SHEET, backup)
        print(f"Backed up existing sheet to {backup}")

    rows, pool_sizes = [], {}
    for q in queries:
        pool = set()
        for m in METHODS:
            pool |= {i for i, _ in eng.search(q["query"], method=m, k=POOL_DEPTH)}
        pool = sorted(pool)
        random.Random(q["id"]).shuffle(pool)  # fixed per-query shuffle
        pool_sizes[q["id"]] = len(pool)
        for i in pool:
            f = eng.films[i]
            label, comment = old.get((q["id"], f["id"]), ("", ""))
            rows.append({"query_id": q["id"], "query": q["query"], "film_id": f["id"],
                         "title": f["title"], "year": f["year"], "language": f["language"],
                         "genres": ", ".join(f["genres_fine"]), "description": describe(f),
                         "wiki_url": f["wiki_url"], "label": label, "comment": comment})

    SHEET.parent.mkdir(exist_ok=True)
    with SHEET.open("w", encoding="utf-8-sig", newline="") as fh:  # BOM: opens in Excel
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    kept = sum(1 for r in rows if r["label"] != "")
    Path("eval/pool_info.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M"), "pool_depth": POOL_DEPTH,
        "methods_pooled": len(METHODS), "films_in_index": len(eng.films),
        "queries": len(queries), "rows": len(rows), "pool_size_per_query": pool_sizes,
    }, indent=1), encoding="utf-8")
    print(f"{len(queries)} queries, {len(rows)} rows to label "
          f"(avg {len(rows) / max(len(queries), 1):.1f} per query); "
          f"{kept} labels carried over. Wrote {SHEET}")


if __name__ == "__main__":
    main()
