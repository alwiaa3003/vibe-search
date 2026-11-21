"""Phase 1, step 2: choose which films to keep, then fetch their Wikipedia text.

Selection rule (decided from the real Wikidata counts):
  - Indian-language films: all of them.
  - English films: per year, the ENGLISH_PER_YEAR films with the most Wikipedia
    language editions (Wikidata "sitelinks", a rough fame/notability signal).
Text is fetched as wikitext, 50 articles per request (Wikimedia asks for
batching), one request at a time, cached (reruns are instant).
Output: data/interim/wikipedia_texts.json (local only, never committed).
Run:  python scripts/fetch_wikipedia.py
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import requests

from text_utils import split_wikitext
from wiki_client import wikipedia_wikitext_batch

ENGLISH_PER_YEAR = 100
BATCH = 50  # the API's maximum number of titles per request (without login)
IN_FILE = Path("data/interim/wikidata_films.json")
OUT = Path("data/interim/wikipedia_texts.json")


def select(films):
    """Return the films we keep, following the rule in the docstring."""
    indian = [f for f in films if f["found_via"] != "en"]
    english_by_year = defaultdict(list)
    for f in films:
        if f["found_via"] == "en":
            english_by_year[f["year"]].append(f)
    english = []
    for year, group in sorted(english_by_year.items()):
        group.sort(key=lambda f: (-f["sitelinks"], f["qid"]))  # stable order
        english += group[:ENGLISH_PER_YEAR]
    return indian + english


if __name__ == "__main__":
    films = json.loads(IN_FILE.read_text(encoding="utf-8"))["films"]
    chosen = select(films)
    print(f"Selected {len(chosen)} of {len(films)} candidates")

    texts, failed = {}, []
    in_a_row = 0  # consecutive failed batches
    started = time.time()
    for i in range(0, len(chosen), BATCH):
        batch = chosen[i:i + BATCH]
        titles = list(dict.fromkeys(f["wiki_title"] for f in batch))  # no dupes
        try:
            result = wikipedia_wikitext_batch(titles)
        except (requests.RequestException, RuntimeError, KeyError) as e:
            failed += [f["qid"] for f in batch]
            print(f"  FAILED batch at {i} ({type(e).__name__})")
            in_a_row += 1
            if in_a_row >= 3:  # something is wrong: stop instead of hammering
                raise SystemExit("3 failed batches in a row: stopping "
                                 "(rerun resumes from the cache).")
            continue
        in_a_row = 0
        for f in batch:
            page = result.get(f["wiki_title"])
            if page is None:
                texts[f["qid"]] = None
                continue
            lead, plot = split_wikitext(page[1])
            texts[f["qid"]] = {"title": page[0], "lead": lead, "plot": plot}
        done = min(i + BATCH, len(chosen))
        if done % 1000 < BATCH or done == len(chosen):
            print(f"{done}/{len(chosen)} done ({time.time() - started:.0f}s)", flush=True)

    OUT.write_text(json.dumps({"selected": [f["qid"] for f in chosen],
                               "failed": failed, "texts": texts},
                              ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(texts)} texts to {OUT}. Failed: {len(failed)}")
