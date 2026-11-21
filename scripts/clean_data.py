"""Phase 1, step 3: clean the fetched data and report honest coverage stats.

Each kept film stores separate fields: title, genres, first lead sentence,
full lead, full plot, language, year, Wikipedia title (+ a few extras).
Text status per film:
  plot       - has a Plot/Synopsis section of >= 40 words     (kept)
  lead_only  - no usable plot, but the lead has >= 50 words    (kept, flagged)
Dropped (with the reason counted):
  fetch_failed / missing_article / duplicate_article / too_short
Output: data/clean/films.jsonl + data/clean/fetch_info.json (local only).
Run:  python scripts/clean_data.py
"""
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from text_utils import (coarse_genres, display_title, normalise_genre,
                        sentences, tidy, words)

MIN_PLOT_WORDS, MIN_LEAD_WORDS = 40, 50
INDIAN = {"hi", "ta", "te", "ml", "kn", "bn", "mr", "pa"}
LANG_ORDER = ["hi", "ta", "te", "ml", "kn", "bn", "mr", "pa", "en"]
OUT = Path("data/clean/films.jsonl")


def clean_all(wd, wp):
    """Return (kept_rows, reason_per_qid) from the Wikidata + Wikipedia files."""
    by_qid = {f["qid"]: f for f in wd["films"]}
    failed = set(wp["failed"])
    rows, reason, seen_titles = [], {}, set()
    for qid in wp["selected"]:
        f, t = by_qid[qid], wp["texts"].get(qid)
        if qid in failed:
            reason[qid] = "fetch_failed"
            continue
        if t is None:
            reason[qid] = "missing_article"
            continue
        if t["title"] in seen_titles:  # two Wikidata items -> same article
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
        seen_titles.add(t["title"])
        lead_sents = sentences(lead)
        first = lead_sents[0] if lead_sents else ""
        # Genres: Wikidata if it has any, else words in the first lead
        # sentence ("... is a 2016 Malayalam romantic drama film ...").
        genres, genre_source = coarse_genres(f["genres_raw"]), "wikidata"
        if not genres:
            genres = coarse_genres([first.split(" film")[0]]) if " film" in first else []
            genre_source = "first_sentence" if genres else "none"
        rows.append({
            "id": qid,
            "title": display_title(t["title"]),
            "wiki_title": t["title"],
            "wiki_url": "https://en.wikipedia.org/wiki/" + t["title"].replace(" ", "_"),
            "year": f["year"],
            "language": f["found_via"],
            "all_languages": f["languages"],
            "indian": f["found_via"] in INDIAN,
            "genres": genres,
            "genre_source": genre_source,
            "genres_fine": sorted({normalise_genre(g) for g in f["genres_raw"]}) or genres,
            "first_sentence": first,
            "lead": lead,
            "plot": plot,
            "text_status": reason[qid],
            "sitelinks": f["sitelinks"],
        })
    return rows, reason


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def length_stats(name, xs):
    xs = sorted(xs)
    if not xs:
        print(f"  {name:24} (none)")
        return
    q = statistics.quantiles(xs, n=10) if len(xs) > 1 else [xs[0]] * 9
    print(f"  {name:24} median {statistics.median(xs):5.0f}  p10 {q[0]:5.0f}  "
          f"p90 {q[8]:5.0f}  max {xs[-1]:5}")


def report(wd, wp, rows, reason):
    lang_of = {f["qid"]: f["found_via"] for f in wd["films"]}
    cand = Counter(f["found_via"] for f in wd["films"])
    print(f"Wikidata fetched {wd['fetched']}; failed queries: "
          f"{wd['failed_queries'] or 'none'}; English min sitelinks: "
          f"{wd.get('min_sitelinks')}")
    print(f"Candidates {len(wd['films'])}, selected {len(wp['selected'])}, "
          f"kept {len(rows)}\n")

    cols = ["plot", "lead_only", "too_short", "missing_article",
            "duplicate_article", "fetch_failed"]
    print("Per language (selected films by outcome):")
    print("  lang  cand   sel " + " ".join(f"{c[:9]:>9}" for c in cols) + "   KEPT usable")
    for code in LANG_ORDER:
        c = Counter(reason[q] for q in wp["selected"] if lang_of[q] == code)
        sel, kept = sum(c.values()), c["plot"] + c["lead_only"]
        print(f"  {code:4} {cand[code]:5} {sel:5} "
              + " ".join(f"{c[x]:9}" for x in cols) + f" {kept:6} {pct(kept, sel):>6}")
    total = Counter(reason.values())
    print("  ALL  " + f"{len(wd['films']):5} {len(wp['selected']):5} "
          + " ".join(f"{total[x]:9}" for x in cols) + f" {len(rows):6}")

    ind = [r for r in rows if r["indian"]]
    eng = [r for r in rows if not r["indian"]]
    print(f"\nKept: Indian {len(ind)} ({pct(len(ind), len(rows))}), "
          f"English {len(eng)} ({pct(len(eng), len(rows))})")
    for name, group in (("Indian", ind), ("English", eng)):
        with_plot = [words(r["plot"]) for r in group if r["plot"]]
        no_genre = sum(1 for r in group if not r["genres"])
        print(f"\n{name}: has plot {len(with_plot)}/{len(group)} "
              f"({pct(len(with_plot), len(group))}), no coarse genre {no_genre} "
              f"({pct(no_genre, len(group))})  -- lengths in words:")
        print("  genre source: " + ", ".join(
            f"{k} {v}" for k, v in Counter(r["genre_source"] for r in group).most_common()))
        length_stats("lead", [words(r["lead"]) for r in group])
        length_stats("first lead sentence", [words(r["first_sentence"]) for r in group])
        length_stats("plot (when present)", with_plot)

    print("\nKept films per year and language:")
    table = defaultdict(Counter)
    for r in rows:
        table[r["year"]][r["language"]] += 1
    print("  year " + " ".join(f"{c:>4}" for c in LANG_ORDER) + "  total")
    for y in sorted(table):
        print(f"  {y} " + " ".join(f"{table[y][c]:4}" for c in LANG_ORDER)
              + f"  {sum(table[y].values()):5}")

    print("\nCoarse genres (kept films):", ", ".join(
        f"{g} {n}" for g, n in Counter(g for r in rows for g in r["genres"]).most_common()))

    print("\nSample rows (random, seed 0):")
    for r in random.Random(0).sample(rows, 6):
        print(f"- [{r['language']}] {r['title']} ({r['year']}) genres={r['genres']} "
              f"status={r['text_status']}\n    first sentence: {r['first_sentence'][:200]}"
              f"\n    plot start: {r['plot'][:200] or '(none)'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    wd = json.loads(Path("data/interim/wikidata_films.json").read_text(encoding="utf-8"))
    wp = json.loads(Path("data/interim/wikipedia_texts.json").read_text(encoding="utf-8"))
    rows, reason = clean_all(wd, wp)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path("data/clean/fetch_info.json").write_text(json.dumps({
        "wikidata_fetched": wd["fetched"],
        "failed_wikidata_queries": wd["failed_queries"],
        "failed_wikipedia": len(wp["failed"]),
        "films": len(rows),
        "drop_reasons": dict(Counter(reason.values())),
        "licence": "Wikidata CC0; Wikipedia text CC BY-SA 4.0 (Wikipedia contributors)",
    }, indent=1), encoding="utf-8")
    report(wd, wp, rows, reason)
    print(f"\nSaved {len(rows)} films to {OUT}")
