"""Terminal search.

Examples (run from the project folder):
  python scripts/search_cli.py "cosy mystery for a rainy evening"
  python scripts/search_cli.py "feel-good family comedy" --method bm25 --k 5
  python scripts/search_cli.py "village revenge drama" --lang ta te --year 2015-2020
  python scripts/search_cli.py "space adventure" --genre sci-fi --why
  python scripts/search_cli.py --like Q1026875          (more like this film)
Methods: bm25, emb_short, emb_chunks, hybrid (default), hybrid_rerank.
"""
import argparse
import sys
import time

from engine import METHODS, load_engine


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Vibe search in the terminal")
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--method", default="hybrid", choices=METHODS)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--lang", nargs="*", help="language codes, e.g. hi ta en")
    p.add_argument("--genre", nargs="*", help="coarse genres, e.g. comedy horror")
    p.add_argument("--year", help="range like 2015-2020, or one year")
    p.add_argument("--like", help="Wikidata id of a film: 'more like this'")
    p.add_argument("--why", action="store_true", help="explain each match")
    a = p.parse_args()

    filters = {"languages": a.lang, "genres": a.genre}
    if a.year:
        lo, _, hi = a.year.partition("-")
        filters["year_min"], filters["year_max"] = int(lo), int(hi or lo)

    t = time.time()
    eng = load_engine(with_rerank=(a.method == "hybrid_rerank"))
    print(f"(loaded {len(eng.films)} films in {time.time() - t:.1f}s)")

    t = time.time()
    if a.like:
        results = eng.more_like_this(a.like, k=a.k, **filters)
        print(f'More like "{eng.films[eng.id_to_idx[a.like]]["title"]}":')
    elif a.query:
        results = eng.search(a.query, method=a.method, k=a.k, **filters)
    else:
        p.error("give a query or --like ID")
    took = time.time() - t
    whys = eng.why_many(a.query, [i for i, _ in results]) if a.why and a.query else None
    for rank, (i, score) in enumerate(results, 1):
        f = eng.films[i]
        print(f"{rank:2}. {f['title']} ({f['year']}, {f['language']}) "
              f"[{', '.join(f['genres']) or '-'}]  score={score:.3f}  {f['id']}")
        if whys:
            w = whys[rank - 1]
            print(f"    why: \"{w['best_sentence'][:160]}\"")
            if w["shared_words"] or w["matching_genres"]:
                print(f"         words: {', '.join(w['shared_words']) or '-'}; "
                      f"genres: {', '.join(w['matching_genres']) or '-'}")
    method = "more_like_this" if a.like else a.method
    print(f"({len(results)} results in {took * 1000:.0f} ms, method={method})")


if __name__ == "__main__":
    main()
