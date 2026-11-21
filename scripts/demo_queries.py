"""Run a few example queries through several methods and print top results,
query times and memory. Used for the Phase 2-4 reports.

Run:  python scripts/demo_queries.py                          (default queries, all methods)
      python scripts/demo_queries.py --methods emb_short emb_chunks --k 5
      python scripts/demo_queries.py --queries "time loop" "village ghost story"
"""
import argparse
import os
import sys
import time

import psutil

from engine import METHODS, load_engine

DEFAULT_QUERIES = [
    "cosy mystery for a rainy evening",
    "feel-good film that will cheer me up after a bad day",
    "big family wedding chaos and drama",
    "ghost story set in an old mansion in a village",
    "time loop",
    "nothing too violent but still exciting",
]


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    p.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()

    t, m0 = time.time(), rss_mb()
    eng = load_engine()
    print(f"Engine loaded in {time.time() - t:.1f}s, RSS {rss_mb():.0f} MB (start {m0:.0f} MB)")
    eng.search("warm up", method="hybrid", k=1)  # first call is slower (lazy init)
    if "hybrid_rerank" in a.methods:
        t = time.time()
        eng.enable_rerank()
        print(f"Re-ranker loaded in {time.time() - t:.1f}s, RSS {rss_mb():.0f} MB")

    times = {m: [] for m in a.methods}
    for q in a.queries:
        print(f"\n=== {q}")
        for m in a.methods:
            t = time.time()
            res = eng.search(q, method=m, k=a.k)
            times[m].append(time.time() - t)
            titles = "; ".join(f"{eng.films[i]['title']} ({eng.films[i]['year']}, "
                               f"{eng.films[i]['language']})" for i, _ in res)
            print(f"  {m:13} {titles}")
    print("\nMedian query time per method:")
    for m, ts in times.items():
        ts = sorted(ts)
        print(f"  {m:13} {ts[len(ts) // 2] * 1000:7.0f} ms   (max {ts[-1] * 1000:.0f} ms)")
    print(f"Final RSS {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
