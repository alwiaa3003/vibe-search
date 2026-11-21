"""Phase 5: score every method against YOUR labels and write eval/results.md.

Reads eval/labelling_sheet.csv (labels 0/1/2 filled in by a person).
For each labelled query, every method is run again (top 10) and scored with
Recall@10, MRR@10 and nDCG@10. Films a method returns that were never labelled
are counted as not relevant, and how often that happens is reported.
Also writes a failure analysis: worst queries per method, with what came back.
Run:  python scripts/evaluate.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank

SHEET = Path("eval/labelling_sheet.csv")
QUERIES = Path("eval/candidate_queries.csv")
OUT = Path("eval/results.md")
K = 10


def read_labels(path=SHEET):
    """Return ({query_id: {film_id: label}}, {query_id: query}, problems)."""
    labels, texts, problems = defaultdict(dict), {}, []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for n, r in enumerate(csv.DictReader(fh), start=2):
            texts[r["query_id"]] = r["query"]
            v = r["label"].strip()
            if v == "":
                continue
            if v not in ("0", "1", "2"):
                problems.append(f"line {n}: label {v!r} is not 0, 1 or 2")
                continue
            labels[r["query_id"]][r["film_id"]] = int(v)
    return dict(labels), texts, problems


def score_runs(runs, labels):
    """runs: {method: {qid: [film ids]}} -> per-method per-query metrics."""
    out = {}
    for m, by_q in runs.items():
        out[m] = {}
        for qid, ranked in by_q.items():
            lab = labels[qid]
            out[m][qid] = {
                "recall": recall_at_k(ranked, lab, K),
                "mrr": reciprocal_rank(ranked, lab, K),
                "ndcg": ndcg_at_k(ranked, lab, K),
                "unjudged": sum(1 for i in ranked[:K] if i not in lab),
            }
    return out


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def write_report(scores, runs, labels, texts, kinds, info, films_by_id):
    methods = list(scores)
    qids = sorted(labels)
    lines = ["# Evaluation results", "",
             f"Labelled queries: {len(qids)}. Cut-off k = {K}. Relevant = label >= 1 "
             "for Recall/MRR; nDCG uses graded gains (2^label - 1).", "",
             "| method | Recall@10 | MRR@10 | nDCG@10 | queries scored | unjudged in top 10 |",
             "|---|---|---|---|---|---|"]
    for m in methods:
        s = scores[m]
        lines.append(f"| {m} | {fmt(mean(v['recall'] for v in s.values()))} | "
                     f"{fmt(mean(v['mrr'] for v in s.values()))} | "
                     f"{fmt(mean(v['ndcg'] for v in s.values()))} | "
                     f"{sum(1 for v in s.values() if v['ndcg'] is not None)} | "
                     f"{sum(v['unjudged'] for v in s.values())} |")

    kinds_present = sorted({kinds.get(q, "?") for q in qids})
    lines += ["", "## nDCG@10 by query type", "",
              "| method | " + " | ".join(kinds_present) + " |",
              "|---|" + "---|" * len(kinds_present)]
    for m in methods:
        cells = [fmt(mean(scores[m][q]["ndcg"] for q in qids if kinds.get(q) == k))
                 for k in kinds_present]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-query nDCG@10", "",
              "| query | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for q in qids:
        lines.append(f"| {q}: {texts[q]} | "
                     + " | ".join(fmt(scores[m][q]["ndcg"]) for m in methods) + " |")

    lines += ["", "## Failure analysis", ""]
    for m in methods:
        worst = sorted((q for q in qids if scores[m][q]["ndcg"] is not None),
                       key=lambda q: scores[m][q]["ndcg"])[:3]
        lines.append(f"### {m}: 3 worst queries")
        for q in worst:
            top3 = runs[m][q][:3]
            shown = "; ".join(f"{films_by_id[i]['title']} (label "
                              f"{labels[q].get(i, 'unjudged')})" for i in top3)
            others = set().union(*(runs[o][q][:K] for o in methods if o != m))
            missed = [i for i, g in labels[q].items()
                      if g == 2 and i not in runs[m][q][:K] and i in others]
            lines.append(f"- **{q}: {texts[q]}** nDCG {fmt(scores[m][q]['ndcg'])}. "
                         f"Top 3: {shown}.")
            if missed:
                lines.append("  - Missed but found by another method (label 2): "
                             + ", ".join(films_by_id[i]["title"] for i in missed[:5]))
        lines.append("")

    lines += ["## How these labels were made, and their limits", "",
              f"- Pool: top {info.get('pool_depth')} of each of "
              f"{info.get('methods_pooled')} methods per query, merged and shuffled; "
              "the labeller did not see which method found which film.",
              "- One labeller, small query set: differences between methods smaller "
              "than a few points are probably noise.",
              "- Pooling bias: a relevant film that no method found is never labelled, "
              "so Recall is relative to the pool, not the whole collection.",
              f"- Data snapshot: Wikidata/Wikipedia fetched {info.get('data_date')}; "
              f"{info.get('films_in_index')} films. Results apply to this snapshot only.",
              "- Labels judge the Wikipedia description shown, not the film itself."]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if not SHEET.exists():
        sys.exit("No labelling sheet yet: run python scripts/pool.py first.")
    labels, texts, problems = read_labels()
    for p in problems:
        print("WARNING", p)
    labels = {q: lab for q, lab in labels.items() if lab}
    if not labels:
        sys.exit("No labels filled in yet. See LABELLING.md, fill the 'label' "
                 "column with 0/1/2, save, then rerun.")

    from engine import METHODS, load_engine
    eng = load_engine(with_rerank=True)
    runs = {m: {q: [eng.films[i]["id"] for i, _ in eng.search(texts[q], method=m, k=K)]
                for q in labels} for m in METHODS}
    scores = score_runs(runs, labels)
    with QUERIES.open(encoding="utf-8", newline="") as fh:
        kinds = {r["id"]: r["kind"] for r in csv.DictReader(fh)}
    info = json.loads(Path("eval/pool_info.json").read_text(encoding="utf-8"))
    info["data_date"] = json.loads(Path("data/clean/fetch_info.json").read_text(
        encoding="utf-8"))["wikidata_fetched"]
    films_by_id = {f["id"]: f for f in eng.films}
    write_report(scores, runs, labels, texts, kinds, info, films_by_id)
    print(OUT.read_text(encoding="utf-8").split("## nDCG@10 by query type")[0])
    print(f"Full report: {OUT}")


if __name__ == "__main__":
    main()
