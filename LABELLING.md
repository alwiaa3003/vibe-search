# How to label (Phase 5 evaluation)

You judge whether each film fits each query. The scripts then score every
search method against your judgements. Nobody, including you, should know
which method found which film while labelling.

## 1. Choose the queries (about 10 minutes)

Open `eval/candidate_queries.csv` in VS Code or Excel. There are 40 candidates
of five kinds: mood, situation, vibe, indian (Indian context) and hard.

- Set `use` to `yes` for about 30 queries you want, and to `no` for the rest.
  Right now 38 are `yes`, so switch off about 8. Keep a mix of kinds and keep
  some `hard` ones, because failures are informative.
- You may reword a query or add your own. Give a new query a new `id` (for
  example `q41`) and don't reuse an old id for a different query.
- Decide the queries **before** you look at any results, and don't change them
  after labelling. Changing them afterwards would bias the test.

## 2. Build the blind sheet

```bash
cd /c/Users/alwia/VibeSearch
source .venv/Scripts/activate
python scripts/pool.py
```

This runs all 5 methods on every chosen query and keeps the top 10 from each.
It merges them so each film appears once per query, shuffles them, and writes
`eval/labelling_sheet.csv`. Expect roughly 20-35 rows per query, because the
methods overlap. That's about 600-1,000 rows for 30 queries, so plan on 2-4
hours in total. You can do it over several sittings.

The sheet does **not** show which method found a film or at what rank. Please
don't try to work it out, for example by running searches while you label.

## 3. Fill in the `label` column

Open `eval/labelling_sheet.csv` in Excel, LibreOffice or VS Code. For each row,
read the query, then the title, genres and `description`. Open `wiki_url` if
the description isn't enough. Then type one number:

| label | meaning | example for "cosy mystery for a rainy evening" |
|---|---|---|
| **2** | fits well: you'd happily suggest it for this query | a gentle whodunit in a small town |
| **1** | partly fits: right in one important way, wrong in another | a mystery, but a violent, gritty one |
| **0** | doesn't fit | a sports drama |

Rules, so the labels stay consistent:
- Judge the **query as written**, the way a friend would mean it. "Something to
  watch with my grandparents" means suitable and enjoyable for them, not just
  any film that has grandparents in it.
- Judge only from what you can see: the description, the genres and the
  Wikipedia page. Don't let your opinion of the film's quality change the label.
- If you know the film well and the description is misleading, label the film
  and say why in `comment`.
- If you really can't tell, use `1` and write `unsure` in `comment`.
- Every row should get a number. Blank rows are ignored, which shrinks the test.
- Don't edit the other columns, and don't sort or delete rows.
- Save as CSV, keeping UTF-8 if asked.

A good pace is to do one whole query at a time, so that your idea of a "2"
stays the same within that query.

## 4. Score the methods

```bash
python scripts/evaluate.py
```

This prints the main table and writes the full report to `eval/results.md`:
- the main table,
- results by query kind,
- results per query,
- a failure analysis: each method's worst queries, what it returned, and which
  good films it missed that other methods found.

## How the scores work
- **Recall@10**: of all the films you labelled 1 or 2 for a query, the share
  that the method put in its top 10.
- **MRR@10**: 1 divided by the rank of the first film you labelled 1 or 2
  (1.0 if it's first, 0.5 if it's second, 0 if none appear in the top 10).
- **nDCG@10**: rewards putting the 2s near the top. 1.0 means the best possible
  order of the films you labelled.

## Limits to write down (they also appear in the report)
- One labeller (you) and about 30 queries, so small differences are noise.
- Pooling: films that no method found never get labelled, so Recall is measured
  only against the films the methods found.
- The pool uses these same 5 methods. A new method added later may return
  unlabelled films. The report counts these as "unjudged", and they score as not
  relevant.
- The data is a snapshot of Wikidata and Wikipedia (fetch date is in
  `data/clean/fetch_info.json`).
- You're judging Wikipedia descriptions, which may differ from the films themselves.
- If you rebuild the data or the index after labelling, rerun `pool.py`. It keeps
  your existing labels and adds any new rows for you to label.
