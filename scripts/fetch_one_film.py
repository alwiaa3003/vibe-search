"""Phase 0 check: fetch ONE film from Wikidata plus its English Wikipedia text.

Prints the film's metadata, its Wikipedia lead, the start of its Plot section,
and the text we would embed for it.
Run:  python scripts/fetch_one_film.py            (a Hindi film)
      python scripts/fetch_one_film.py Q5885      (another language item)
"""
import re
import sys
from urllib.parse import unquote

from wiki_client import sparql, wikipedia_plain_text

LANG_QID = sys.argv[1] if len(sys.argv) > 1 else "Q1568"  # Hindi (see lookup_ids.py)
PLOT_HEADINGS = {"plot", "synopsis", "plot summary", "premise", "story", "summary"}
LEAD_WORDS = 60      # keep the lead short: it is mostly cast/crew/box office
TOTAL_WORDS = 180    # ~180 words fits inside MiniLM's 256 tokens (checked in Phase 2)

# One film: instance of film, given original language, released 2010+,
# with an English Wikipedia article. Genres are joined into one string.
query = f"""
SELECT ?film ?article (MIN(YEAR(?date)) AS ?year)
       (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") AS ?genres) WHERE {{
  {{ SELECT ?film ?article WHERE {{
      ?film wdt:P31 wd:Q11424; wdt:P364 wd:{LANG_QID}; wdt:P577 ?d .
      FILTER(YEAR(?d) >= 2010)
      ?article schema:about ?film; schema:isPartOf <https://en.wikipedia.org/> .
  }} LIMIT 1 }}
  ?film wdt:P577 ?date .
  OPTIONAL {{ ?film wdt:P136 ?g . ?g rdfs:label ?genreLabel .
             FILTER(LANG(?genreLabel) = "en") }}
}} GROUP BY ?film ?article"""

row = sparql(query)[0]
qid = row["film"]["value"].rsplit("/", 1)[-1]
title = unquote(row["article"]["value"].rsplit("/wiki/", 1)[-1]).replace("_", " ")
year = row["year"]["value"]
genres = row["genres"]["value"]

title, text = wikipedia_plain_text(title)

# Split the plain text on level-2 headings like "== Plot ==".
parts = re.split(r"^==([^=].*?)==\s*$", text, flags=re.MULTILINE)
lead = parts[0].strip()
plot = ""
for heading, body in zip(parts[1::2], parts[2::2]):
    if heading.strip().lower() in PLOT_HEADINGS:
        plot = body.strip()
        break


def first_words(s, n, marker=" ..."):
    words = s.split()
    return " ".join(words[:n]) + (marker if len(words) > n else "")


# Text to embed: title + genres + short lead + as much plot as fits.
# (No "..." markers here: they would just be noise for the model.)
head = f"{title}. Genres: {genres or 'unknown'}. {first_words(lead, LEAD_WORDS, '')}"
room = TOTAL_WORDS - len(head.split())
embed_text = f"{head} {first_words(plot, max(room, 0), '')}".strip()

# Windows consoles default to an old encoding; print UTF-8 so é, – show properly.
sys.stdout.reconfigure(encoding="utf-8")

print(f"Wikidata item : {qid}")
print(f"Title         : {title}")
print(f"Year          : {year}")
print(f"Language item : {LANG_QID}")
print(f"Genres        : {genres or '(none on Wikidata)'}")
print(f"Lead          : {len(lead.split())} words, Plot: {len(plot.split())} words")
print(f"\n--- Lead (first 80 words) ---\n{first_words(lead, 80)}")
print(f"\n--- Plot (first 80 words) ---\n{first_words(plot, 80) or '(no Plot section)'}")
print(f"\n--- Text to embed ({len(embed_text.split())} words) ---\n{embed_text}")
