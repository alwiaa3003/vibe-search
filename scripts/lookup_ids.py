"""Look up the Wikidata IDs we rely on, so they are checked, not guessed.

Languages are found by their ISO 639-1 code (Wikidata property P218).
Run:  python scripts/lookup_ids.py
"""
from wiki_client import sparql

CODES = ["en", "hi", "ta", "te", "ml", "kn", "bn", "mr", "pa"]

# 1) Language items by ISO code
values = " ".join(f'"{c}"' for c in CODES)
rows = sparql(f"""
SELECT ?code ?lang ?langLabel WHERE {{
  VALUES ?code {{ {values} }}
  ?lang wdt:P218 ?code .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} ORDER BY ?code""")
print("Languages (ISO 639-1 code -> Wikidata item):")
for r in rows:
    qid = r["lang"]["value"].rsplit("/", 1)[-1]
    print(f"  {r['code']['value']:3} {qid:10} {r['langLabel']['value']}")

# 2) Labels of the properties/classes the queries use, as a sanity check
ids = {"P31": "instance of", "P364": "original language", "P577": "publication date",
       "P136": "genre", "P218": "ISO 639-1 code", "Q11424": "film"}
rows = sparql(f"""
SELECT ?item ?label WHERE {{
  VALUES ?item {{ {" ".join("wd:" + i for i in ids)} }}
  ?item rdfs:label ?label . FILTER(LANG(?label) = "en")
}}""")
print("\nOther IDs used (Wikidata's own English label):")
for r in rows:
    print(f"  {r['item']['value'].rsplit('/', 1)[-1]:7} {r['label']['value']}")
