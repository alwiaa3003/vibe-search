"""Polite, cached HTTP client for Wikidata SPARQL and the Wikipedia API.

- Sends a descriptive User-Agent with contact info (Wikimedia policy).
- One request at a time, at least MIN_INTERVAL seconds apart.
- Retries on 429 / 503 / maxlag, waiting as long as the server's Retry-After says.
- Caches every successful JSON response in data/raw/ so reruns don't re-download.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

from config import CACHE_DIR, CONTACT, MIN_INTERVAL

SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

if not CONTACT.strip():
    sys.exit("Please set CONTACT in scripts/config.py first "
             "(a URL, 'User:YourWikiName', or an email).")

USER_AGENT = (f"VibeSearchPortfolioBot/0.1 ({CONTACT.strip()}) "
              f"python-requests/{requests.__version__}")

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT,
                         "Accept-Encoding": "gzip"})  # gzip, as Wikimedia asks
_last_request = 0.0
_next_gap = MIN_INTERVAL  # grows to 5 s after a slow request (Robot policy)


def _cache_path(source, url, params):
    # Same URL + same params -> same file name.
    key = url + "?" + json.dumps(params, sort_keys=True)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return Path(CACHE_DIR) / source / f"{digest}.json"


def get_json(source, url, params, max_tries=5):
    """GET url with params and return parsed JSON, using the local cache."""
    global _last_request, _next_gap
    path = _cache_path(source, url, params)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    for attempt in range(max_tries):
        # Wait so requests are spaced out (1 s normally, 5 s after a slow one).
        wait = _next_gap - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        start = time.time()

        resp = _session.get(url, params=params, timeout=90)
        _last_request = time.time()
        # Robot policy: "if your request takes more than 1 second to serve,
        # please wait 5 seconds before making another request."
        _next_gap = 5.0 if _last_request - start > 1.0 else MIN_INTERVAL
        retry_after = resp.headers.get("Retry-After")
        backoff = float(retry_after) if retry_after and retry_after.isdigit() \
            else 5 * 2 ** attempt

        if resp.status_code in (429, 503):
            print(f"  server busy (HTTP {resp.status_code}), waiting {backoff:.0f}s")
            time.sleep(backoff)
            continue
        resp.raise_for_status()
        data = resp.json()

        # The Wikipedia API reports 'maxlag'/'ratelimited' inside a 200 reply.
        err = data.get("error", {}).get("code") if isinstance(data, dict) else None
        if err in ("maxlag", "ratelimited"):
            print(f"  Wikipedia said '{err}', waiting {backoff:.0f}s")
            time.sleep(backoff)
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    raise RuntimeError(f"Gave up after {max_tries} tries: {url}")


def sparql(query):
    """Run a Wikidata SPARQL query and return the list of result rows."""
    data = get_json("wikidata", SPARQL_URL, {"query": query, "format": "json"})
    return data["results"]["bindings"]


def wikipedia_wikitext_batch(titles):
    """Fetch the wikitext (article source) of up to 50 articles in ONE request.

    Returns {requested_title: (final_title, wikitext) or None if missing}.
    Follows redirects ("Foo (film)" -> "Foo") and title normalisation.
    """
    params = {
        "action": "query", "prop": "revisions", "rvprop": "content",
        "rvslots": "main", "titles": "|".join(titles), "redirects": 1,
        "format": "json", "formatversion": 2, "maxlag": 5,
    }
    q = get_json("wikipedia_batch", WIKIPEDIA_API, params)["query"]
    # Map each requested title to the final page title.
    hop = {x["from"]: x["to"] for x in q.get("normalized", []) + q.get("redirects", [])}
    pages = {p["title"]: p for p in q["pages"]}
    out = {}
    for t in titles:
        final = t
        for _ in range(3):  # normalised -> redirect -> ...
            final = hop.get(final, final)
        p = pages.get(final, {})
        revs = p.get("revisions")
        out[t] = (final, revs[0]["slots"]["main"]["content"]) if revs else None
    return out


def wikipedia_plain_text(title):
    """Return (resolved_title, plain text with '== Heading ==' markers) or None."""
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "exsectionformat": "wiki", "titles": title, "redirects": 1,
        "format": "json", "formatversion": 2,
        "maxlag": 5,  # back off if Wikipedia's servers are lagging
    }
    data = get_json("wikipedia", WIKIPEDIA_API, params)
    page = data["query"]["pages"][0]
    if page.get("missing") or "extract" not in page:
        return None
    return page["title"], page["extract"]
