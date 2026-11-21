"""Polite, cached client for the MusicBrainz web service (used for songs).

MusicBrainz rules (https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting):
- a meaningful User-Agent with contact information;
- on average at most 1 request per second per client (we wait 1.1 s);
- HTTP 503 means "slow down": wait and retry.
Every successful JSON reply is cached in data/raw/musicbrainz/, so reruns are
instant and nothing is downloaded twice.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

from config import CACHE_DIR, CONTACT

API = "https://musicbrainz.org/ws/2/"
GAP = 1.1  # seconds between requests

if not CONTACT.strip():
    sys.exit("Please set CONTACT in scripts/config.py first.")

_session = requests.Session()
_session.headers.update({
    "User-Agent": f"VibeSearchPortfolioBot/0.1 ({CONTACT.strip()}) "
                  f"python-requests/{requests.__version__}",
    "Accept": "application/json", "Accept-Encoding": "gzip"})
_last = 0.0


def _cache_path(endpoint, params):
    key = endpoint + "?" + json.dumps(params, sort_keys=True)
    return Path(CACHE_DIR) / "musicbrainz" / (hashlib.sha1(key.encode()).hexdigest() + ".json")


def mb_get(endpoint, params, max_tries=6):
    """GET API + endpoint (e.g. 'release-group') with params; return JSON."""
    global _last
    params = {**params, "fmt": "json"}
    path = _cache_path(endpoint, params)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(max_tries):
        wait = GAP - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = _session.get(API + endpoint, params=params, timeout=60)
        except requests.RequestException as e:
            _last = time.time()
            print(f"  network error ({type(e).__name__}), retrying")
            time.sleep(5 * 2 ** attempt)
            continue
        _last = time.time()
        if resp.status_code in (429, 503):
            ra = resp.headers.get("Retry-After")
            backoff = float(ra) if ra and ra.isdigit() else 5 * 2 ** attempt
            print(f"  MusicBrainz busy (HTTP {resp.status_code}), waiting {backoff:.0f}s")
            time.sleep(backoff)
            continue
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data
    raise RuntimeError(f"MusicBrainz: gave up after {max_tries} tries: {endpoint} {params}")
