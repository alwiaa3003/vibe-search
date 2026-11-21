"""Settings shared by the fetch scripts."""

# Wikimedia's User-Agent policy asks every script to say who runs it.
# Put ONE of these here (an email is not required):
#   - a website/repo URL, e.g. "https://github.com/yourname/vibe-search"
#   - a Wikipedia username, e.g. "User:YourWikiName"
#   - an email address you are happy to share with Wikimedia
CONTACT = "https://github.com/alwiaa3003"

# Seconds to wait between requests (Wikimedia asks for 1 request at a time and
# under 5 per second without login; 1 per second is well under that).
MIN_INTERVAL = 1.0

# Where raw API responses are cached (local only, never committed).
CACHE_DIR = "data/raw"
