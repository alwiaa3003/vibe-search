"""The three things Vibe search can search: movies, books and songs.

Every domain uses the SAME record format (one JSON object per line), so one
search engine works for all three:
  id, title, year, language (the filter group, see "groups"), indian, genres,
  genres_fine, first_sentence, lead, plot, wiki_url (the link on the card)
  optional: creator (author / artist), subtitle, link_label, group_id
  (songs: the film a song comes from; search results show at most
  MAX_PER_GROUP songs from the same film).
"""
DOMAINS = {
    "movies": {
        "items": "data/clean/films.jsonl", "index": "data/index",
        "group_label": "Film industry",
        "groups": {"hi": "Bollywood (Hindi)", "us": "Hollywood (American)"},
    },
    "books": {
        "items": "data/clean/books.jsonl", "index": "data/index_books",
        "group_label": "Origin",
        "groups": {"in": "Indian", "intl": "International"},
    },
    "songs": {
        "items": "data/clean/songs.jsonl", "index": "data/index_songs",
        "group_label": "Language",
        "groups": {"hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
                   "kn": "Kannada", "bn": "Bengali", "mr": "Marathi", "pa": "Punjabi",
                   "ur": "Urdu", "gu": "Gujarati", "en": "English", "other": "Other / unknown"},
    },
}
MAX_PER_GROUP = 2
# The user asked (2026-11-22) to leave out South Indian films and songs:
# anything in Tamil, Telugu, Malayalam or Kannada (also Hindi films that list
# one of these as a language, e.g. Hindi + Telugu releases).
SOUTH_INDIAN = {"ta", "te", "ml", "kn"}
