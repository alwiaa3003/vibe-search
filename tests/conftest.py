"""Shared test setup: a tiny fake film collection and a fake 'encoder'.

The fake encoder turns text into word-count vectors over a small vocabulary,
so 'similar meaning' = 'shared vocabulary words'. That is enough to test the
ranking, filtering and fusion logic without downloading any model or data.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from engine import Engine  # noqa: E402
from text_utils import chunk_prefix, plot_chunks, sentences  # noqa: E402

VOCAB = ["ghost", "haunted", "mansion", "fear", "love", "wedding", "family",
         "cricket", "match", "team", "space", "robot", "detective", "murder",
         "rain", "village", "funny", "laugh"]


def fake_encode(texts):
    out = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for r, t in enumerate(texts):
        words = t.lower().replace(".", " ").replace(",", " ").split()
        for w in words:
            if w in VOCAB:
                out[r, VOCAB.index(w)] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return out / norms


def film(id, title, year, lang, genres, lead, plot):
    return {"id": id, "title": title, "wiki_title": title, "year": year,
            "language": lang, "indian": lang != "en", "genres": genres,
            "genres_fine": genres, "first_sentence": sentences(lead)[0],
            "lead": lead, "plot": plot, "wiki_url": f"https://en.wikipedia.org/wiki/{id}",
            "text_status": "plot" if plot else "lead_only", "all_languages": [lang]}


FILMS = [
    film("F1", "Bhoot Bungalow", 2015, "hi", ["horror"],
         "Bhoot Bungalow is a horror film.",
         "A family moves into a haunted mansion. A ghost spreads fear. The family escapes."),
    film("F2", "Shaadi Mubarak", 2018, "hi", ["comedy", "romance"],
         "Shaadi Mubarak is a comedy film.",
         "A big wedding brings love and funny chaos. Everyone will laugh at the family."),
    film("F3", "Last Over", 2012, "ta", ["drama", "sports"],
         "Last Over is a sports drama.",
         "A village cricket team plays the final match. The team wins in the rain."),
    film("F4", "Star Child", 2021, "en", ["sci-fi"],
         "Star Child is a science fiction film.",
         "A robot travels through space. The robot learns about love."),
    film("F5", "Rain Murders", 2019, "en", ["mystery", "thriller"],
         "Rain Murders is a mystery film.",
         "A detective solves a murder on a night of rain. The detective is afraid."),
]


def build_toy_engine():
    shorts = [f"{f['title']}. {f['first_sentence']} {f['plot']}" for f in FILMS]
    chunks, owner = [], []
    for i, f in enumerate(FILMS):
        for c in plot_chunks(chunk_prefix(f), sentences(f["plot"]), budget=12):
            chunks.append(c)
            owner.append(i)
    return Engine(FILMS, fake_encode(shorts), fake_encode(chunks),
                  np.array(owner, dtype=np.int32), fake_encode,
                  rerank_fn=lambda q, texts: [len(set(q.split()) & set(t.lower().split()))
                                             for t in texts],
                  short_texts=shorts, chunk_texts=chunks)


@pytest.fixture(scope="session")
def toy_engine():
    return build_toy_engine()
