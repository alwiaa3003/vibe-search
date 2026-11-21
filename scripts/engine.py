"""The search engine: embeddings, BM25, hybrid (RRF), optional re-rank, filters,
"more like this" and "why it matched". One Engine per domain (movies, books,
songs; see domains.py). The word "film" below means "item" for books and songs.

The Engine class only needs films + vectors + an `encode` function, so tests
can build a tiny engine with fake vectors. `load_engine()` builds the real one
from data/clean and data/index.
"""
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from domains import DOMAINS, MAX_PER_GROUP
from text_utils import chunk_prefix, sentences

METHODS = ["bm25", "emb_short", "emb_chunks", "hybrid", "hybrid_rerank"]
RRF_K = 60  # standard constant for reciprocal rank fusion
STOPWORDS = set("""a an and are as at be but by for from has have he her his i in
into is it its me my of on or our she so that the their them they this to was we
were what when where which who will with you your about after again all also am
any because been before being both can could did do does film films movie movies
over some such than then there these those through up very would one like want
something someone""".split())


def tokenize(text):
    """Lowercase words, no stop-words. Used for BM25 and 'why it matched'."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def bm25_text(film):
    return " ".join([film["title"], " ".join(film["genres_fine"]),
                     film["lead"], film["plot"]])


class Engine:
    def __init__(self, films, short_vecs, chunk_vecs, chunk_owner, encode,
                 rerank_fn=None, short_texts=None, chunk_texts=None, max_per_group=None):
        self.films = films
        self.max_per_group = max_per_group  # songs: at most N songs per film
        self.short = short_vecs            # (n_films, d), unit length
        self.chunks = chunk_vecs           # (n_chunks, d), unit length
        self.owner = chunk_owner           # chunk -> film index
        self.encode = encode               # list[str] -> (n, d) unit vectors
        self.rerank_fn = rerank_fn         # (query, list[str]) -> scores, or None
        self.short_texts = short_texts or [bm25_text(f)[:1000] for f in films]
        self.chunk_texts = chunk_texts     # optional: speeds up "why"
        # chunks of film i are rows chunk_start[i]:chunk_end[i] (owner is sorted)
        n = np.arange(len(films))
        self.chunk_start = np.searchsorted(chunk_owner, n, side="left")
        self.chunk_end = np.searchsorted(chunk_owner, n, side="right")
        self.id_to_idx = {f["id"]: i for i, f in enumerate(films)}
        self.bm25 = BM25Okapi([tokenize(bm25_text(f)) for f in films])

    # ---------- scoring: one score per film, higher = better ----------
    def scores_short(self, qvec):
        return self.short @ qvec

    def scores_chunks(self, qvec):
        """A film's score is the score of its best-matching chunk."""
        best = np.full(len(self.films), -1.0, dtype=np.float32)
        np.maximum.at(best, self.owner, self.chunks @ qvec)
        return best

    def scores_bm25(self, query):
        return self.bm25.get_scores(tokenize(query))

    # ---------- filters ----------
    def mask(self, year_min=None, year_max=None, languages=None, genres=None,
             indian=None, exclude=None):
        """Boolean array: True for films that pass every given filter."""
        m = np.ones(len(self.films), dtype=bool)
        for i, f in enumerate(self.films):
            if year_min is not None and f["year"] < year_min:
                m[i] = False
            elif year_max is not None and f["year"] > year_max:
                m[i] = False
            elif languages and f["language"] not in languages:
                m[i] = False
            elif genres and not set(genres) & set(f["genres"]):
                m[i] = False
            elif indian is not None and f["indian"] != indian:
                m[i] = False
        if exclude is not None:
            m[exclude] = False
        return m

    @staticmethod
    def top(scores, mask, k):
        """Indices of the k best films that pass the mask, best first."""
        s = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
        if k <= 0:
            return []
        idx = np.argpartition(-s, k - 1)[:k]
        return [int(i) for i in idx[np.argsort(-s[idx], kind="stable")]]

    # ---------- main entry point ----------
    def search(self, query, method="hybrid", k=10, **filters):
        """Return a list of (film_index, score) for the query."""
        if not self.max_per_group:
            return self._search(query, method, k, **filters)
        # Songs from one film share the film's text, so they score alike:
        # fetch more, then keep at most max_per_group per film.
        return self.diversify(self._search(query, method, k * 6, **filters), k)

    def diversify(self, ranked, k):
        out, seen = [], {}
        for i, s in ranked:
            g = self.films[i].get("group_id") or self.films[i]["id"]
            if seen.get(g, 0) < self.max_per_group:
                seen[g] = seen.get(g, 0) + 1
                out.append((i, s))
                if len(out) == k:
                    break
        return out

    def _search(self, query, method, k, **filters):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
        m = self.mask(**filters)
        if method == "bm25":
            s = self.scores_bm25(query)
            return [(i, float(s[i])) for i in self.top(s, m, k)]
        qvec = self.encode([query])[0]
        if method == "emb_short":
            s = self.scores_short(qvec)
            return [(i, float(s[i])) for i in self.top(s, m, k)]
        if method == "emb_chunks":
            s = self.scores_chunks(qvec)
            return [(i, float(s[i])) for i in self.top(s, m, k)]
        # hybrid: reciprocal rank fusion of the three ranked lists (top 100 each)
        depth = 100
        lists = [self.top(self.scores_bm25(query), m, depth),
                 self.top(self.scores_short(qvec), m, depth),
                 self.top(self.scores_chunks(qvec), m, depth)]
        fused = {}
        for ranked in lists:
            for rank, i in enumerate(ranked, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
        if method == "hybrid":
            return ranked[:k]
        # hybrid_rerank: cross-encoder re-scores the top 30 hybrid results
        self.enable_rerank()  # loads the model on first use
        cand = [i for i, _ in ranked[:max(30, k)]]
        rs = self.rerank_fn(query, [self.short_texts[i] for i in cand])
        order = sorted(zip(cand, rs), key=lambda x: (-x[1], x[0]))
        return [(i, float(s)) for i, s in order[:k]]

    def more_like_this(self, film_id, k=10, **filters):
        """Films whose short vector is closest to this film's short vector."""
        i = self.id_to_idx[film_id]
        m = self.mask(exclude=i, **filters)
        group = self.films[i].get("group_id")
        if group:  # songs: suggest songs from OTHER films
            m &= np.array([f.get("group_id") != group for f in self.films])
        s = self.short @ self.short[i]
        if self.max_per_group:
            return self.diversify([(j, float(s[j])) for j in self.top(s, m, k * 6)], k)
        return [(j, float(s[j])) for j in self.top(s, m, k)]

    def why(self, query, film_index):
        return self.why_many(query, [film_index])[0]

    def why_many(self, query, film_indices, max_sents=40):
        """Explain matches: the film sentence closest in meaning to the query,
        query words that also appear in the film text, and matching genres.

        To stay fast, we first pick each film's best-matching chunk using the
        precomputed chunk vectors, then embed only that chunk's sentences
        (plus the first lead sentence). All of them go in ONE batch."""
        qwords = set(tokenize(query))
        qvec = self.encode([query])[0]
        per_film, all_sents = [], []
        for i in film_indices:
            f = self.films[i]
            sents = self._why_candidates(i, qvec)[:max_sents]
            per_film.append((len(all_sents), len(sents)))
            all_sents += sents
        if all_sents:
            svecs = self.encode(all_sents)
        out = []
        for i, (start, n) in zip(film_indices, per_film):
            f = self.films[i]
            best = ""
            if n:
                best = all_sents[start + int(np.argmax(svecs[start:start + n] @ qvec))]
            film_words = set(tokenize(bm25_text(f)))
            out.append({"best_sentence": best,
                        "shared_words": sorted(qwords & film_words),
                        "matching_genres": [g for g in f["genres_fine"]
                                            if qwords & set(tokenize(g))]})
        return out

    def _why_candidates(self, i, qvec):
        """Sentences worth explaining film i with."""
        f = self.films[i]
        lo, hi = self.chunk_start[i], self.chunk_end[i]
        if self.chunk_texts is None or lo == hi:  # fallback: all sentences
            return sentences(f["plot"]) + sentences(f["lead"])
        best = lo + int(np.argmax(self.chunks[lo:hi] @ qvec))
        body = self.chunk_texts[best][len(chunk_prefix(f)) + 2:]  # drop "Title (genres): "
        # The generic first lead sentence ("X is a 2019 film directed by...")
        # tends to win but explains nothing, so use it only as a fallback.
        return sentences(body) or [f["first_sentence"]]

    def enable_rerank(self):
        """Load the cross-encoder re-ranker (only when first needed)."""
        if self.rerank_fn is None:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(RERANK_MODEL, device="cpu")
            self.rerank_fn = lambda q, texts: ce.predict(
                [(q, t) for t in texts], show_progress_bar=False)

    def film_card(self, i, score=None):
        f = self.films[i]
        card = {"id": f["id"], "title": f["title"], "year": f["year"],
                "language": f["language"], "genres": f["genres"],
                "genres_fine": f["genres_fine"], "first_sentence": f["first_sentence"],
                "wiki_url": f["wiki_url"], "score": score}
        for extra in ("creator", "subtitle", "link_label", "extra_url", "extra_label"):
            if f.get(extra):
                card[extra] = f[extra]
        return card


_MODELS = {}


def get_encoder(name):
    """Load the embedding model once and share it between domains."""
    if name not in _MODELS:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name, device="cpu")
        _MODELS[name] = lambda texts: model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, batch_size=64)
    return _MODELS[name]


def available_domains(root="."):
    """Domains whose data and index have been built."""
    return [d for d, c in DOMAINS.items()
            if (Path(root) / c["index"] / "meta.json").exists()]


def load_engines(root="."):
    """{domain: Engine} for every built domain (movies first)."""
    return {d: load_engine(root, d) for d in available_domains(root)}


def load_engine(root=".", domain="movies", with_rerank=False):
    """Build the real engine for one domain from its clean data + index
    (run the fetch/clean scripts and build_index.py first)."""
    root, conf = Path(root), DOMAINS[domain]
    films = [json.loads(line) for line in
             (root / conf["items"]).open(encoding="utf-8")]
    idx = root / conf["index"]
    meta = json.loads((idx / "meta.json").read_text(encoding="utf-8"))
    if meta["ids"] != [f["id"] for f in films]:
        raise RuntimeError(f"{domain} index is out of date: rerun "
                           f"scripts/build_index.py {domain}")
    encode = get_encoder(meta["model"])

    short_texts = json.loads((idx / "short_texts.json").read_text(encoding="utf-8"))
    chunk_texts = json.loads((idx / "chunk_texts.json").read_text(encoding="utf-8"))
    eng = Engine(films, np.load(idx / "short.npy"), np.load(idx / "chunks.npy"),
                 np.load(idx / "chunk_owner.npy"), encode, None, short_texts, chunk_texts,
                 max_per_group=MAX_PER_GROUP if domain == "songs" else None)
    if with_rerank:
        eng.enable_rerank()
    return eng


RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
