"""Phase 2: embed every film with all-MiniLM-L6-v2 and save the vectors.

Two ways to represent a film:
  (a) short  - ONE vector per film: title + genres + first lead sentence + as
               many whole plot sentences as fit in the model's 256-token window.
  (b) chunks - the plot (or the lead, if there is no plot) split into chunks of
               whole sentences, ~128 tokens each, ONE vector per chunk. A film
               is later scored by its best-matching chunk.
Token counts use the model's own tokenizer, so nothing is silently truncated
(except single sentences longer than the window, which are counted below).
Output: data/index/ (local only, never committed). Prints time and memory.
Run:  python scripts/build_index.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from text_utils import build_short_text, chunk_prefix, plot_chunks, sentences

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENS = 128
FILMS = Path("data/clean/films.jsonl")
OUT = Path("data/index")


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def film_texts(film, size, window):
    """Return (short_text, chunk_texts) for one film."""
    body = sentences(film["plot"]) if film["plot"] else sentences(film["lead"])[1:]
    if not body:  # lead-only film with a one-sentence lead
        body = sentences(film["lead"])
    short = build_short_text(film["title"], film["genres_fine"],
                             film["first_sentence"], body, window, size)
    chunks = plot_chunks(chunk_prefix(film), body, CHUNK_TOKENS, size)
    return short, chunks


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0, mem0 = time.time(), rss_mb()
    from sentence_transformers import SentenceTransformer  # slow import
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    tok = model.tokenizer
    window = model.max_seq_length - 2  # [CLS] and [SEP] take 2 of the 256
    print(f"Model loaded in {time.time() - t0:.1f}s, max_seq_length="
          f"{model.max_seq_length}, RSS {rss_mb():.0f} MB (start {mem0:.0f} MB)")

    def size(text):
        return len(tok.tokenize(text))

    films = [json.loads(line) for line in FILMS.open(encoding="utf-8")]
    t1 = time.time()
    shorts, chunks, owner = [], [], []
    for i, f in enumerate(films):
        s, cs = film_texts(f, size, window)
        shorts.append(s)
        chunks += cs
        owner += [i] * len(cs)
    short_tokens = [size(s) for s in shorts]
    chunk_tokens = [size(c) for c in chunks]
    print(f"Built texts for {len(films)} films in {time.time() - t1:.1f}s")
    print(f"  short: tokens median {int(np.median(short_tokens))}, max "
          f"{max(short_tokens)}, over window {sum(t > window for t in short_tokens)}")
    print(f"  chunks: {len(chunks)} total ({len(chunks) / len(films):.1f} per film), "
          f"tokens median {int(np.median(chunk_tokens))}, over window "
          f"{sum(t > window for t in chunk_tokens)}, over {CHUNK_TOKENS} "
          f"{sum(t > CHUNK_TOKENS for t in chunk_tokens)}")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, texts in (("short", shorts), ("chunks", chunks)):
        t2 = time.time()
        vecs = model.encode(texts, batch_size=64, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
        vecs = vecs.astype(np.float32)
        np.save(OUT / f"{name}.npy", vecs)
        print(f"Encoded {name}: {len(texts)} texts in {time.time() - t2:.1f}s "
              f"({len(texts) / (time.time() - t2):.0f}/s), vectors "
              f"{vecs.nbytes / 2**20:.1f} MB, RSS now {rss_mb():.0f} MB")
    np.save(OUT / "chunk_owner.npy", np.array(owner, dtype=np.int32))
    (OUT / "short_texts.json").write_text(json.dumps(shorts, ensure_ascii=False),
                                         encoding="utf-8")
    (OUT / "chunk_texts.json").write_text(json.dumps(chunks, ensure_ascii=False),
                                         encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps({
        "model": MODEL_NAME, "films": len(films), "chunks": len(chunks),
        "chunk_tokens": CHUNK_TOKENS, "window": window,
        "ids": [f["id"] for f in films]}), encoding="utf-8")
    print(f"Total {time.time() - t0:.1f}s, peak-ish RSS {rss_mb():.0f} MB. "
          f"Saved to {OUT}/")
    print("\nExample short text (film 0):\n" + shorts[0])


if __name__ == "__main__":
    main()
