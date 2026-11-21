"""Helpers for turning Wikipedia plain text into lead / plot / text-to-embed."""
import re

import mwparserfromhell

DROP_TAGS = {"ref", "references", "gallery", "table", "timeline", "math", "score"}
PLOT_HEADINGS = {"plot", "synopsis", "plot summary", "premise", "story",
                 "summary", "storyline", "plot synopsis"}
# Books also use these for the section that says what the book is about.
BOOK_HEADINGS = PLOT_HEADINGS | {"plot overview", "content", "contents", "overview",
                                 "synopsis and themes", "plot and themes", "outline"}

# Coarse genres for filtering: if a Wikidata genre label contains the key,
# the film gets that coarse genre. "romantic comedy" -> romance + comedy.
GENRE_RULES = [
    ("action", "action"), ("masala", "action"), ("superhero", "action"),
    ("martial arts", "action"), ("comedy", "comedy"), ("drama", "drama"),
    ("romance", "romance"), ("romantic", "romance"), ("thriller", "thriller"),
    ("horror", "horror"), ("crime", "crime"), ("gangster", "crime"),
    ("mystery", "mystery"), ("detective", "mystery"),
    ("science fiction", "sci-fi"), ("fantasy", "fantasy"),
    ("animat", "animation"), ("documentary", "documentary"),
    ("family", "family"), ("children", "family"), ("musical", "musical"),
    ("war film", "war"), ("historical", "history"), ("period", "history"),
    ("biograph", "biography"), ("sport", "sports"), ("adventure", "adventure"),
    ("western", "western"), ("satire", "comedy"), ("parody", "comedy"),
    ("spoof", "comedy"), ("suspense", "thriller"), ("spy", "thriller"),
    ("slasher", "horror"), ("heist", "crime"), ("noir", "crime"),
    ("dance", "musical"),
]
# Extra coarse genres for books (memoir -> biography, dystopian -> sci-fi, ...).
BOOK_GENRE_RULES = GENRE_RULES + [
    ("memoir", "biography"), ("autobiograph", "biography"), ("young adult", "young adult"),
    ("non-fiction", "non-fiction"), ("nonfiction", "non-fiction"), ("essay", "non-fiction"),
    ("self-help", "non-fiction"), ("literary fiction", "literary"), ("poetry", "poetry"),
    ("short stor", "short stories"), ("graphic novel", "graphic novel"), ("dystopia", "sci-fi"),
    ("speculative", "sci-fi"), ("gothic", "horror"), ("crime fiction", "crime"),
    ("love story", "romance"), ("history", "history"), ("children's", "family"),
]


def words(s):
    return len(s.split())


def split_wikitext(wikitext, headings=PLOT_HEADINGS):
    """Return (lead, plot) as plain text from an article's wikitext.

    mwparserfromhell drops templates (infoboxes, citations), <ref> footnotes
    and markup, and keeps the visible text of links. Picture/category links
    are removed first so their captions don't leak into the text.
    """
    code = mwparserfromhell.parse(wikitext)
    # Footnotes, galleries and tables are not prose: remove them entirely
    # (otherwise footnote text like "cite..." leaks into sentences).
    for tag in code.filter_tags():
        if str(tag.tag).strip().lower() in DROP_TAGS:
            try:
                code.remove(tag)
            except ValueError:
                pass
    for link in code.filter_wikilinks():
        if str(link.title).strip().lower().startswith(("file:", "image:", "category:")):
            try:
                code.remove(link)
            except ValueError:  # already removed with its parent
                pass
    lead = clean(code.get_sections(include_lead=True, flat=True)[0].strip_code())
    plot = ""
    for sec in code.get_sections(levels=[2]):
        heading = sec.filter_headings()[0]
        if heading.title.strip_code().strip().lower() in headings:
            for h in sec.filter_headings():  # drop "Plot" and any sub-headings
                sec.remove(h)
            plot = clean(sec.strip_code())
            break
    return lead, plot


def clean(s):
    """Drop sub-headings like '=== Cast ===', blank lines and extra spaces."""
    s = re.sub(r"^=+.*?=+\s*$", " ", s, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", s).strip()


# Words that end in "." without ending a sentence ("Dr. Rao", "vs. Kerala").
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "lt", "col",
                 "gen", "prof", "no", "rs", "mt", "capt", "sgt", "inc", "ltd",
                 "co", "etc", "approx", "smt", "shri", "sri", "u.s", "u.k"}
_BOUNDARY = re.compile(r"[.!?][\"')\]]*\s+(?=[A-Z0-9\"'(])")


_INITIALS = re.compile(r"^([a-z]\.)*[a-z]$")  # "s", "c.j", "a.r" (from S. / C.J. / A.R.)
_EMPTY_BRACKETS = re.compile(r"\s*\(\s*[;,:\s]*\)")  # "()" left by removed templates


def tidy(s):
    """Remove leftovers of removed templates: '()', '( ; )', ' ,' etc."""
    s = _EMPTY_BRACKETS.sub("", s)
    s = re.sub(r"\(\s*[;,]\s*", "(", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def sentences(s):
    """Split text into sentences without breaking at initials or abbreviations.

    'Directed by S. S. Rajamouli. It stars Prabhas.' -> 2 sentences, because a
    single capital letter before the full stop ('S.') is treated as an initial.
    """
    out, start = [], 0
    for m in _BOUNDARY.finditer(s):
        before = s[start:m.start() + 1].split()
        last = before[-1].rstrip(".!?\"')]").lower() if before else ""
        if _INITIALS.match(last) or last in ABBREVIATIONS:
            continue  # initial or abbreviation, not the end of a sentence
        out.append(s[start:m.end()].strip())
        start = m.end()
    if s[start:].strip():
        out.append(s[start:].strip())
    return out


def take_sentences(sents, budget, size=words):
    """Take whole sentences from the start while they fit in budget.

    Never cuts a sentence: a sentence that doesn't fit ends the selection
    (so the result can be empty if the first sentence is already too long).
    """
    out, used = [], 0
    for sent in sents:
        n = size(sent)
        if used + n > budget:
            break
        out.append(sent)
        used += n
    return out


def display_title(wiki_title):
    """'Prince (2010 film)' -> 'Prince'; 'Drishyam 2' stays as it is."""
    return re.sub(r"\s*\([^()]*\bfilm\)$", "", wiki_title).strip()


def normalise_genre(label):
    """'romantic comedy film' -> 'romantic comedy'."""
    label = label.lower().strip()
    return re.sub(r"\s+(film|movie)s?$", "", label)


def coarse_genres(raw_labels, rules=GENRE_RULES):
    found = set()
    for label in raw_labels:
        low = label.lower()
        for key, genre in rules:
            if key in low:
                found.add(genre)
    return sorted(found)


def build_short_text(title, genres, first_sentence, plot_sents, budget, size=words):
    """Method (a): one short text per film, cut only at sentence boundaries.

    Title + genres + first lead sentence (what kind of film it is) + as many
    whole plot sentences as fit in the budget. The first lead sentence is kept
    only if it fits; it is never cut.
    """
    head = f"{title}. Genres: {', '.join(genres) or 'unknown'}."
    parts = [head]
    used = size(head)
    if first_sentence and used + size(first_sentence) <= budget:
        parts.append(first_sentence)
        used += size(first_sentence)
    parts += take_sentences(plot_sents, budget - used, size)
    return " ".join(parts)


def chunk_prefix(film):
    """'Title (genre, genre)', put in front of every chunk of a film."""
    return f"{film['title']} ({', '.join(film['genres_fine']) or 'film'})"


def plot_chunks(prefix, plot_sents, budget, size=words):
    """Method (b): split the plot into chunks of whole sentences.

    Each chunk starts with a short prefix (title + genres) so the model knows
    what it is about. A single sentence longer than the budget becomes its own
    chunk (the model then truncates it; this is rare and counted in Phase 2).
    """
    chunks, cur = [], []
    prefix = prefix + ": "
    for sent in plot_sents:
        if cur and size(prefix + " ".join(cur + [sent])) > budget:
            chunks.append(prefix + " ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        chunks.append(prefix + " ".join(cur))
    return chunks
