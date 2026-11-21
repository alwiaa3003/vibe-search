"""Text helpers: sentence splitting, cutting only at sentence ends, chunking."""
from text_utils import (build_short_text, coarse_genres, display_title,
                        normalise_genre, plot_chunks, sentences, split_wikitext,
                        take_sentences, tidy, words)


def test_sentences_keeps_initials_and_abbreviations():
    s = "Directed by S. S. Rajamouli. It stars Prabhas vs. Rana. Dr. Rao helps. The end"
    assert sentences(s) == ["Directed by S. S. Rajamouli.", "It stars Prabhas vs. Rana.",
                            "Dr. Rao helps.", "The end"]


def test_sentences_dotted_initials():
    s = "It was produced by C.J. Rao and A.R. Rahman. Music is great."
    assert sentences(s) == ["It was produced by C.J. Rao and A.R. Rahman.", "Music is great."]


def test_tidy_removes_template_leftovers():
    assert tidy("Bezubaan Ishq () is a 2015 film ( ; ) about love , fate .") == \
        "Bezubaan Ishq is a 2015 film about love, fate."


def test_take_sentences_never_cuts_mid_sentence():
    sents = ["one two three.", "four five.", "six."]
    assert take_sentences(sents, 5) == ["one two three.", "four five."]
    assert take_sentences(sents, 4) == ["one two three."]
    assert take_sentences(sents, 2) == []  # first sentence too long: nothing, not a cut


def test_short_text_fits_budget_and_uses_whole_sentences():
    plot = sentences("A man meets a ghost. The ghost is lonely. They become friends. The end.")
    text = build_short_text("Boo", ["comedy"], "Boo is a 2015 film.", plot, budget=16)
    assert words(text) <= 16
    assert text.startswith("Boo. Genres: comedy. Boo is a 2015 film.")
    assert text.endswith(".")  # ends on a sentence boundary


def test_plot_chunks_respect_budget():
    plot = sentences("One two three four. Five six seven. Eight nine. Ten eleven twelve.")
    chunks = plot_chunks("T", plot, budget=8)
    assert all(words(c) <= 8 for c in chunks)
    assert " ".join(c.replace("T: ", "") for c in chunks) == " ".join(plot)


def test_display_title():
    assert display_title("Prince (2010 film)") == "Prince"
    assert display_title("Lucia (2013 Kannada film)") == "Lucia"
    assert display_title("Drishyam 2") == "Drishyam 2"
    assert display_title("Super (filmmaker)") == "Super (filmmaker)"


def test_genres():
    assert normalise_genre("Romantic comedy film") == "romantic comedy"
    assert coarse_genres(["romantic comedy film", "masala film"]) == ["action", "comedy", "romance"]


def test_split_wikitext_lead_and_plot():
    wt = """{{Infobox film|name=X}}
'''X''' is a 2015 [[Tamil language|Tamil]] film.<ref>cite</ref>
[[File:Poster.jpg|thumb|A caption that must not appear]]
== Plot ==
A boy finds a [[dog]]. They go home.
=== Part two ===
More plot.
== Cast ==
* Someone
"""
    lead, plot = split_wikitext(wt)
    assert lead == "X is a 2015 Tamil film."
    assert plot == "A boy finds a dog. They go home. More plot."
    assert "caption" not in lead + plot
