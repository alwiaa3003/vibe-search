# Credits

This repository contains **no downloaded data**, only scripts that fetch it from
the public sources below. Fetched data stays on your own machine.

## Wikidata

Movie and book metadata (titles, release/publication years, genres, languages,
authors, authors' countries and links to Wikipedia articles) comes from
[Wikidata](https://www.wikidata.org). Wikidata's structured data is released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain).

## Wikipedia

Movie and book descriptions (article lead sections and plot/synopsis sections)
come from [English Wikipedia](https://en.wikipedia.org), written by Wikipedia's
contributors. The text is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Text shown or derived from it in this project is credited to Wikipedia and its
contributors. Each movie and book keeps a link to its Wikipedia article, where
the full author history is available. Film songs are searched using their
film's Wikipedia text, and each film song links to that article too.

## MusicBrainz

Song titles, artists, release dates, languages and links between soundtracks and
their tracks come from [MusicBrainz](https://musicbrainz.org), the open music
encyclopedia, via its web service. MusicBrainz **core data** is released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Tags (used as
genre/style words for non-film singles) are MusicBrainz **supplementary data**,
licensed [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/):
credit to MusicBrainz and its contributors; non-commercial use only; data derived
from the tags must be shared under the same licence. Because of this, the songs
part of this project is non-commercial. See
<https://musicbrainz.org/doc/About/Data_License>.

Only song titles and artist names are shown. No lyrics and no audio are used.

## Models

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
  Apache License 2.0 (embeddings).
- [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
  Apache License 2.0 (optional re-ranker; trained on MS MARCO, whose data terms are non-commercial).
