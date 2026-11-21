"""Evaluation tooling on toy data: reading labels and scoring runs.

These are TOY labels made up for testing the code only; they are not results.
"""
import csv

import pytest

from evaluate import read_labels, score_runs
from pool import describe
from conftest import FILMS


def write_sheet(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["query_id", "query", "film_id", "label", "comment"])
        w.writeheader()
        w.writerows(rows)


def test_read_labels_skips_blank_and_flags_bad(tmp_path):
    p = tmp_path / "sheet.csv"
    write_sheet(p, [
        {"query_id": "q1", "query": "ghosts", "film_id": "F1", "label": "2", "comment": ""},
        {"query_id": "q1", "query": "ghosts", "film_id": "F2", "label": "", "comment": ""},
        {"query_id": "q1", "query": "ghosts", "film_id": "F3", "label": "yes", "comment": ""},
        {"query_id": "q2", "query": "love", "film_id": "F2", "label": "1", "comment": ""},
    ])
    labels, texts, problems = read_labels(p)
    assert labels == {"q1": {"F1": 2}, "q2": {"F2": 1}}
    assert texts == {"q1": "ghosts", "q2": "love"}
    assert len(problems) == 1 and "yes" in problems[0]


def test_score_runs_toy():
    labels = {"q1": {"F1": 2, "F2": 0, "F3": 1}}
    runs = {"good": {"q1": ["F1", "F3", "F2"]}, "bad": {"q1": ["F2", "F9", "F3"]}}
    s = score_runs(runs, labels)
    assert s["good"]["q1"]["ndcg"] == pytest.approx(1.0)
    assert s["good"]["q1"]["mrr"] == 1.0
    assert s["bad"]["q1"]["mrr"] == pytest.approx(1 / 3)
    assert s["bad"]["q1"]["recall"] == pytest.approx(0.5)
    assert s["bad"]["q1"]["unjudged"] == 1  # F9 was never labelled


def test_write_report_runs_on_toy_data(tmp_path, monkeypatch):
    import evaluate
    monkeypatch.setattr(evaluate, "OUT", tmp_path / "results.md")
    labels = {"q1": {"F1": 2, "F2": 0}, "q2": {"F3": 1, "F4": 2}}
    runs = {"a": {"q1": ["F1", "F2"], "q2": ["F4", "F3"]},
            "b": {"q1": ["F2", "F1"], "q2": ["F5", "F3"]}}
    scores = score_runs(runs, labels)
    evaluate.write_report(scores, runs, labels, {"q1": "ghosts", "q2": "sport"},
                          {"q1": "mood", "q2": "hard"},
                          {"pool_depth": 10, "methods_pooled": 2, "data_date": "toy",
                           "films_in_index": 5}, {f["id"]: f for f in FILMS})
    text = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "| a | 1.000 | 1.000 | 1.000 | 2 | 0 |" in text
    assert "Failure analysis" in text and "Pooling bias" in text


def test_describe_uses_whole_sentences():
    d = describe(FILMS[0], max_words=12)
    assert d.startswith(FILMS[0]["first_sentence"]) and d.endswith(".")
