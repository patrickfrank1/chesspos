import chess

from bdd import given, then, when
from src.evaluation.observability.render import render_report
from src.evaluation.observability.samples import make_sample
from src.dataset.token_stream import TokenStreamEncoder


def metric(top1=0.6, mrr=0.7, n=10):
    return {
        "top1": top1,
        "mrr": mrr,
        "mean_rank": 2.0,
        "n_queries": n,
        "chance_top1": 0.2,
        "chance_mrr": 0.4567,
        "chance_mean_rank": 3.0,
    }


def example(query_fen, candidate_fens, group="fork", positive=0):
    candidates = [
        {
            "sample_id": index,
            "fen": fen,
            "score": 0.9 - index * 0.1,
            "rank": float(index + 1),
            "is_positive": index == positive,
            "themes": [group] if index == positive else [],
        }
        for index, fen in enumerate(candidate_fens)
    ]
    return {
        "group": group,
        "task": "motif",
        "class_label": group,
        "matched": True,
        "query_sample_id": 0,
        "query_fen": query_fen,
        "positive_rank": float(positive + 1),
        "top1": positive == 0,
        "candidates": candidates,
    }


def build_report():
    encoder = TokenStreamEncoder()
    start = encoder.encode(chess.Board())
    sparse = encoder.encode(chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"))
    samples = [
        make_sample(start, "test", sample_id=0),
        make_sample(sparse, "test", sample_id=1),
    ]
    fens = [sample.fen for sample in samples]
    ranking = example(fens[0], fens)
    return {
        "info": {"path": "tiny.pt", "step": 10, "val_loss": 1.0},
        "n_samples": len(samples),
        "test": {
            "n": len(samples),
            "metrics": {"random": metric(), "matched": metric(0.5, 0.6)},
            "examples": {
                "matched": {
                    "best": [ranking],
                    "random": [ranking],
                    "worst": [ranking],
                }
            },
        },
        "motifs": {
            "n": len(samples),
            "themes": ["fork"],
            "metrics": {
                "per_theme": {
                    "fork": {"random": metric(), "matched": metric(0.3, 0.5)}
                },
                "mean_random": metric(),
                "mean_matched": metric(0.3, 0.5),
            },
            "examples": {
                "matched": {
                    "fork": {
                        "best": [ranking],
                        "random": [ranking],
                        "worst": [ranking],
                    }
                }
            },
        },
        "surrogates": {
            "n": len(samples),
            "source": "test",
            "tasks": [
                {
                    "name": "material",
                    "label": "Material balance",
                    "classes": ["equal"],
                    "description": "Material balance in piece points.",
                    "class_definitions": [["equal", "between -2 and +2"]],
                }
            ],
            "metrics": {
                "per_task": {
                    "material": {
                        "label": "Material balance",
                        "classes": ["equal"],
                        "per_class": {
                            "equal": {
                                "random": metric(),
                                "matched": metric(0.4, 0.55),
                            }
                        },
                        "mean_random": metric(),
                        "mean_matched": metric(0.4, 0.55),
                    }
                },
                "mean_random": metric(),
                "mean_matched": metric(0.4, 0.55),
            },
            "examples": {
                "material": {
                    "equal": {
                        "matched": {
                            "best": [ranking],
                            "random": [ranking],
                            "worst": [ranking],
                        }
                    }
                }
            },
        },
    }


def test_render_report_has_focused_tabs_and_examples(tmp_path):
    with given("a focused report with test, motif, and surrogate data"):
        report = build_report()
        path = tmp_path / "report.html"

    with when("the report is rendered"):
        written = render_report(path, report=report, modes=("matched",))

    with then("the tabs, tables, and candidate boards are present"):
        assert written == path
        html = path.read_text()
        assert "Position similarity ranking" in html
        assert "data-tab='test'" in html
        assert "data-tab='motifs'" in html
        assert "data-tab='surrogates'" in html
        assert "in-game vs out-of-game" in html
        assert "Motif labels" in html
        assert "Surrogate classes" in html
        assert "Material balance" in html
        assert "positive #" in html
        assert "positive rank" in html
        assert "data-fen=" in html
        assert "Best 3" in html
        assert "Random 3" in html
        assert "Worst 3" in html
        assert "Material balance in piece points" in html
        assert "between -2 and +2" in html
        assert "<details>" in html
        assert "<script src=" not in html
        assert 'src="http' not in html
    return


def test_render_report_escapes_untrusted_labels(tmp_path):
    with given("a theme containing HTML markup"):
        report = build_report()
        report["motifs"]["themes"] = ["<script>alert(1)</script>"]
        report["motifs"]["metrics"]["per_theme"] = {
            "<script>alert(1)</script>": {
                "random": metric(),
                "matched": metric(),
            }
        }
        report["motifs"]["examples"] = {
            "matched": {
                "<script>alert(1)</script>": {
                    "best": [],
                    "random": [],
                    "worst": [],
                }
            }
        }
        path = tmp_path / "report.html"

    with when("the report is rendered"):
        render_report(path, report=report, modes=("matched",))

    with then("the markup is escaped"):
        html = path.read_text()
        assert "&lt;script&gt;" in html
        assert "<script>alert(1)</script>" not in html
    return


def test_render_report_handles_empty_tasks(tmp_path):
    with given("a report with no test, motif, or surrogate data"):
        report = {
            "info": {"path": "tiny.pt"},
            "n_samples": 0,
            "test": {"n": 0, "metrics": None, "examples": {}},
            "motifs": {"n": 0, "themes": [], "metrics": None, "examples": {}},
            "surrogates": {
                "n": 0,
                "source": "all",
                "tasks": [],
                "metrics": None,
                "examples": {},
            },
        }
        path = tmp_path / "report.html"

    with when("the report is rendered"):
        render_report(path, report=report, modes=("matched",))

    with then("empty-state messages are present"):
        html = path.read_text()
        assert "No test samples" in html
        assert "No puzzle samples" in html
        assert "No surrogate ranking metrics" in html
    return
