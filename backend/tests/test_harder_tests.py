"""Larger MMLU-Pro sets (nested), the RULER-style suite, protocol validation, and Holm correction."""
import random

import pytest

from app import benchmarks, evals, profile, ruler


@pytest.fixture
def fake_mmlu(monkeypatch, tmp_path):
    cats = [f"cat{i:02d}" for i in range(14)]
    qs = [{"q": f"{c} question {j}", "options": ["a", "b", "c", "d"], "answer": "A", "tag": c} for c in cats for j in range(300)]
    f = tmp_path / "mmlu.json"
    f.write_text("[]")
    monkeypatch.setattr(benchmarks, "path", lambda name: f)
    monkeypatch.setattr(evals, "bench_questions", lambda name: qs)


def test_mmlu_sizes_are_nested_and_never_test_the_shots(fake_mmlu):
    shots, s560 = profile.mmlu_sample()
    _, s1400 = profile.mmlu_sample(per_category=100)
    _, s3500 = profile.mmlu_sample(per_category=250)
    assert (len(s560), len(s1400), len(s3500)) == (560, 1400, 3500)
    key = lambda q: q["q"]  # noqa: E731
    assert [key(q) for q in s1400[:560]] == [key(q) for q in s560]          # paired tests compare the common prefix
    assert [key(q) for q in s3500[:1400]] == [key(q) for q in s1400]
    assert not {key(q) for q in shots} & {key(q) for q in s3500}          # few-shot examples are never test questions
    assert len({key(q) for q in s3500}) == 3500


def test_protocol_validation(fresh_db, make_gguf):
    make_gguf("M.gguf")
    eid = profile.create_experiment("x", "M.gguf")
    p = profile.set_protocol(eid, {"mmlu_per_category": 100, "ruler": {"samples": 8}})
    assert p == {"mmlu_per_category": 100, "ruler": {"samples": 8}}
    assert profile.set_protocol(eid, {"mmlu_per_category": 40}) == {}      # the default isn't stored
    with pytest.raises(ValueError):
        profile.set_protocol(eid, {"mmlu_per_category": 77})


@pytest.mark.parametrize("task", ruler.TASKS)
def test_ruler_samples_have_their_answers_in_the_prompt(task):
    text = "The quick brown fox jumps over the lazy dog. " * 4000
    s = ruler.make_sample(task, 4096, 4.0, text, random.Random(5))
    assert all(a in s["prompt"] for a in s["answers"])
    assert ruler.score(task, " ".join(s["answers"]), s["answers"]) == 1.0
    assert ruler.score(task, "I don't know", s["answers"]) == 0.0
    # Same seed, same sample: every run of an experiment sees identical prompts.
    assert ruler.make_sample(task, 4096, 4.0, text, random.Random(5)) == s


def test_ruler_partial_credit():
    assert ruler.score("niah_multivalue", "1111111 and 2222222", ["1111111", "2222222", "3333333", "4444444"]) == 0.5


def test_holm():
    assert profile.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert profile.holm([]) == []
    assert profile.holm([0.5]) == [0.5]
