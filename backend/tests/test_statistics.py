"""The statistics every comparison in Blocky rests on."""
import math

from app import profile


def test_wilson_interval_contains_the_estimate_and_stays_in_bounds():
    lo, hi = profile.wilson(336, 560)
    assert lo < 336 / 560 < hi
    assert 0 <= lo and hi <= 1
    assert profile.wilson(0, 0) == (0.0, 0.0)
    lo, hi = profile.wilson(0, 50)
    assert lo == 0 and hi > 0          # 0/50 still leaves room above zero
    lo, hi = profile.wilson(50, 50)
    assert hi == 1 and lo < 1


def test_wilson_width_shrinks_with_more_questions():
    w = lambda n: (lambda b: b[1] - b[0])(profile.wilson(n // 2, n))  # noqa: E731
    assert w(560) < w(280) < w(50)
    assert abs(w(560) - 2 * 1.96 * math.sqrt(0.25 / 560)) < 0.01   # ≈ normal approximation at p = 0.5


def test_mcnemar_identical_answers_are_not_significant():
    a = "1010110011" * 56
    assert profile.mcnemar_p(a, a) == 1.0


def test_mcnemar_detects_a_clear_change_and_is_symmetric():
    base = "1" * 300 + "0" * 260
    worse = "0" * 40 + "1" * 260 + "0" * 260      # lost 40, gained 0
    p = profile.mcnemar_p(base, worse)
    assert p < 1e-6
    assert profile.mcnemar_p(worse, base) == p


def test_mcnemar_small_imbalance_is_not_significant():
    base = "1" * 10 + "0" * 10 + "1" * 540
    other = "0" * 10 + "1" * 8 + "0" * 2 + "1" * 540   # lost 10, gained 8
    assert profile.mcnemar_p(base, other) > 0.5


def test_mcnemar_rejects_mismatched_question_sets():
    assert profile.mcnemar_p("101", "1010") is None
    assert profile.mcnemar_p("", "") is None


def test_robust_stats_ignore_impossible_repetitions():
    st = profile.robust_stats([60.03, 0.0, 58.44, 113831.0, 57.68])
    assert st["median"] == 58.44
    assert st["dropped"] == 2
    assert st["sd"] < 2


def test_robust_stats_spread_matches_sigma_for_clean_data():
    st = profile.robust_stats([10.0, 11.0, 9.0, 10.5, 9.5])
    assert st["median"] == 10.0 and st["dropped"] == 0
    assert 0.5 < st["sd"] < 1.0


def test_robust_stats_all_bad():
    assert profile.robust_stats([0, 0]) == {"median": None, "sd": None, "dropped": 2}


def test_parse_bench_reads_llama_bench_jsonl():
    out = "\n".join([
        "some log line",
        '{"n_prompt": 512, "n_gen": 0, "n_depth": 0, "samples_ts": [1700, 1750, 1720, 1730, 1690]}',
        '{"n_prompt": 0, "n_gen": 128, "n_depth": 0, "samples_ts": [62.1, 62.5, 0.0, 61.9, 62.3]}',
        '{"n_prompt": 0, "n_gen": 128, "n_depth": 4096, "samples_ts": [60.0, 59.5, 59.8, 59.9, 60.1]}',
    ])
    r = profile.parse_bench(out)
    assert r["pp"] == 1720 and r["tg"] == 62.3 and r["tg_d4096"] == 59.9
    assert r["tg_dropped"] == 1 and "pp_dropped" not in r
    assert r["pp_samples"] == [1700, 1750, 1720, 1730, 1690]
