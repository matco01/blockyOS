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


def test_robust_stats_preserve_positive_outliers_and_reject_invalid_samples():
    st = profile.robust_stats([60.03, 0.0, 58.44, 113831.0, 57.68])
    assert st["median"] == 59.23
    assert st["dropped"] == 1
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
    assert r["pp"] == 1720 and r["tg"] == 62.2 and r["tg_d4096"] == 59.9
    assert r["tg_dropped"] == 1 and "pp_dropped" not in r
    assert r["pp_samples"] == [1700, 1750, 1720, 1730, 1690]


def test_paired_diff_ci_basic_properties():
    a = "1" * 300 + "0" * 260
    assert profile.paired_diff_ci(a, a)["diff"] == 0
    lo, hi = profile.paired_diff_ci(a, a)["ci95"]
    assert lo < 0 < hi and hi - lo < 0.02           # no flips: a narrow interval around zero
    b = "0" * 20 + "1" * 280 + "0" * 260            # 20 answers lost
    r = profile.paired_diff_ci(a, b)
    assert abs(r["diff"] - (-20 / 560)) < 1e-4
    assert r["ci95"][0] < r["diff"] < r["ci95"][1] < 0
    # Swapping the runs mirrors the interval.
    m = profile.paired_diff_ci(b, a)
    assert abs(m["ci95"][0] + r["ci95"][1]) < 1e-4 and abs(m["ci95"][1] + r["ci95"][0]) < 1e-4
    assert profile.paired_diff_ci(a, b[:-1]) is None


def test_paired_diff_ci_covers_the_true_difference_about_95_percent_of_the_time():
    import random
    rng = random.Random(7)
    n, p_a, flip_lose, flip_gain = 560, 0.55, 0.06, 0.04   # 6% of questions lost, 4% gained: true difference -2 points
    hits = trials = 0
    for _ in range(1500):
        a, b = [], []
        for _ in range(n):
            x = rng.random() < p_a
            if x:
                y = rng.random() >= flip_lose / p_a
            else:
                y = rng.random() < flip_gain / (1 - p_a)
            a.append("1" if x else "0"); b.append("1" if y else "0")
        true = flip_gain - flip_lose
        lo, hi = profile.paired_diff_ci("".join(a), "".join(b))["ci95"]
        hits += lo <= true <= hi
        trials += 1
    assert 0.93 <= hits / trials <= 0.975


def test_experiment_records_perplexity_sample_size(fresh_db, make_gguf):
    make_gguf("chunks.gguf")
    eid = profile.create_experiment("chunks", "chunks.gguf")
    assert profile.set_protocol(eid, {"ppl_chunks": 128}) == {"ppl_chunks": 128}
    assert profile.ppl_chunks(profile.get_experiment(eid)) == 128
    assert profile.set_protocol(eid, {"ppl_chunks": 32}) == {}
    import pytest
    with pytest.raises(ValueError):
        profile.set_protocol(eid, {"ppl_chunks": 100})


def test_kld_reference_must_use_the_same_text(tmp_path):
    import json
    import pytest
    ref = tmp_path / "base.kld"
    ref.write_bytes(b"x")
    ref.with_suffix(".json").write_text(json.dumps({"run_id": 1, "protocol": profile.PROTOCOL}))   # older reference: default size
    profile._check_kld_reference(ref, {"id": 1}, profile.PPL_CHUNKS)
    with pytest.raises(RuntimeError, match="chunks"):
        profile._check_kld_reference(ref, {"id": 1}, 128)


def test_compare_flags_different_perplexity_samples():
    def run(rid, chunks):
        prov = {"protocol": profile.PROTOCOL, "extra_tests": {"ppl_chunks": chunks} if chunks else {}}
        return {"id": rid, "config": {}, "results": {"provenance": prov}}
    assert "perplexity/KLD sample size" not in profile.compare(run(1, None), run(2, 32))["env_mismatch"]
    assert "perplexity/KLD sample size" in profile.compare(run(1, None), run(2, 128))["env_mismatch"]
