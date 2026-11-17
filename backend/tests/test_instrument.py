"""Instrument checks: properties a correct measurement pipeline must have, judged from stored results."""
import json

from app import instrument

BASE_CFG = {"model": "m-F16.gguf", "kv": "f16", "ctx": 4096}


def run(rid, cfg=None, **res):
    return {"id": rid, "status": "done", "config": cfg or dict(BASE_CFG), "results": res, "stats": res.pop("stats", {})}


def exp(*runs):
    return {"runs_list": list(runs)}


def by_id(report):
    return {c["id"]: c for c in report["checks"]}


def test_nothing_to_check_without_runs():
    assert instrument.check(exp(), [])["checks"] == []


def test_missing_reruns_say_what_to_add():
    c = by_id(instrument.check(exp(run(1, kld=0.0)), []))
    assert {c[k]["status"] for k in ("self_kld", "determinism", "speed_repeat", "quant_order", "reference_values")} == {"not_run"}
    assert "Repeat of baseline" in c["self_kld"]["todo"]


def test_healthy_reruns_pass():
    answers = "10" * 280
    base = run(1, kld=0.0, same_top=100.0, mmlu={"answers": answers}, speed={"tg": 50.0})
    rep = run(2, kld=0.00002, same_top=99.9, mmlu={"answers": answers}, speed={"tg": 50.6}, stats={"mmlu_p": 1.0})
    c = by_id(instrument.check(exp(base, rep), []))
    assert c["self_kld"]["status"] == "pass"
    assert c["determinism"]["status"] == "pass"
    assert c["speed_repeat"]["status"] == "pass"


def test_broken_reruns_fail():
    base = run(1, kld=0.0, mmlu={"answers": "1" * 560}, speed={"tg": 50.0})
    rep = run(2, kld=0.05, mmlu={"answers": "1" * 500 + "0" * 60}, speed={"tg": 40.0}, stats={"mmlu_p": 0.0001})
    c = by_id(instrument.check(exp(base, rep), []))
    assert c["self_kld"]["status"] == "fail"
    assert c["determinism"]["status"] == "fail"
    assert c["speed_repeat"]["status"] == "fail"


def _variant(rid, model, bpw, kld):
    params = 1_000_000_000
    return run(rid, {**BASE_CFG, "model": model}, kld=kld, weight_summary={"bytes": params * bpw / 8, "params": params})


def test_quant_order_passes_when_fewer_bits_lose_more():
    runs = [_variant(1, "m-F16.gguf", 16, 0.0), _variant(2, "m-Q8_0.gguf", 8.5, 0.001), _variant(3, "m-Q4_K_M.gguf", 4.8, 0.03),
            _variant(4, "m-Q2_K.gguf", 2.6, 0.4)]
    c = by_id(instrument.check(exp(*runs), []))
    assert c["quant_order"]["status"] == "pass" and abs(c["quant_order"]["value"] + 1) < 1e-9


def test_quant_order_flags_a_clear_inversion():
    runs = [_variant(1, "m-F16.gguf", 16, 0.0), _variant(2, "m-Q8_0.gguf", 8.5, 0.5), _variant(3, "m-Q4_K_M.gguf", 4.8, 0.03)]
    c = by_id(instrument.check(exp(*runs), []))
    assert c["quant_order"]["status"] == "fail" and ("m-Q8_0.gguf", "m-Q4_K_M.gguf") in c["quant_order"]["inversions"]


def test_roofline_exceeded_is_a_warning():
    c = by_id(instrument.check(exp(run(1, hw={"decode": {"roof_util": 1.3}})), []))
    assert c["roofline"]["status"] == "warn"
    c = by_id(instrument.check(exp(run(1, hw={"decode": {"roof_util": 0.7}})), []))
    assert c["roofline"]["status"] == "pass"


def test_reference_values(tmp_path, monkeypatch):
    base = run(1, ppl=6.20, provenance={"model": {"sha256": "abc"}})
    refs = [{"model_sha256": "abc", "metric": "ppl", "value": 6.23, "tolerance": 0.05, "source": "paper X, table 2"}]
    assert by_id(instrument.check(exp(base), refs))["reference_values"]["status"] == "pass"
    refs[0]["value"] = 7.0
    assert by_id(instrument.check(exp(base), refs))["reference_values"]["status"] == "fail"
    # The file loader skips entries without a source or with unknown metrics.
    f = tmp_path / "refs.json"
    f.write_text(json.dumps(refs + [{"model_sha256": "abc", "metric": "ppl", "value": 1, "tolerance": 1, "source": ""},
                                    {"model_sha256": "abc", "metric": "nope", "value": 1, "tolerance": 1, "source": "x"}]))
    monkeypatch.setattr(instrument, "REFERENCE_FILE", f)
    assert len(instrument.load_references()) == 1
