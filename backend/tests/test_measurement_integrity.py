"""Missing instrumentation must not become plausible-looking measurements."""
import asyncio
import json

import httpx
import pytest

from app import power, profile, ruler, scans


def _stream(usage=None):
    events = [{"choices": [{"delta": {"content": "many tokens in one chunk"}}]},
              {"choices": [{"delta": {"content": "another batch"}}]}]
    if usage is not None:
        events.append({"choices": [], "usage": usage})
    return "\n\n".join("data: " + json.dumps(e) for e in events) + "\n\ndata: [DONE]\n\n"


@pytest.mark.parametrize("usage", [None, {}, {"completion_tokens": 0, "prompt_tokens": 0}])
def test_stream_events_are_not_tokens(usage, tmp_path, monkeypatch):
    text = tmp_path / "text.txt"
    text.write_text("example text " * 1000)

    async def ensure_text():
        return text

    async def cpt(*args):
        return 4

    monkeypatch.setattr(profile.quantlab, "ensure_text", ensure_text)
    monkeypatch.setattr(profile, "chars_per_token", cpt)

    async def check():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=_stream(usage)))
        async with httpx.AsyncClient(transport=transport, base_url="http://model") as client:
            speed = await profile.serving_bench(client, "llamacpp", lambda: False)
            assert speed["tg"] is None and speed["pp"] is None
            assert speed["ttft_ms"] is not None
            assert speed["note"]
            tokens, start, end = await profile.sustained_generation(client, None, 4096, lambda: False)
            assert tokens == 0
            assert "error" in power.energy_summary([], [], tokens, end - start)

    asyncio.run(check())


def test_energy_uses_reported_tokens_and_whole_request():
    async def check():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=_stream({"completion_tokens": 123})))
        async with httpx.AsyncClient(transport=transport, base_url="http://model") as client:
            tokens, start, end = await profile.sustained_generation(client, None, 4096, lambda: False)
            assert tokens == 123  # two stream chunks can carry 123 tokens
            assert end > start
    asyncio.run(check())


def test_rejected_benchmark_samples_cannot_return_as_aggregate():
    result = profile.parse_bench(json.dumps({"n_gen": 128, "samples_ts": [0, -1], "avg_ts": 900, "stddev_ts": 0}))
    assert result["tg"] is None and result["tg_sd"] is None
    assert result["tg_dropped"] == 2
    assert profile.parse_bench('{"n_gen":128}')["tg"] is None
    assert profile.robust_stats([float("nan"), float("inf"), -1])["median"] is None


def test_missing_letter_probabilities_fail_measurement(monkeypatch):
    q = {"q": "Question", "options": ["yes", "no"], "answer": "A", "tag": "test"}
    monkeypatch.setattr(profile, "mmlu_sample", lambda *args: ([], [q]))

    async def check():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        async with httpx.AsyncClient(transport=transport, base_url="http://model") as client:
            with pytest.raises(RuntimeError, match="log probabilities"):
                await profile.mmlu_eval(client, 1, lambda: False, lambda *args: None)
    asyncio.run(check())


def test_missing_layer_kld_cannot_rank_as_zero(tmp_path, monkeypatch):
    async def proc(*args, **kwargs):
        return "evaluation output changed format"
    monkeypatch.setattr(scans.quantlab, "_proc", proc)
    with pytest.raises(RuntimeError, match="cannot be ranked"):
        asyncio.run(scans._kld(tmp_path / "m", tmp_path / "base", tmp_path / "text", [], 1))


def test_cpu_sensor_is_reported_as_partial_device_power():
    e = power.energy_summary([{"cpu_power": 10}] * 5, [{"cpu_power": 30}] * 5, 100, 10)
    assert e["primary"] == "cpu"
    assert "CPU package only" in e["primary_label"]
    assert e["sources"]["cpu"]["j_per_token"] == 3


def test_kld_reference_must_belong_to_the_actual_baseline(tmp_path):
    path = tmp_path / "base.kld"
    path.write_bytes(b"test logits")
    for metadata in ({}, {"run_id": 1, "protocol": "old"}, {"run_id": 2, "protocol": profile.PROTOCOL}):
        path.with_suffix(".json").write_text(json.dumps(metadata))
        with pytest.raises(RuntimeError, match="different run/protocol"):
            profile._check_kld_reference(path, {"id": 1})
    path.with_suffix(".json").write_text(json.dumps({"run_id": 1, "protocol": profile.PROTOCOL}))
    profile._check_kld_reference(path, {"id": 1})


def test_protocol_mismatch_does_not_produce_paired_significance():
    def run(i, protocol):
        return {"id": i, "config": {}, "results": {"mmlu": {"answers": "11110000" if i == 1 else "00000000"},
                "provenance": {"protocol": protocol, "model": {"file": "m", "sha256": "x"}}}}
    result = profile.compare(run(1, "v2"), run(2, "v3"))
    assert "protocol" in result["env_mismatch"]
    assert "mmlu_p" not in result


def test_unrun_retrieval_is_missing_not_zero(tmp_path, monkeypatch):
    text = tmp_path / "text.txt"
    text.write_text("haystack")
    async def ensure_text():
        return text
    async def cpt(*args):
        return 4
    monkeypatch.setattr(profile.quantlab, "ensure_text", ensure_text)
    monkeypatch.setattr(profile, "chars_per_token", cpt)
    result = asyncio.run(profile.needle_eval(None, 1024, lambda: False, lambda *args: None))
    assert result["cells"] == [] and result["acc"] is None
    result = asyncio.run(ruler.run(None, "haystack", 4, 1024, 2, lambda: False, lambda *args: None))
    assert result["score"] is None and result["by_task"] == {}


def test_new_runs_cannot_mix_with_an_old_protocol(fresh_db, make_gguf, monkeypatch):
    monkeypatch.setattr(profile, "limits", lambda: {})
    make_gguf("protocol.gguf")
    eid = profile.create_experiment("Protocol", "protocol.gguf")
    rid = profile.create_run(eid, {"config": {}})
    profile._save(rid, status="done", results={"provenance": {"protocol": "old"}})
    with pytest.raises(ValueError, match="new experiment"):
        profile.create_run(eid, {"config": {}})
