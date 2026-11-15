import copy
import json

import pytest

from app import db, profile, validation


@pytest.fixture
def setups(fresh_db, make_gguf, monkeypatch):
    monkeypatch.setattr(profile, "limits", lambda: {})
    make_gguf("validate.gguf")
    eid = profile.create_experiment("Validate", "validate.gguf")
    a = profile.create_run(eid, {"config": {}})
    b = profile.create_run(eid, {"config": {"kv": "q4_0"}})
    for rid in (a, b):
        profile._save(rid, status="done", results={"provenance": {"protocol": profile.PROTOCOL, "model": {"file": "validate.gguf", "sha256": "test"}}})
    return eid, a, b


def test_batch_is_alternating_and_frozen(setups):
    eid, a, b = setups
    protocol = profile.set_protocol(eid, {"mmlu_per_category": 100})
    batch = validation.create(eid, {"reference_id": a, "candidate_id": b})
    exp = profile.get_experiment(eid)
    queued = exp["runs_list"][2:]
    assert len(queued) == 12
    assert [r["parent_id"] for r in queued] == [a, b] * 6
    assert [r["validation"]["arm"] for r in queued] == ["reference", "candidate"] * 6
    assert all(r["protocol_snapshot"] == {"version": profile.PROTOCOL, "tests": protocol} for r in queued)
    profile.set_protocol(eid, {})
    assert profile.get_experiment(eid)["runs_list"][2]["protocol_snapshot"]["tests"] == protocol
    validation.cancel(eid, batch["id"])
    exp = profile.get_experiment(eid)
    assert all(r["status"] == "cancelled" for r in exp["runs_list"][2:])
    assert exp["validations"][0]["summary"]["conclusion"] == "insufficient"


def test_invalid_batch_inserts_nothing(setups):
    eid, a, b = setups
    for patch in ({"candidate_id": a}, {"pairs": 13}, {"metric": "roof"}, {"min_effect_pct": float("nan")}, {"candidate_id": 999}):
        with pytest.raises(ValueError):
            validation.create(eid, {"reference_id": a, "candidate_id": b, **patch})
    assert len(profile.get_experiment(eid)["runs_list"]) == 2


def completed_fixture(improvements):
    plan = {"metric": "tg", "pairs": len(improvements), "min_effect_pct": 3, "protocol": profile.PROTOCOL,
            "sources": [{"config": {"kv": "f16"}}, {"config": {"kv": "q4_0"}}], "run_ids": []}
    runs = []
    for pair, improvement in enumerate(improvements):
        ids = []
        for arm in (0, 1):
            rid = pair * 2 + arm + 1
            ids.append(rid)
            runs.append({"id": rid, "status": "done", "config": plan["sources"][arm]["config"], "results": {
                "speed": {"tg": 100 * (1 + improvement / 100) if arm else 100},
                "provenance": {"protocol": profile.PROTOCOL, "model": {"sha256": "same"}, "host": {"device": "test"},
                               "data": {"hash": "data"}, "gpu": {"name": "test"}, "llama_cpp": {"commit": "test"}}}})
        plan["run_ids"].append(ids)
    return plan, runs


def test_conclusion_requires_all_pairs_and_effect_interval():
    plan, runs = completed_fixture([8, 9, 10, 11, 12, 13])
    s = validation.summarize(plan, runs)
    assert s["conclusion"] == "candidate_better" and s["stats"]["p"] == pytest.approx(0.03125)
    assert s["stats"]["ci95_pct"] == pytest.approx([8, 13])
    for mutate in (lambda rs: rs.pop(), lambda rs: rs[0].update(status="error"),
                   lambda rs: rs[0]["results"]["provenance"]["host"].update(device="other"),
                   lambda rs: rs[2]["results"]["provenance"]["model"].update(sha256="changed"),
                   lambda rs: rs[0]["results"]["speed"].update(tg=None)):
        altered = copy.deepcopy(runs)
        mutate(altered)
        assert validation.summarize(plan, altered)["conclusion"] == "insufficient"
    runs[0]["status"] = "queued"
    assert validation.summarize(plan, runs)["conclusion"] == "pending"


def test_small_noisy_and_tied_samples_have_no_winner():
    for improvements in ([10] * 3, [10, -10, 12, -9, 0, 1], [0] * 6, [1] * 6):
        assert validation.summarize(*completed_fixture(improvements))["conclusion"] == "insufficient"
    assert validation.summarize(*completed_fixture([-10] * 6))["conclusion"] == "reference_better"


def test_retained_evidence_does_not_resolve_imported_local_paths(setups):
    eid, a, _ = setups
    profile._retain_output(a, "speed", "raw benchmark output")
    assert profile.evidence_outputs(a)["outputs"]["speed"]["text"] == "raw benchmark output"
    profile._save(a, results={"imported_from": {"device": "elsewhere"}})
    assert profile.evidence_outputs(a)["outputs"] == {}
    with pytest.raises(ValueError):
        profile.evidence_outputs(99999)
