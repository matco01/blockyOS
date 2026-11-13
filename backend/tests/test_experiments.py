"""Experiment bookkeeping: labels, comparisons, sweeps, the run tree, and moving results between devices."""
import json

import pytest

from app import gguf, profile


@pytest.fixture
def no_hardware(monkeypatch):
    """Tests don't have a GPU: give the profiler fixed hardware limits."""
    lim = {"gpu": "NVIDIA GeForce RTX 9999", "gpu_bandwidth_gbps": 500, "vram_mb": 8192, "ram_mb": 16384, "cpu": "Test CPU", "cpu_threads": 8}
    monkeypatch.setattr(profile, "limits", lambda: lim)
    return lim


def test_tokenizer_signature_and_metadata(make_gguf):
    m = gguf.metadata(make_gguf("A-Q4_K_M.gguf", vocab=40))
    assert m["architecture"] == "qwen35" and m["layers"] == 4 and m["quant"] == "Q4_K_M"
    assert m["tokenizer"] == "gpt2/qwen35/40"
    other = gguf.metadata(make_gguf("B.gguf", vocab=41))
    assert other["tokenizer"] != m["tokenizer"]


def test_weights_label():
    assert profile.weights_label("Qwen3.5-9B-Q3_K_M-imat-custom.gguf") == "Q3_K_M-imat-custom"
    assert profile.weights_label("Qwen3.5-9B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert profile.weights_label("Qwen3.5-9B-BF16.gguf") == "BF16"
    assert profile.weights_label("plain.gguf") == "plain"


def test_describe_change(make_gguf):
    make_gguf("Fam-Q4_K_M.gguf", model_name="Fam")
    make_gguf("Fam-Q3_K_M.gguf", model_name="Fam", ftype=12)
    make_gguf("Other-Q4_K_M.gguf", model_name="Other")
    base = {"model": "Fam-Q4_K_M.gguf", "kv": "q8_0", "ctx": 8192, "ngl": "auto", "vram_limit_mb": None, "fa": "auto", "threads": None, "engine": "stock"}
    assert profile.describe_change(base, dict(base)) == "Repeat of baseline"
    assert profile.describe_change(base, dict(base, kv="q4_0")) == "KV q4_0"
    assert profile.describe_change(base, dict(base, model="Fam-Q3_K_M.gguf")) == "Q3_K_M"        # same model: just the weights
    assert profile.describe_change(base, dict(base, model="Other-Q4_K_M.gguf")) == "Other-Q4_K_M"  # different model: full name
    assert profile.describe_change(base, dict(base, engine="upstream")) == "engine upstream"
    assert profile.describe_change(base, dict(base, vram_limit_mb=4096, ctx=16384)) == "ctx 16k · 4 GB GPU"


def _prov(**over):
    p = {"protocol": "v2", "llama_cpp": {"commit": "abc"}, "gpu": {"name": "GPU A", "driver": "1", "cuda_toolkit": "12.9", "vram_mb": 8192},
         "data": {"mmlu_pro_sha256": "x"}, "host": {"wsl": False, "cpu": "CPU A", "ram_gb": 16, "device": "box-a"}, "model": {"file": "m.gguf", "sha256": "h"}}
    for k, v in over.items():
        p[k] = {**p[k], **v}
    return p


def _run(i, prov, cfg=None, answers=None, parent=None):
    return {"id": i, "status": "done", "config": cfg or {"engine": "stock"}, "parent_id": parent,
            "results": {"provenance": prov, **({"mmlu": {"answers": answers}} if answers else {})}}


def test_compare_separates_software_and_hardware_differences():
    base = _run(1, _prov())
    same = profile.compare(base, _run(2, _prov()))
    assert same["env_mismatch"] == [] and same["hw_diff"] == []
    driver = profile.compare(base, _run(3, _prov(gpu={"driver": "2"})))
    assert driver["env_mismatch"] == ["GPU driver"]
    other_box = profile.compare(base, _run(4, _prov(gpu={"name": "GPU B"}, host={"device": "box-b"})))
    assert set(other_box["hw_diff"]) == {"GPU", "Device"} and other_box["env_mismatch"] == []


def test_compare_ignores_the_engine_commit_when_the_engine_is_the_variable():
    base = _run(1, _prov())
    r = profile.compare(base, _run(2, _prov(llama_cpp={"commit": "def"}), cfg={"engine": "upstream"}))
    assert "llama.cpp commit" not in r["env_mismatch"]
    r = profile.compare(base, _run(3, _prov(llama_cpp={"commit": "def"})))
    assert "llama.cpp commit" in r["env_mismatch"]


def test_compare_runs_the_paired_test():
    a = "1" * 300 + "0" * 260
    r = profile.compare(_run(1, _prov(), answers=a), _run(2, _prov(), answers="0" * 40 + a[40:]))
    assert r["mmlu_flips"] == {"lost": 40, "gained": 0} and r["mmlu_p"] < 1e-6


def test_enrich_links_old_runs_to_the_baseline_and_compares_with_parents(fresh_db):
    a = "1" * 100 + "0" * 100
    runs = [_run(1, _prov(), answers=a), _run(2, _prov(), answers=a), _run(3, _prov(), answers="0" * 30 + a[30:], parent=2)]
    for r in runs:
        r["config"]["model"] = "missing.gguf"
    exp = profile.enrich({"limits": {}, "runs_list": runs})
    assert exp["runs_list"][1]["parent_id"] == 1           # legacy run without parent -> baseline
    s = exp["runs_list"][2]["stats"]
    assert s["baseline_id"] == 1 and s["vs_parent"]["mmlu_flips"] == {"lost": 30, "gained": 0}


def test_sweep_validation_and_size(fresh_db, make_gguf, no_hardware):
    make_gguf("S-Q4_K_M.gguf")
    eid = profile.create_experiment("sweep test", "S-Q4_K_M.gguf")
    with pytest.raises(ValueError):
        profile.create_sweep(eid, {"grid": {}})
    with pytest.raises(ValueError, match="limit"):
        profile.create_sweep(eid, {"grid": {"ctx": [4096, 8192, 16384, 32768, 65536, 131072], "kv": ["f16", "q8_0", "q4_0"]}, "repeats": 2})
    ids = profile.create_sweep(eid, {"grid": {"kv": ["f16", "q4_0"], "ctx": [4096, 8192]}})
    assert len(ids) == 4
    exp = profile.get_experiment(eid)
    labels = [r["label"] for r in exp["runs_list"]]
    assert labels[0].startswith("Baseline")                # the first run of a fresh experiment
    assert all(r["status"] == "queued" for r in exp["runs_list"])


def test_branching_records_the_parent_and_labels_relative_to_it(fresh_db, make_gguf, no_hardware):
    make_gguf("T-Q4_K_M.gguf")
    eid = profile.create_experiment("tree", "T-Q4_K_M.gguf")
    b = profile.create_run(eid, {"config": {}})
    exp = profile.get_experiment(eid)
    cfg = exp["runs_list"][0]["config"]
    c = profile.create_run(eid, {"config": dict(cfg, kv="q4_0"), "parent_id": b})
    d = profile.create_run(eid, {"config": dict(cfg, kv="q4_0", ctx=16384), "parent_id": c})
    rows = {r["id"]: r for r in profile.get_experiment(eid)["runs_list"]}
    assert rows[c]["parent_id"] == b and rows[c]["label"] == "KV q4_0"
    assert rows[d]["parent_id"] == c and rows[d]["label"] == "ctx 16k"        # relative to its parent, not the baseline
    with pytest.raises(ValueError):
        profile.create_run(eid, {"config": cfg, "parent_id": 99999})


def test_deleting_a_run_reattaches_its_children(fresh_db, make_gguf, no_hardware):
    make_gguf("U-Q4_K_M.gguf")
    eid = profile.create_experiment("del", "U-Q4_K_M.gguf")
    b = profile.create_run(eid, {"config": {}})
    cfg = profile.get_experiment(eid)["runs_list"][0]["config"]
    c = profile.create_run(eid, {"config": dict(cfg, kv="q4_0"), "parent_id": b})
    d = profile.create_run(eid, {"config": dict(cfg, kv="f16"), "parent_id": c})
    from app import db
    with db.connect() as conn:
        conn.execute("UPDATE exp_runs SET status = 'done'")
    profile.delete_run(c)
    rows = {r["id"]: r for r in profile.get_experiment(eid)["runs_list"]}
    assert c not in rows and rows[d]["parent_id"] == b


def test_import_keeps_the_tree_and_hardware(fresh_db, make_gguf, no_hardware):
    export = {"name": "Study on box B", "seed": 1, "limits": {"gpu": "NVIDIA GeForce RTX 1234", "gpu_bandwidth_gbps": 300},
              "runs_list": [
                  {"id": 10, "status": "done", "label": "Baseline · X", "parent_id": None, "config": {"model": "X.gguf"},
                   "results": {"speed": {"tg": 40}, "provenance": {"host": {"device": "box-b"}}}},
                  {"id": 11, "status": "done", "label": "KV q4_0", "parent_id": 10, "config": {"model": "X.gguf"},
                   "results": {"speed": {"tg": 41}, "hardware": {"gpu": "NVIDIA GeForce RTX 1234", "device": "box-b"}}},
                  {"id": 12, "status": "error", "label": "broken", "parent_id": 10, "config": {}, "results": {}},
              ]}
    r = profile.import_experiment(json.loads(json.dumps(export)))
    assert r["runs"] == 2 and r["from"]["device"] == "box-b"
    rows = profile.get_experiment(r["experiment_id"])["runs_list"]
    assert rows[1]["parent_id"] == rows[0]["id"]
    assert rows[0]["results"]["hardware"]["gpu_bandwidth_gbps"] == 300          # older export: snapshot from the file's limits
    assert rows[1]["results"]["imported_from"]["gpu"] == "RTX 1234"

    # Merging into an existing experiment hangs the other device's baseline under ours, labeled with the device.
    make_gguf("Y.gguf")
    eid = profile.create_experiment("local", "Y.gguf")
    base = profile.create_run(eid, {"config": {}})
    from app import db
    with db.connect() as conn:
        conn.execute("UPDATE exp_runs SET status = 'done' WHERE id = ?", (base,))
    m = profile.import_experiment(export, into=eid)
    rows = profile.get_experiment(eid)["runs_list"]
    imported = [x for x in rows if x["id"] != base]
    assert m["runs"] == 2 and imported[0]["parent_id"] == base and imported[0]["label"].endswith("@ box-b")

    with pytest.raises(ValueError):
        profile.import_experiment({"runs_list": []})
