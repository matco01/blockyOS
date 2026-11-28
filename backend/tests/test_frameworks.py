"""Second framework (vLLM), answer scoring that works for any engine, and GPU kernel profiles."""
import json

import pytest

from app import kernels, profile


def test_letter_choice_renormalizes_over_the_options():
    top = [(" B", -0.2), (" A", -1.8), ("\n", -5.0), (" D", -4.2), (" C", -5.2), (" B", -9.0)]
    pred, p = profile.letter_choice(top, 4, "B")
    assert pred == "B" and 0.7 < p < 0.9
    pred, p = profile.letter_choice(top, 4, "C")
    assert pred == "B" and p < 0.05
    assert profile.letter_choice([("\n", -0.1)], 4, "A") == (None, 0.0)
    pred, _ = profile.letter_choice([(" J", -0.1)], 4, "A")    # J isn't an option when there are only 4
    assert pred is None


@pytest.fixture
def checkpoint():
    d = profile.model_path("hf:Org--Small")
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["X"], "num_hidden_layers": 2}))
    (d / "model.safetensors").write_bytes(b"\0" * 4000)
    return "hf:Org--Small"


def test_checkpoints_need_vllm_and_gguf_needs_llamacpp(make_gguf, checkpoint, monkeypatch):
    monkeypatch.setattr(profile.engines, "vllm_installed", lambda: True)
    make_gguf("G-Q4_K_M.gguf")
    exp = {"model": "G-Q4_K_M.gguf"}
    assert profile.normalize_config(exp, {"model": checkpoint})["engine"] == "vllm"     # picked automatically
    with pytest.raises(ValueError, match="GGUF"):
        profile.normalize_config(exp, {"model": checkpoint, "engine": "stock"})
    with pytest.raises(ValueError, match="checkpoints"):
        profile.normalize_config(exp, {"engine": "vllm"})
    assert profile.normalize_config(exp, {})["engine"] == "stock"


def test_vllm_needs_to_be_installed(make_gguf, checkpoint, monkeypatch):
    monkeypatch.setattr(profile.engines, "vllm_installed", lambda: False)
    make_gguf("H-Q4_K_M.gguf")
    with pytest.raises(ValueError, match="isn't installed"):
        profile.normalize_config({"model": "H-Q4_K_M.gguf"}, {"model": checkpoint})


def test_vllm_does_not_claim_to_apply_llamacpp_settings(checkpoint, monkeypatch):
    monkeypatch.setattr(profile.engines, "vllm_installed", lambda: True)
    assert profile.normalize_config({"model": checkpoint}, {})["kv"] == "auto"
    for settings in ({"kv": "q4_0"}, {"ngl": 8}, {"fa": "off"}, {"threads": 4}, {"ram_limit_mb": 2048}):
        with pytest.raises(ValueError, match="does not apply"):
            profile.normalize_config({"model": checkpoint}, settings)


def test_checkpoint_weights_and_labels(checkpoint):
    w = profile._weights(profile.model_path(checkpoint))
    assert w == {"bytes": 4000, "params": 2000, "embedding_bytes": 0}
    base = {"model": "G-Q4_K_M.gguf", "engine": "stock", "kv": "q8_0", "ctx": 8192, "ngl": "auto", "vram_limit_mb": None, "fa": "auto", "threads": None}
    assert profile.describe_change(base, dict(base, model=checkpoint, engine="vllm")) == "hf:Org--Small · engine vllm"


def test_analysis_falls_back_to_serving_speed(checkpoint):
    res = {"serving": {"tg": 100.0, "pp": 2000.0}, "memory": {"gpu": {}, "ram": {}}}
    hw = profile.analyze(res, {"model": checkpoint, "ctx": 4096}, {"gpu_bandwidth_gbps": 0.004, "gpu_tflops": 10})
    assert hw["decode"]["roof_tps"] == pytest.approx(1000.0, rel=0.01)    # 4,000,000 bytes/s ÷ 4,000 bytes per token
    assert hw["decode"]["roof_util"] == pytest.approx(0.1, rel=0.01)       # 100 of a possible 1000 tok/s


@pytest.mark.parametrize("name,cat", [
    ("void mul_mat_q<(ggml_type)12, 64, false>(...)", "Matrix multiply (weights)"),
    ("mul_mat_vec_q<(ggml_type)8, 1>", "Matrix multiply (weights)"),
    ("void flash_attn_ext_f16<128, 128, 4>(...)", "Attention"),
    ("soft_max_f32<true, 0, 0>", "Attention"),
    ("rms_norm_f32<1024>", "Normalization"),
    ("rope_neox<true, float>", "Positional (RoPE)"),
    ("quantize_q8_1(const float*, ...)", "(De)quantization"),
    ("k_bin_bcast<&op_add, float, float, float>", "Elementwise"),
    ("cpy_f32_f16<&cpy_1_f32_f16>", "Data movement"),
    ("gated_delta_net_kernel", "Linear attention / state"),
    ("something_new", "Other"),
])
def test_kernel_categories(name, cat):
    assert kernels.categorize(name) == cat


def test_parse_nsys_kernel_summary():
    csv_text = ('"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"\n'
                '70.0,"7,000,000",320,21875,21000,20000,30000,1000,"void mul_mat_vec_q<(ggml_type)12, 1>(const void*)"\n'
                '20.0,2000000,64,31250,31000,30000,33000,500,"void flash_attn_ext_f16<128, 128, 4, 8>(const char*)"\n'
                '10.0,1000000,640,1562,1500,1400,2000,100,"rms_norm_f32<1024>"\n')
    r = kernels.parse_kern_sum(csv_text)
    assert r["total_ms"] == pytest.approx(10.0)
    assert [c["category"] for c in r["categories"]] == ["Matrix multiply (weights)", "Attention", "Normalization"]
    assert r["categories"][0]["share"] == pytest.approx(0.7) and r["categories"][0]["calls"] == 320
    assert r["top"][0]["name"].startswith("void mul_mat_vec_q") and "(const void*)" not in r["top"][0]["name"]   # signature trimmed
    assert kernels.parse_kern_sum("") == {"total_ms": 0.0, "categories": [], "top": []}


def test_kernel_profile_runs_on_the_engines_own_libraries(tmp_path, monkeypatch):
    """nsys starts llama-bench as an argument, so the runner can't tell whose libraries to load: a custom engine's
    kernel profile must not trace the stock build's GPU libraries."""
    import asyncio

    from app import quantlab
    calls = []

    async def fake_proc(cmd, log, run_id, timeout=1800, libdir=None):
        calls.append((cmd[:2], libdir))
        return ""
    monkeypatch.setattr(kernels, "available", lambda: True)
    monkeypatch.setattr(quantlab, "_proc", fake_proc)
    bench = tmp_path / "engines" / "my-engine" / "bin" / "llama-bench"
    asyncio.run(kernels.profile(bench, tmp_path / "m.gguf", [], tmp_path / "work", [], 1))
    assert calls[0] == (["nsys", "profile"], bench.parent)


def test_runner_loads_libraries_next_to_the_binary(tmp_path):
    import asyncio

    from app import quantlab
    show = ["/bin/sh", "-c", "echo $LD_LIBRARY_PATH"]
    assert asyncio.run(quantlab._proc(show, [], 0)).strip() == "/bin"                      # the binary's own folder
    assert asyncio.run(quantlab._proc(show, [], 0, libdir=tmp_path)).strip() == str(tmp_path)  # or the caller's choice


def test_sglang_is_a_checkpoint_framework(make_gguf, checkpoint, monkeypatch):
    monkeypatch.setattr(profile.engines, "sglang_installed", lambda: True)
    make_gguf("G-Q4_K_M.gguf")
    exp = {"model": "G-Q4_K_M.gguf"}
    cfg = profile.normalize_config(exp, {"model": checkpoint, "engine": "sglang"})
    assert cfg["engine"] == "sglang" and cfg["kv"] == "auto"
    with pytest.raises(ValueError, match="SGLang runs original"):
        profile.normalize_config(exp, {"engine": "sglang"})
    with pytest.raises(ValueError, match="SGLang does not apply"):
        profile.normalize_config(exp, {"model": checkpoint, "engine": "sglang", "kv": "q8_0"})
    monkeypatch.setattr(profile.engines, "sglang_installed", lambda: False)
    with pytest.raises(ValueError, match="setup-sglang"):
        profile.normalize_config(exp, {"model": checkpoint, "engine": "sglang"})


def test_multi_gpu_settings(make_gguf, monkeypatch):
    make_gguf("G-Q4_K_M.gguf")
    exp = {"model": "G-Q4_K_M.gguf"}
    monkeypatch.setattr(profile.lab, "hardware", lambda: {"gpus": [{"name": "A"}, {"name": "A"}]})
    base = profile.normalize_config(exp, {})
    assert "gpus" not in base and "split_mode" not in base          # unset settings aren't stored: older configs stay equal
    cfg = profile.normalize_config(exp, {"gpus": 2, "split_mode": "row"})
    assert cfg["gpus"] == 2 and cfg["split_mode"] == "row"
    assert profile._placement_args(cfg, 8000)[-2:] == ["-sm", "row"]
    assert profile._pin(["llama-server"], cfg) == ["env", "CUDA_VISIBLE_DEVICES=0,1", "llama-server"]
    assert profile.describe_change(base, cfg) == "2 GPUs · split row"
    for bad in ({"gpus": 3}, {"gpus": 0}, {"split_mode": "diagonal"}, {"gpus": 1, "device": "cpu"}):
        with pytest.raises(ValueError):
            profile.normalize_config(exp, bad)


def test_bandwidth_roof_adds_up_only_for_tensor_parallel():
    lim = {"gpu_count": 2, "gpu_bandwidth_gbps": 300}
    assert profile.parallel_gpus({"split_mode": "row"}, lim) == 2
    assert profile.parallel_gpus({"split_mode": "layer"}, lim) == 1        # pipeline: one GPU at a time per sequence
    assert profile.parallel_gpus({}, lim) == 1                               # llama.cpp default is layer split
    assert profile.parallel_gpus({"engine": "vllm", "gpus": 2}, lim) == 2
    assert profile.parallel_gpus({"engine": "vllm"}, lim) == 1
    assert profile.parallel_gpus({"device": "cpu", "split_mode": "row"}, lim) == 1


def test_trtllm_is_a_checkpoint_framework(make_gguf, checkpoint, monkeypatch):
    monkeypatch.setattr(profile.engines, "trtllm_installed", lambda: True)
    make_gguf("G-Q4_K_M.gguf")
    exp = {"model": "G-Q4_K_M.gguf"}
    cfg = profile.normalize_config(exp, {"model": checkpoint, "engine": "trtllm"})
    assert cfg["engine"] == "trtllm" and cfg["kv"] == "auto"
    monkeypatch.setattr(profile.engines, "trtllm_installed", lambda: False)
    with pytest.raises(ValueError, match="setup-trtllm"):
        profile.normalize_config(exp, {"model": checkpoint, "engine": "trtllm"})


def test_checkpoint_server_commands(checkpoint, tmp_path):
    cfg = {"model": checkpoint, "ctx": 4096, "gpus": 2}
    vllm, note = profile.checkpoint_server_cmd("vllm", tmp_path, cfg, 0.8)
    assert vllm[1] == "serve" and vllm[vllm.index("--tensor-parallel-size") + 1] == "2" and note is None
    assert "--pipeline-parallel-size" not in vllm
    vpp, _ = profile.checkpoint_server_cmd("vllm", tmp_path, {**cfg, "pp": 2}, 0.8)
    assert vpp[vpp.index("--pipeline-parallel-size") + 1] == "2" and vpp[vpp.index("--distributed-executor-backend") + 1] == "ray"
    sg, note = profile.checkpoint_server_cmd("sglang", tmp_path, cfg, 0.8)
    assert sg[1:3] == ["-m", "sglang.launch_server"] and sg[sg.index("--tp-size") + 1] == "2" and "chat template" in note
    trt, note = profile.checkpoint_server_cmd("trtllm", tmp_path, {**cfg, "pp": 2}, 0.95)
    assert trt[0].endswith("trtllm-serve") and trt[trt.index("--backend") + 1] == "pytorch"
    assert trt[trt.index("--tp_size") + 1] == "2" and trt[trt.index("--pp_size") + 1] == "2"
    assert float(trt[trt.index("--kv_cache_free_gpu_memory_fraction") + 1]) == 0.9
    for c in (vllm, sg, trt):   # every framework answers to the same model name the measurement client uses
        assert "blocky" in c or c is trt
        assert c[c.index("--port") + 1] == str(profile.PORT) and c[c.index("--host") + 1] == "127.0.0.1"


def test_rpc_workers_and_pipeline_stages(make_gguf, checkpoint, monkeypatch):
    make_gguf("G-Q4_K_M.gguf")
    exp = {"model": "G-Q4_K_M.gguf"}
    cfg = profile.normalize_config(exp, {"rpc": "192.168.1.40:50052, box2.lan:50052"})
    assert cfg["rpc"] == ["192.168.1.40:50052", "box2.lan:50052"]
    assert profile._placement_args(cfg, 8000)[-2:] == ["--rpc", "192.168.1.40:50052,box2.lan:50052"]
    assert profile.describe_change(profile.normalize_config(exp, {}), cfg) == "+2 RPC workers"
    assert "rpc" not in profile.normalize_config(exp, {"rpc": ""})
    for bad in ("no-port", "a:b", "x;rm -rf /:50052", ["1.2.3.4:50052"] * 17):
        with pytest.raises(ValueError):
            profile.normalize_config(exp, {"rpc": bad})
    with pytest.raises(ValueError, match="Pipeline stages apply"):
        profile.normalize_config(exp, {"pp": 2})                       # llama.cpp: use RPC workers instead
    monkeypatch.setattr(profile.engines, "vllm_installed", lambda: True)
    assert profile.normalize_config(exp, {"model": checkpoint, "engine": "vllm", "pp": 2})["pp"] == 2
    assert "pp" not in profile.normalize_config(exp, {"model": checkpoint, "engine": "vllm", "pp": 1})
    with pytest.raises(ValueError, match="does not apply"):
        profile.normalize_config(exp, {"model": checkpoint, "engine": "vllm", "rpc": "1.2.3.4:50052"})


def test_unreachable_rpc_worker_fails_fast():
    with pytest.raises(RuntimeError, match="isn't reachable"):
        profile.check_rpc_workers({"rpc": ["127.0.0.1:1"]}, timeout=0.5)


def test_distributed_runs_have_no_single_machine_roofline(checkpoint):
    res = {"weight_summary": {"bytes": 4e9, "params": 4e9, "embedding_bytes": 0}, "speed": {"tg": 30.0, "pp": 900.0}}
    lim = {"gpu_bandwidth_gbps": 300, "gpu_tflops": 20, "gpu_count": 1}
    local = profile.analyze(res, {"model": checkpoint, "ctx": 4096}, lim)
    assert local["decode"]["roof_util"] is not None
    remote = profile.analyze(res, {"model": checkpoint, "ctx": 4096, "rpc": ["10.0.0.2:50052"]}, lim)
    assert remote["decode"]["roof_util"] is None and "other machines" in remote["distributed_note"]
