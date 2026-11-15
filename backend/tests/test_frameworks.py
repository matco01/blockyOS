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
