"""Hardware lab simulator (torch_tools/hwsim.py): the chip's physics and math against values worked out by hand or
taken from IBM AIHWKit's PCM model, plus one end-to-end run on a tiny model. Needs PyTorch (CI: "PyTorch tools")."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
TOOLS = Path(__file__).resolve().parents[1] / "torch_tools"
sys.path.insert(0, str(TOOLS))
import hwsim  # noqa: E402

ANALOG = {"kind": "analog", "tile_rows": 4, "weight_scaling": "column", "g_max_us": 25.0, "levels": 0, "prog_noise": None,
          "drift": None, "read_noise": None, "t_seconds": 0.0, "drift_compensation": False, "stuck": {"zero": 0.0, "max": 0.0},
          "dac_bits": 0, "adc_bits": 0, "adc_range": None, "out_noise": 0.0}
HALF_NORMAL_VAR = 0.5 - 1 / (2 * math.pi)      # variance of max(0, N(0, 1))
SIG0 = 0.26348                                  # PCM programming noise at G = 0 (µS)


def chip(**over):
    return hwsim.AnalogChip({**ANALOG, **over})


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_an_ideal_analog_chip_computes_the_exact_product():
    W, x = torch.randn(6, 10, generator=gen(1)), torch.randn(3, 10, generator=gen(2))
    for scaling in ("column", "tile"):
        c = chip(weight_scaling=scaling)
        st = c.program(W, gen())
        assert torch.allclose(c.forward(x, st), x @ W.T, atol=1e-5)
        assert st["werr"][0] < 1e-10


def test_converters():
    xn = torch.tensor([-1.0, -0.51, 0.0, 0.26, 0.74, 1.0])
    assert torch.allclose(hwsim.dac(xn, 3), torch.tensor([-1.0, -2 / 3, 0.0, 1 / 3, 2 / 3, 1.0]))     # 3 bits: ±3 steps
    yn = torch.tensor([-5.0, -0.3, 0.2, 0.55, 3.0])
    assert torch.allclose(hwsim.adc(yn, 3, 1.0), torch.tensor([-1.0, -1 / 3, 1 / 3, 2 / 3, 1.0]))      # clipped, then read
    assert torch.equal(hwsim.adc(yn, 0, None), yn) and torch.equal(hwsim.dac(xn, 0), xn)


def test_pcm_programming_noise_has_the_measured_spread():
    c = chip(prog_noise={"model": "pcm"})
    for r in (1.0, 0.5):
        sig = SIG0 + 1.9650 * r - 1.1731 * r * r          # AIHWKit: polynomial in G/g_max, µS at g_max = 25 µS
        assert float(c.sig_prog(torch.tensor([r * 25.0]))) == pytest.approx(sig)
        w0, _ = c.devices(torch.full((1, 400_000), r), gen(3))
        g = w0 * 25.0
        # the positive device spreads by sig; the negative one (target 0) is clipped at 0: a half-normal of SIG0
        assert float(g.std()) == pytest.approx(math.sqrt(sig ** 2 + SIG0 ** 2 * HALF_NORMAL_VAR), rel=0.01)
        assert float(g.mean()) == pytest.approx(r * 25.0 - SIG0 / math.sqrt(2 * math.pi), abs=0.01)


def test_drift_follows_the_power_law_and_compensation_undoes_it():
    drift = {"model": "constant", "nu": 0.05, "nu_sd": 0.0}
    wn = torch.tensor([[1.0, 0.5, -0.25]])
    _, wt = chip(drift=drift, t_seconds=3600.0).devices(wn, gen())
    assert torch.allclose(wt, wn * ((3600 + 20) / 20) ** -0.05, rtol=1e-6)         # G(t) = G0 ((t + t0)/t0)^-ν
    W, x = torch.randn(8, 16, generator=gen(7)), torch.randn(5, 16, generator=gen(8))
    drift = {"model": "constant", "nu": 0.1, "nu_sd": 0.0}
    off, on = chip(drift=drift, t_seconds=86400.0), chip(drift=drift, t_seconds=86400.0, drift_compensation=True)
    assert torch.allclose(off.forward(x, off.program(W, gen())), ((86400 + 20) / 20) ** -0.1 * (x @ W.T), atol=1e-4)
    assert torch.allclose(on.forward(x, on.program(W, gen())), x @ W.T, atol=1e-4)


def test_pcm_drift_coefficients():
    c = chip(drift={"model": "pcm"}, t_seconds=1.0)
    nu = c.nu(torch.full((400_000,), 25.0), gen(4))      # at g_max both are clipped: mean 0.049, spread 0.008
    assert float(nu.mean()) == pytest.approx(0.049, abs=5e-4) and float(nu.std()) == pytest.approx(0.008, rel=0.03)
    nu = c.nu(torch.full((400_000,), 2.5), gen(5))       # at 0.1 g_max: mean −0.0155 ln 0.1 + 0.0244, spread −0.0125 ln 0.1 − 0.0059
    assert float(nu.mean()) == pytest.approx(0.0155 * math.log(10) + 0.0244, abs=1e-3)
    assert float(nu.std()) == pytest.approx(0.0125 * math.log(10) - 0.0059, rel=0.03)


def test_pcm_read_noise_at_the_reference_time():
    c = chip(read_noise={"model": "pcm"})
    expect = 0.0088 * math.sqrt(math.log((20 + 250e-9) / (2 * 250e-9)))   # ≈ 3.7 % of G at g_max, 20 s after programming
    assert float(c.sig_read(torch.tensor([25.0]))) == pytest.approx(expect)
    _, wt = c.devices(torch.ones(1, 400_000), gen(6))
    assert float(wt.std()) == pytest.approx(expect, rel=0.02) and float(wt.mean()) == pytest.approx(1.0, abs=1e-3)


def test_stuck_devices_stay_stuck():
    c = chip(stuck={"zero": 0.05, "max": 0.02}, drift={"model": "constant", "nu": 0.1, "nu_sd": 0.0}, t_seconds=86400.0)
    w0, wt = c.devices(torch.full((1, 400_000), 0.5), gen(9))
    frac = lambda w, v: float((w - v).abs().lt(1e-6).float().mean())  # noqa: E731
    # weight = G+ − G− (in g_max): G+ stuck at max with a working G− -> 1; G− stuck at max with a working G+ -> −0.5 ...
    assert frac(w0, 1.0) == pytest.approx(0.02 * 0.98, abs=0.002) and frac(w0, -0.5) == pytest.approx(0.93 * 0.02, abs=0.002)
    assert frac(w0, 0.0) == pytest.approx(0.05 * 0.98 + 0.02 * 0.02, abs=0.002)
    assert frac(wt, 1.0) == pytest.approx(frac(w0, 1.0))       # stuck devices don't drift; working ones do
    assert float(wt.max()) == 1.0 and frac(wt, 0.5) == 0.0


def test_device_levels():
    w0, _ = chip(levels=5).devices(torch.tensor([[0.1, 0.3, -0.6, 0.9]]), gen())
    assert torch.allclose(w0, torch.tensor([[0.0, 0.25, -0.5, 1.0]]))


def test_number_formats():
    x = torch.zeros(1, 32)
    x[0, :6] = torch.tensor([0.2, 0.3, 2.4, 2.6, 5.2, 6.0])
    assert hwsim.q_mxfp4(x)[0, :6].tolist() == [0.0, 0.5, 2.0, 3.0, 6.0, 6.0]       # scale 2^(floor(log2 6) − 2) = 1
    x[0, :2] = torch.tensor([12.0, 1.0])
    assert hwsim.q_mxfp4(x)[0, :2].tolist() == [12.0, 1.0]                            # scale 2
    x[0, :2] = torch.tensor([7.0, 1.0])
    assert hwsim.q_mxfp4(x)[0, 0].item() == 6.0                                       # MX saturates (7 / 1 > 6)
    q = hwsim.q_int(torch.tensor([[1.0, -0.55, 0.25, 0.1]]), 4)                       # ±7 steps of 1/7 (no exact ties)
    assert torch.allclose(q, torch.tensor([[1.0, -4 / 7, 2 / 7, 1 / 7]]))
    v = torch.tensor([[448.0, 0.3, -17.0, 100.0]])
    assert torch.equal(hwsim.q_fp8(v), v.to(torch.float8_e4m3fn).float())             # scale 1: plain E4M3 rounding
    v = torch.tensor([[1.0, -1.0, 4.0, -4.0]])
    assert torch.allclose(hwsim.q_int(v, 4, group=2), v)            # one scale per group of 2: both groups exact
    assert hwsim.q_int(v, 4)[0, 0].item() == pytest.approx(8 / 7)   # one scale for the row (4/7): 1.0 lands on 2 steps


def test_each_trial_is_a_different_chip_but_reproducible():
    c = chip(prog_noise={"model": "pcm"})
    W = torch.randn(4, 8, generator=gen(10))
    one = c.program(W, gen(hwsim.seed_for(0, 0, "x")))["w"][0]
    assert torch.equal(one, c.program(W, gen(hwsim.seed_for(0, 0, "x")))["w"][0])
    assert not torch.equal(one, c.program(W, gen(hwsim.seed_for(0, 1, "x")))["w"][0])
    assert c.deterministic is False and chip().deterministic is True
    assert chip(drift={"model": "constant", "nu": 0.05, "nu_sd": 0.0}, t_seconds=10.0).deterministic is True


def test_output_error_grows_in_proportion_to_device_noise():
    W, x = torch.randn(64, 256, generator=gen(11)), torch.randn(32, 256, generator=gen(12))

    def err(sigma):
        c = chip(tile_rows=64, prog_noise={"model": "constant", "sigma": sigma})
        tot = sum(float(((c.forward(x, c.program(W, gen(100 + s))) - x @ W.T) ** 2).sum()) for s in range(20))
        return math.sqrt(tot)
    assert err(0.02) / err(0.01) == pytest.approx(2.0, rel=0.1)


def test_adc_range_from_calibration():
    c = chip(adc_bits=4, adc_range="auto", adc_percentile=100.0)
    W, x = torch.randn(5, 8, generator=gen(13)), torch.randn(50, 8, generator=gen(14))
    store = []
    c.observe(x, W, store, gen())
    ranges = c.ranges_from(store)
    assert len(ranges) == 2                                                          # two tiles of 4 rows
    wn, _ = c.normalized(W, 0, 4)
    xn, _ = c.tile_inputs(x, 0, 4)
    assert ranges[0] == pytest.approx(float((xn @ wn.T).abs().max()))                # 100th percentile = the largest output


# ---------- End to end on a tiny model (needs transformers) ----------

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    d = tmp_path_factory.mktemp("hw")
    words = "the a chip weight noise drift model layer memory column current voltage device phase change read write".split()
    vocab = {"<unk>": 0, **{w: i + 1 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>").save_pretrained(d / "m")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256, initializer_range=0.2)
    LlamaForCausalLM(cfg).save_pretrained(d / "m")
    g = torch.Generator().manual_seed(1)
    for name in ("test", "train"):
        (d / f"{name}.txt").write_text(" ".join(words[i] for i in torch.randint(0, len(words), (400,), generator=g).tolist()))
    return d


def simulate(d, chip_cfg, trials=2):
    (d / "chip.json").write_text(json.dumps(chip_cfg))
    out = d / "out.json"
    subprocess.run([sys.executable, str(TOOLS / "hwsim.py"), "--model", str(d / "m"), "--text", str(d / "test.txt"),
                    "--calib", str(d / "train.txt"), "--out", str(out), "--chip", str(d / "chip.json"), "--trials", str(trials),
                    "--chunks", "3", "--ctx", "64", "--calib_chunks", "2", "--device", "cpu"], check=True, capture_output=True, timeout=600)
    return json.loads(out.read_text())


def test_end_to_end(tiny):
    full = {**ANALOG, "tile_rows": 32, "dac_bits": 0}
    exact = simulate(tiny, {**full, "adc_range": None}, trials=3)
    assert exact["trials"] == 1 and exact["deterministic"]                         # nothing random: one chip is enough
    assert exact["summary"]["kld"]["mean"] < 1e-8 and exact["summary"]["same_top"]["mean"] == 100.0
    assert exact["output_error"] < 1e-5 and exact["on_chip"]["matrices"] == 14     # 7 matrices per layer, 2 layers
    pcm = simulate(tiny, {**full, "prog_noise": {"model": "pcm"}, "drift": {"model": "pcm"}, "read_noise": {"model": "pcm"},
                          "t_seconds": 3600.0, "drift_compensation": True, "dac_bits": 8, "adc_bits": 8, "adc_range": "auto"})
    assert pcm["trials"] == 2 and len(pcm["per_trial"]) == 2 and "sd" in pcm["summary"]["kld"]
    assert pcm["per_trial"][0]["kld"] != pcm["per_trial"][1]["kld"]               # two chips, two results
    assert 0.01 < pcm["output_error"] < 1 and pcm["summary"]["kld"]["mean"] > 1e3 * max(exact["summary"]["kld"]["mean"], 1e-12)
    assert pcm["calibration"]["chunks"] == 2 and set(pcm["per_kind"]) >= {"q_proj", "down_proj"}
    int4 = simulate(tiny, {"kind": "digital", "weights": {"format": "int4", "group": 32}, "inputs": {"format": "fp16"}})
    assert int4["trials"] == 1 and int4["weight_error"] > 0.01 and int4["calibration"] is None
