"""Hardware lab, level 1: run a real model on a simulated chip and measure what the chip's arithmetic does to it
(run inside the ML venv).

Physics -> math, analog in-memory computing (one crossbar tile):
  A weight is stored as the conductance difference of a pair of devices, w ∝ G+ − G−. Inputs arrive as voltages from a
  digital-to-analog converter (DAC). Each device passes a current I = G·V (Ohm's law) and the currents on a column wire
  add up (Kirchhoff's current law), so every column computes a dot product y = Σ_i (G+_i − G−_i)·V_i in one step. An
  analog-to-digital converter (ADC) reads y with a limited range and resolution. A large matrix is split into tiles of
  `tile_rows` inputs; the tiles' outputs are added digitally.

  Normalized units: per tile, every input vector is scaled so its largest entry is 1 (the DAC's full scale), and every
  column's weights so the largest is 1 (g_max). A tile output of 1 is then "one full-scale weight times one full-scale
  input"; the ADC range is given in these units ("auto": a percentile of the outputs on calibration text, per tile).

Device imperfections (optional, each applied per device):
  programming noise   G_prog = G_target + σ_prog(G_target)·N(0,1), clipped at 0          (when the chip is programmed)
  stuck devices       a fraction stuck at 0 and at g_max                                    (from the start, never change)
  drift               G(t) = G_prog·((t + t0)/t0)^(−ν), t0 = 20 s, ν per device            (phase-change memory)
  read noise          accumulated 1/f noise at time t: G += |G|·σ_read·N(0,1)               (frozen at time t)
  output noise        additive Gaussian noise on every tile output                          (each matrix-vector product)
  drift compensation  one factor per tile, mean|w| at programming / mean|w| at time t       (IBM's global compensation)
The "pcm" models port IBM AIHWKit's PCMLikeNoiseModel (MIT licence), fitted to measurements of IBM's phase-change
memory devices (Nandakumar et al., ICECS 2019): same equations and constants.

Digital chips: weights and/or inputs in a number format (int8, int4 with optional groups, fp8 e4m3, OCP MXFP4); the
products are accumulated exactly (fp32).

What runs on the simulated chip: every linear layer inside the decoder layers (attention projections, MLP). Attention
scores (Q·Kᵀ, softmax·V), norms, embeddings and the output head stay exact digital: they multiply tensors that change
with every token, which a programmed array can't hold.

Measured on held-out text (WikiText-2 test) against the exact model: KL divergence per token, same top token,
perplexity. Each trial is a separate chip (new programming noise, drift coefficients, read noise and stuck devices), so
the spread over trials is chip-to-chip variation. Per matrix: output error of the simulated layer against the exact
layer on the same input, and the error of the effective weights.
"""
import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

T0 = 20.0                                   # s: reference time after programming (AIHWKit t_0)
T_READ = 250e-9                             # s: duration of one read (AIHWKit t_read)
PCM_PROG = (0.26348, 1.9650, -1.1731)       # µS, polynomial in G/g_max (AIHWKit prog_coeff) ...
PCM_GMAX_REF = 25.0                         # ... fitted at g_max = 25 µS
FP4_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)

NOT_MODELED = [
    "IR drop (resistance of the array wires) and sneak paths",
    "temperature: the PCM constants are for the conditions they were measured at",
    "converter noise and nonlinearity beyond quantization (only the optional output noise)",
    "how inputs are encoded in time (pulse width, bit-serial): the DAC is an ideal quantizer with per-vector range",
    "precision of the digital parts (attention, norms, accumulation of tile outputs): exact",
    "speed, energy and area: level 2 of the lab",
]


def seed_for(base: int, *parts) -> int:
    return zlib.crc32(json.dumps([base, *parts]).encode()) & 0x7FFFFFFF


def _randn(like: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(like.shape, generator=gen, device=like.device, dtype=torch.float32)


# ---------- Number formats (digital chips; also the DAC/ADC grids) ----------

def _grouped(x: torch.Tensor, group: int, fn) -> torch.Tensor:
    """fn over groups of `group` values along the last dimension (0: the whole row)."""
    k = x.shape[-1]
    if not group or group >= k:
        return fn(x)
    if k % group == 0:
        return fn(x.reshape(*x.shape[:-1], k // group, group)).reshape(x.shape)
    out = torch.empty_like(x)
    for c in range(0, k, group):
        out[..., c:c + group] = fn(x[..., c:c + group])
    return out


def q_int(x: torch.Tensor, bits: int, group: int = 0) -> torch.Tensor:
    """Symmetric integer grid, ±(2^(bits−1) − 1) steps, one scale per row or per group (the largest value maps to the top)."""
    qmax = 2 ** (bits - 1) - 1

    def f(b):
        s = b.abs().amax(-1, keepdim=True).clamp_min(1e-30) / qmax
        return (b / s).round().clamp(-qmax, qmax) * s
    return _grouped(x.float(), group, f)


def q_fp8(x: torch.Tensor, group: int = 0) -> torch.Tensor:
    """FP8 E4M3 (largest value 448), one scale per row or per group; rounding to nearest even (PyTorch's conversion)."""
    if not hasattr(torch, "float8_e4m3fn"):
        raise SystemExit("This PyTorch has no float8 support")

    def f(b):
        s = b.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 448.0
        return (b / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return _grouped(x.float(), group, f)


def q_mxfp4(x: torch.Tensor) -> torch.Tensor:
    """OCP Microscaling MXFP4: blocks of 32 values share a power-of-two scale 2^(floor(log2 amax) − 2); each value is FP4
    E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6), saturating at 6. Ties round toward zero."""
    grid = torch.tensor(FP4_E2M1, device=x.device)
    mids = (grid[1:] + grid[:-1]) / 2

    def f(b):
        amax = b.abs().amax(-1, keepdim=True)
        s = torch.exp2(torch.floor(torch.log2(amax.clamp_min(2.0 ** -126))) - 2)
        v = (b / s).clamp(-6.0, 6.0)
        return grid[torch.bucketize(v.abs(), mids)] * v.sign() * s
    return _grouped(x.float(), 32, f)


def fmt(x: torch.Tensor, name: str, group: int = 0) -> torch.Tensor:
    if name == "fp16":
        return x.float()
    if name == "int8":
        return q_int(x, 8, group)
    if name == "int4":
        return q_int(x, 4, group)
    if name == "fp8_e4m3":
        return q_fp8(x, group)
    if name == "mxfp4":
        return q_mxfp4(x)
    raise SystemExit(f"Unknown number format {name}")


def dac(xn: torch.Tensor, bits: int) -> torch.Tensor:
    """Inputs in [−1, 1] on a symmetric grid of ±(2^(bits−1) − 1) steps; 0 bits = ideal."""
    if not bits:
        return xn
    q = 2 ** (bits - 1) - 1
    return (xn * q).round() / q


def adc(yn: torch.Tensor, bits: int, rng: float | None) -> torch.Tensor:
    """Outputs clipped to ±rng and read on a symmetric grid of ±(2^(bits−1) − 1) steps; no range = no clipping."""
    if not rng:
        return yn
    yn = yn.clamp(-rng, rng)
    if not bits:
        return yn
    q = 2 ** (bits - 1) - 1
    return (yn / rng * q).round() / q * rng


# ---------- Chips ----------

class DigitalChip:
    """Weights and inputs in a number format; exact accumulation."""

    deterministic = True
    needs_calibration = False

    def __init__(self, c: dict):
        self.wf, self.wg = c["weights"]["format"], int(c["weights"].get("group") or 0)
        self.af = c["inputs"]["format"]

    def program(self, w: torch.Tensor, gen=None, ranges=None) -> dict:
        W = w.float()
        Wq = fmt(W, self.wf, self.wg)
        return {"w": Wq, "werr": (float(((Wq - W) ** 2).sum()), float((W ** 2).sum()))}

    def forward(self, x: torch.Tensor, st: dict, gen=None) -> torch.Tensor:
        return fmt(x, self.af) @ st["w"].T


class AnalogChip:
    """Crossbar tiles of device pairs, converters and device physics (see the module docstring)."""

    def __init__(self, c: dict):
        self.c = c
        self.g_max = float(c["g_max_us"])
        self.rows = int(c["tile_rows"] or 0)
        self.levels = int(c["levels"] or 0)
        self.scaling = c["weight_scaling"]
        self.prog, self.drift, self.read = c.get("prog_noise"), c.get("drift"), c.get("read_noise")
        self.t = float(c["t_seconds"])
        self.gdc = bool(c["drift_compensation"])
        self.p0, self.p1 = float(c["stuck"]["zero"]), float(c["stuck"]["max"])
        self.dac_bits, self.adc_bits = int(c["dac_bits"] or 0), int(c["adc_bits"] or 0)
        self.adc_range, self.pct = c.get("adc_range"), float(c.get("adc_percentile") or 99.99)
        self.out_noise = float(c.get("out_noise") or 0)
        if self.adc_bits and not self.adc_range:
            raise SystemExit("An ADC with bits needs a range (a number or 'auto')")

    @property
    def deterministic(self) -> bool:
        random_drift = bool(self.drift) and self.t > 0 and (self.drift["model"] == "pcm" or float(self.drift.get("nu_sd") or 0) > 0)
        return not (self.prog or self.read or self.out_noise or self.p0 or self.p1 or random_drift)

    @property
    def needs_calibration(self) -> bool:
        return self.adc_range == "auto"

    def tiles(self, k: int) -> list[tuple[int, int]]:
        step = self.rows or k
        return [(s, min(k, s + step)) for s in range(0, k, step)]

    # -- device physics (conductances in µS) --

    def sig_prog(self, g_target: torch.Tensor) -> torch.Tensor:
        if self.prog["model"] == "pcm":                      # AIHWKit apply_programming_noise_to_conductance
            r = g_target / self.g_max
            return (PCM_PROG[0] + PCM_PROG[1] * r + PCM_PROG[2] * r * r) * (self.g_max / PCM_GMAX_REF)
        return torch.full_like(g_target, float(self.prog["sigma"]) * self.g_max)

    def nu(self, g_target: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        if self.drift["model"] == "pcm":                     # AIHWKit generate_drift_coefficients
            g_rel = (g_target / self.g_max).abs().clamp(min=1e-7)
            mu = (-0.0155 * torch.log(g_rel) + 0.0244).clamp(min=0.049, max=0.1)
            sd = (-0.0125 * torch.log(g_rel) - 0.0059).clamp(min=0.008, max=0.045)
            return (mu + sd * _randn(g_target, gen)).abs()
        return (float(self.drift["nu"]) + float(self.drift.get("nu_sd") or 0) * _randn(g_target, gen)).abs()

    def sig_read(self, g_prog: torch.Tensor) -> torch.Tensor:
        if self.read["model"] == "pcm":                      # AIHWKit apply_drift_noise_to_conductance (1/f part)
            q_s = (0.0088 / ((g_prog.abs() / self.g_max) ** 0.65).clamp(min=1e-3)).clamp(max=0.2)
            return q_s * math.sqrt(math.log((self.t + T0 + T_READ) / (2 * T_READ)))
        return torch.full_like(g_prog, float(self.read["sigma"]))

    def devices(self, wn: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized weights in [−1, 1] -> (effective weights right after programming, effective weights at time t)."""
        gm = self.g_max
        prog_w, final_w = [], []
        for g_t in (wn.clamp(0, 1) * gm, (-wn).clamp(0, 1) * gm):   # one device per sign, the other at 0
            if self.levels >= 2:
                g_t = torch.round(g_t / gm * (self.levels - 1)) / (self.levels - 1) * gm
            g = g_t
            if self.prog:
                g = (g + self.sig_prog(g_t) * _randn(g, gen)).clamp_min(0.0)
            st0 = st1 = None
            if self.p0 or self.p1:
                u = torch.rand(g.shape, generator=gen, device=g.device)
                st0, st1 = u < self.p0, (u >= self.p0) & (u < self.p0 + self.p1)
                g = torch.where(st0, torch.zeros_like(g), torch.where(st1, torch.full_like(g, gm), g))
            g_prog = g
            if self.drift and self.t > 0:
                g = g * ((self.t + T0) / T0) ** (-self.nu(g_t, gen))
            if self.read:
                g = (g + g.abs() * self.sig_read(g_prog) * _randn(g, gen)).clamp_min(0.0)
            if st0 is not None:
                g = torch.where(st0, torch.zeros_like(g), torch.where(st1, torch.full_like(g, gm), g))
            prog_w.append(g_prog)
            final_w.append(g)
        return (prog_w[0] - prog_w[1]) / gm, (final_w[0] - final_w[1]) / gm

    def normalized(self, W: torch.Tensor, s: int, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        Wt = W[:, s:e]
        sc = Wt.abs().amax(1, keepdim=True) if self.scaling == "column" else Wt.abs().amax().reshape(1, 1)
        sc = sc.clamp_min(1e-30)
        return Wt / sc, sc

    def program(self, w: torch.Tensor, gen: torch.Generator, ranges: list[float] | None = None) -> dict:
        W = w.float()
        st = {"tiles": self.tiles(W.shape[1]), "w": [], "scale": [], "alpha": [], "ranges": ranges}
        werr = [0.0, 0.0]
        for s, e in st["tiles"]:
            wn, sc = self.normalized(W, s, e)
            w0, wt = self.devices(wn, gen)
            alpha = 1.0
            if self.gdc and ((self.drift and self.t > 0) or self.read):
                alpha = float(w0.abs().mean().clamp_min(1e-4) / wt.abs().mean().clamp_min(1e-4))
            st["w"].append(wt)
            st["scale"].append(sc.reshape(-1))
            st["alpha"].append(alpha)
            werr[0] += float(((wt * sc * alpha - W[:, s:e]) ** 2).sum())
            werr[1] += float((W[:, s:e] ** 2).sum())
        st["werr"] = tuple(werr)
        return st

    def tile_inputs(self, x: torch.Tensor, s: int, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        xt = x[:, s:e]
        m = xt.abs().amax(-1, keepdim=True).clamp_min(1e-30)    # per input vector and tile: the DAC's full scale
        return dac(xt / m, self.dac_bits), m

    def forward(self, x: torch.Tensor, st: dict, gen: torch.Generator | None = None) -> torch.Tensor:
        y = None
        for i, (s, e) in enumerate(st["tiles"]):
            xn, m = self.tile_inputs(x, s, e)
            yn = xn @ st["w"][i].T
            if self.out_noise:
                yn = yn + self.out_noise * _randn(yn, gen)
            rng = st["ranges"][i] if self.adc_range == "auto" else self.adc_range
            yn = adc(yn, self.adc_bits, rng)
            part = yn * st["scale"][i] * m * st["alpha"][i]
            y = part if y is None else y + part
        return y

    def observe(self, x: torch.Tensor, W: torch.Tensor, store: list, gen: torch.Generator, keep: int = 200_000) -> None:
        """Calibration: the ideal tile outputs an ADC would see (exact weights, quantized inputs), sampled per tile."""
        for i, (s, e) in enumerate(self.tiles(W.shape[1])):
            wn, _ = self.normalized(W, s, e)
            xn, _ = self.tile_inputs(x, s, e)
            v = (xn @ wn.T).abs().flatten()
            if v.numel() > keep:
                v = v[torch.randint(0, v.numel(), (keep,), generator=gen, device=v.device)]
            if len(store) <= i:
                store.append([])
            store[i].append(v.float().cpu())

    def ranges_from(self, store: list) -> list[float]:
        return [max(1e-6, float(torch.quantile(torch.cat(v), self.pct / 100))) for v in store]


def make_chip(c: dict):
    if c.get("kind") == "analog":
        return AnalogChip(c)
    if c.get("kind") == "digital":
        return DigitalChip(c)
    raise SystemExit("chip kind must be 'analog' or 'digital'")


# ---------- Wrappers that put a linear layer on the chip ----------

class OnChip(torch.nn.Module):
    """A linear layer computed by the chip; also measures its output error against the exact layer on the same input."""

    def __init__(self, base: torch.nn.Linear, chip, st: dict, gen, err: list):
        super().__init__()
        self.base, self.chip, self.st, self.gen, self.err = base, chip, st, gen, err

    def forward(self, x):
        xf = x.float()
        shp = xf.shape
        x2 = xf.reshape(-1, shp[-1])
        bias = None if self.base.bias is None else self.base.bias.float()
        y = self.chip.forward(x2, self.st, self.gen)
        if bias is not None:
            y = y + bias
        ye = F.linear(x2, self.base.weight.float(), bias)
        self.err[0] += float(((y - ye) ** 2).sum())
        self.err[1] += float((ye ** 2).sum())
        return y.reshape(*shp[:-1], -1).to(x.dtype)


class RangeProbe(torch.nn.Module):
    """Calibration pass: exact outputs, while recording what each tile's ADC would see."""

    def __init__(self, base: torch.nn.Linear, chip, gen):
        super().__init__()
        self.base, self.chip, self.gen, self.store = base, chip, gen, []

    def forward(self, x):
        self.chip.observe(x.float().reshape(-1, x.shape[-1]), self.base.weight.float(), self.store, self.gen)
        return self.base(x)


def _swap(layer: torch.nn.Module, mods: dict) -> None:
    for name, m in mods.items():
        *path, leaf = name.split(".")
        setattr(layer.get_submodule(".".join(path)) if path else layer, leaf, m)


def summarize(vals: list[float]) -> dict:
    n = len(vals)
    mean = sum(vals) / n
    out = {"mean": mean, "min": min(vals), "max": max(vals), "n": n}
    if n > 1:
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        out.update(sd=sd, ci95=(T975[n - 2] if n - 2 < len(T975) else 1.96) * sd / math.sqrt(n))
    return out


def _rel(e) -> float | None:
    return math.sqrt(e[0] / e[1]) if e[1] else None


# ---------- The model, one decoder layer at a time ----------

def run(a, chip_cfg: dict, dev: str) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import quantsim as qs
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    chip = make_chip(chip_cfg)
    tok = AutoTokenizer.from_pretrained(a.model)
    t_dtype = torch.float16 if dev == "cuda" else torch.float32
    cfg = AutoConfig.from_pretrained(a.model)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=t_dtype)
    model.eval()
    ck = qs.Checkpoint(a.model)
    keep32 = tuple(getattr(model, "_keep_in_fp32_modules", None) or ())
    parent, _, layers = qs._find_layers(model)
    base = parent
    prefix_of = {id(m): n for n, m in model.named_modules()}
    layer_prefix = prefix_of[id(layers)] + "."
    n_layers = len(layers)
    print(f"loading layer by layer ({n_layers} layers) on {dev}", flush=True)

    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    norm = getattr(base, "norm", None) or getattr(base, "final_layernorm", None)
    qs._materialize(emb, prefix_of[id(emb)] + ".", ck, dev, t_dtype)
    if norm is not None:
        qs._materialize(norm, prefix_of[id(norm)] + ".", ck, dev, t_dtype, keep32)
    for _, m in base.named_children():
        if m is not layers and m is not emb and m is not norm:
            qs._materialize(m, prefix_of[id(m)] + ".", ck, dev, t_dtype, keep32)

    calib = qs.chunks_of(tok, a.calib, a.calib_chunks, a.ctx) if chip.needs_calibration else []
    evals = qs.chunks_of(tok, a.text, a.chunks, a.ctx)
    if not evals:
        raise SystemExit("The evaluation text is shorter than one chunk")
    trials = 1 if chip.deterministic else a.trials

    store: dict = {}
    real = list(layers)
    for i in range(n_layers):
        layers[i] = qs.Catcher(i, store)
    with torch.no_grad():
        for c in calib + evals:
            base(input_ids=c.unsqueeze(0).to(dev), use_cache=False)
    for i in range(n_layers):
        layers[i] = real[i]
    kws = [store[i]["kw"] for i in range(n_layers)]
    h0 = store[0]["h"]
    h_cal, h_ref = h0[:len(calib)], h0[len(calib):]
    h_chip = [list(h_ref) for _ in range(trials)]
    del store
    qs._release(emb)
    if dev == "cuda":
        torch.cuda.empty_cache()

    out_err = [{} for _ in range(trials)]              # matrix name -> [Σ(y − y_exact)², Σ y_exact²]
    w_err = [{} for _ in range(trials)]
    n_mats = n_weights = 0
    with torch.no_grad():
        for i, layer in enumerate(real):
            lp = f"{layer_prefix}{i}."
            qs._materialize(layer, lp, ck, dev, t_dtype, keep32)
            kw = qs._to(kws[i], dev)
            lins = {n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}
            n_mats += len(lins)
            n_weights += sum(m.weight.numel() for m in lins.values())
            ranges = {}
            if calib:
                gen = torch.Generator(device=dev).manual_seed(seed_for(a.seed, "calibration", i))
                probes = {n: RangeProbe(m, chip, gen) for n, m in lins.items()}
                _swap(layer, probes)
                try:
                    h_cal = [layer(h, **kw) for h in h_cal]
                finally:
                    _swap(layer, lins)
                ranges = {n: chip.ranges_from(p.store) for n, p in probes.items()}
                del probes
            h_ref = [layer(h, **kw) for h in h_ref]
            for t in range(trials):
                sims = {}
                for n, m in lins.items():
                    full = lp + n
                    gen = torch.Generator(device=dev).manual_seed(seed_for(a.seed, t, full))
                    st = chip.program(m.weight.data, gen, ranges.get(n))
                    w_err[t][full] = list(st.pop("werr"))
                    sims[n] = OnChip(m, chip, st, gen, out_err[t].setdefault(full, [0.0, 0.0]))
                _swap(layer, sims)
                try:
                    h_chip[t] = [layer(h, **kw) for h in h_chip[t]]
                finally:
                    _swap(layer, lins)
                del sims
            qs._release(layer)
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"layer {i + 1}/{n_layers}", flush=True)

    if head is not None and next(head.parameters()).device.type == "meta":
        try:
            qs._materialize(head, prefix_of[id(head)] + ".", ck, dev, t_dtype)
        except KeyError:                                   # tied to the embeddings
            head._parameters["weight"] = torch.nn.Parameter(ck.get(prefix_of[id(emb)] + ".weight").to(dev, t_dtype), requires_grad=False)
    per_trial = []
    with torch.no_grad():
        refs = [norm(h) if norm is not None else h for h in h_ref]
        for t in range(trials):
            sc = qs.Scorer()
            for c, hb, hq in zip(evals, refs, h_chip[t]):
                if norm is not None:
                    hq = norm(hq)
                sc.add(head(hb)[0, :-1], head(hq)[0, :-1], c[1:].to(dev))
            per_trial.append({"trial": t + 1, **sc.result()})
            print(f"scored chip {t + 1}/{trials}", flush=True)

    def pooled(maps, key=None):
        tot = {}
        for mp in maps:
            for name, e in mp.items():
                k = key(name) if key else "all"
                b = tot.setdefault(k, [0.0, 0.0])
                b[0] += e[0]
                b[1] += e[1]
        return tot

    kind = lambda n: n.split(".")[-1]  # noqa: E731
    o_kind, w_kind = pooled(out_err, kind), pooled(w_err, kind)
    o_layer, w_layer = pooled(out_err, qs.layer_of), pooled(w_err, qs.layer_of)
    summary = {k: summarize([r[k] for r in per_trial]) for k in ("kld", "same_top", "ppl")}
    return {
        "chip": chip_cfg, "trials": trials, "deterministic": chip.deterministic, "seed": a.seed,
        "per_trial": per_trial, "summary": summary, "ppl_base": per_trial[0]["ppl_base"], "tokens": per_trial[0]["tokens"],
        "output_error": _rel(pooled(out_err)["all"]), "weight_error": _rel(pooled(w_err)["all"]),
        "per_kind": {k: {"output_error": _rel(o_kind[k]), "weight_error": _rel(w_kind[k])} for k in sorted(o_kind)},
        "per_layer": [{"layer": L, "output_error": _rel(o_layer[L]), "weight_error": _rel(w_layer[L])}
                      for L in sorted(o_layer, key=lambda v: (v is None, v or 0))],
        "on_chip": {"matrices": n_mats, "weights": n_weights},
        "digital": "attention scores, norms, embeddings and the output head",
        "calibration": {"chunks": len(calib), "tokens": len(calib) * a.ctx} if calib else None,
        "not_modeled": NOT_MODELED if chip_cfg.get("kind") == "analog" else ["speed, energy and area: level 2 of the lab"],
        "mode": "layer by layer", "device": dev,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--calib", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chip", required=True, help="chip description (JSON file)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--calib_chunks", type=int, default=4)
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    chip_cfg = json.loads(Path(a.chip).read_text())
    if chip_cfg.get("adc_range") == "auto" and not a.calib:
        raise SystemExit("An 'auto' ADC range needs calibration text (--calib)")
    t0 = time.time()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = run(a, chip_cfg, dev)
    res["seconds"] = round(time.time() - t0, 1)
    if dev == "cuda":
        res["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    Path(a.out).write_text(json.dumps(res))
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
