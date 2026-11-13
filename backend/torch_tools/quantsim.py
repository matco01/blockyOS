"""Technique playground: simulate a weight / KV-cache quantization method on a Hugging Face checkpoint and measure what
it does to the model, before anyone writes a kernel for it (run inside the ML venv).

"Fake quantization": weights are quantized and immediately dequantized back to fp16, so the model computes with exactly
the values the method would store. Quality effects are real; speed and memory are not measured here (export the
result to a real GGUF and profile it for that).

Calibration (--calib): one pass over calibration text (WikiText-2 *train*, never the evaluation text) collects, for
every linear layer, H = Σ xᵀx over its inputs. It powers the calibration-based methods and an output-error metric:
  output error of a layer = tr(ΔW H ΔWᵀ) / tr(W H Wᵀ)   (how much the layer's outputs change on real inputs)

Methods (weights of the decoder's linear layers; embeddings and the output head stay fp16 unless --include_head):
  rtn       round-to-nearest, `bits`, group-wise scales, symmetric or asymmetric
  gptq      GPTQ (Frantar et al., 2022): quantize column by column and push each column's error onto the columns not yet
            quantized, through the inverse of H (block size 128, 1% damping). Inputs come from the full-precision model
            (non-sequential variant).
  awq       AWQ-style (Lin et al., 2023): per input channel scale s = (activation magnitude)^α, quantize W·s, divide by s;
            α is chosen per layer from a grid to minimise the output error. (The original also searches weight clipping.)
  nf4       4-bit NormalFloat code book (QLoRA), absmax scale per group
  ternary   {-1, 0, +1} × absmean scale per group (BitNet b1.58 style)
  custom    your own Python: quantize(w, name) -> dequantized w, or quantize(w, name, h) to also get H (mean xᵀx)
Grid (--grid): "int" = the bits/group/sym settings above; "q4_0" = exactly llama.cpp's Q4_0 grid (4-bit, groups of 32,
scale = signed max / -8), so a Q4_0 GGUF export reproduces the simulated values up to fp16 rounding (rtn, gptq, awq).
KV: --kv_bits N fake-quantizes the key/value projection outputs per token (asymmetric, groups of 32), before rotary
embeddings: an approximation of a quantized KV cache.

Measured on held-out text (WikiText-2 test) against the unmodified checkpoint: KL divergence per token (mean ± SE, p99),
same-top-token %, perplexity of both; per-layer weight and output error; effective bits per weight.
--export DIR saves the quantized checkpoint (for conversion to GGUF) and --no_eval skips the evaluation.
"""
import argparse
import importlib.util
import inspect
import json
import math
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])


# ---------- Grids: per-row parameters for a group of columns, and quantize-dequantize with them ----------

class Grid:
    def __init__(self, kind: str, bits: int, sym: bool):
        self.kind, self.bits, self.sym = kind, bits, sym

    def fit(self, x: torch.Tensor):
        """x: [rows, g] -> parameters per row."""
        if self.kind == "q4_0":
            i = x.abs().argmax(1, keepdim=True)
            d = x.gather(1, i) / -8.0                      # llama.cpp: d = max / -8 (max = the value with the largest |x|)
            return (d,)
        if self.sym:
            qmax = 2 ** (self.bits - 1) - 1
            return (x.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax,)
        lo, hi = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
        return ((hi - lo).clamp_min(1e-8) / (2 ** self.bits - 1), lo)

    def qdq(self, x: torch.Tensor, p) -> torch.Tensor:
        if self.kind == "q4_0":
            d = p[0]
            inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
            return (x * inv).round().clamp(-8, 7) * d
        if self.sym:
            qmax = 2 ** (self.bits - 1) - 1
            return (x / p[0]).round().clamp(-qmax - 1, qmax) * p[0]
        s, lo = p
        return ((x - lo) / s).round().clamp(0, 2 ** self.bits - 1) * s + lo

    def bpw(self, group: int) -> float:
        if self.kind == "q4_0":
            return 4.5
        return self.bits + 16 / group * (1 if self.sym else 2)


def grouped(w: torch.Tensor, g: int, fn) -> torch.Tensor:
    """Apply fn to [rows, g] column groups of w (the last group may be shorter)."""
    out = torch.empty_like(w)
    for c in range(0, w.shape[1], g):
        out[:, c:c + g] = fn(w[:, c:c + g])
    return out


def rtn(w, grid: Grid, g):
    x = w.float()
    return grouped(x, g, lambda b: grid.qdq(b, grid.fit(b))).to(w.dtype)


def nf4(w, g):
    code = NF4.to(w.device)
    def f(b):
        s = b.abs().amax(1, keepdim=True).clamp_min(1e-8)
        return code[((b / s).unsqueeze(-1) - code).abs().argmin(-1)] * s
    return grouped(w.float(), g, f).to(w.dtype)


def ternary(w, g):
    def f(b):
        s = b.abs().mean(1, keepdim=True).clamp_min(1e-8)
        return (b / s).round().clamp(-1, 1) * s
    return grouped(w.float(), g, f).to(w.dtype)


def gptq(w, H, grid: Grid, g, block=128, damp=0.01):
    W = w.float().clone()
    cols = W.shape[1]
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=H.device)
    try:
        L = torch.linalg.cholesky(H)
    except RuntimeError:                                   # badly conditioned: more damping
        H += 0.1 * torch.mean(torch.diag(H)) * torch.eye(cols, device=H.device)
        L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    block = max(block, g) // g * g                         # groups never straddle blocks
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1, Q1, E1, Hi = W[:, i1:i2].clone(), torch.zeros_like(W[:, i1:i2]), torch.zeros_like(W[:, i1:i2]), Hinv[i1:i2, i1:i2]
        p = None
        for i in range(i2 - i1):
            if (i1 + i) % g == 0:
                p = grid.fit(W1[:, i:i + g])               # scales from the already error-corrected weights of this group
            q = grid.qdq(W1[:, i:i + 1], p)[:, 0]
            Q1[:, i] = q
            err = (W1[:, i] - q) / Hi[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hi[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q.to(w.dtype)


def out_err(w, q, H) -> tuple[float, float]:
    D = (w.float() - q.float())
    return float(((D @ H) * D).sum()), float(((w.float() @ H) * w.float()).sum())


def awq(w, H, grid: Grid, g):
    xs = torch.diag(H).clamp_min(1e-10).sqrt()            # activation magnitude per input channel
    best, best_e = None, None
    for a in [i / 10 for i in range(11)]:
        s = xs ** a
        s = s / (s.max() * s.min()).sqrt()
        q = rtn(w.float() * s, grid, g).float() / s
        e = out_err(w, q, H)[0]
        if best_e is None or e < best_e:
            best, best_e = q, e
    return best.to(w.dtype)


# ---------- Model plumbing ----------

def decoder_linears(model, include_head: bool):
    head = model.get_output_embeddings()
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and (include_head or m is not head) and ".layers." in f".{name}.".replace(".h.", ".layers."):
            yield name, m
    if include_head and head is not None:
        yield "lm_head", head


def layer_of(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts[:-1]):
        if p in ("layers", "h") and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def kv_hooks(model, bits: int):
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and name.split(".")[-1] in ("k_proj", "v_proj"):
            def hook(mod, inp, out):
                x = out.float()
                d = x.shape[-1]
                g = 32 if d % 32 == 0 else d
                xs = x.reshape(*x.shape[:-1], d // g, g)
                lo, hi = xs.amin(-1, keepdim=True), xs.amax(-1, keepdim=True)
                s = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
                q = ((xs - lo) / s).round().clamp(0, 2 ** bits - 1) * s + lo
                return q.reshape(x.shape).to(out.dtype)
            handles.append(m.register_forward_hook(hook))
    return handles


def calibrate(model, mods: list, chunks: list, dev: str) -> dict:
    """Mean xᵀx of each module's inputs over the calibration chunks (fp32, on the GPU)."""
    H, n = {}, {}
    def mk(name):
        def hook(m, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            if name not in H:
                H[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                n[name] = 0
            H[name].addmm_(x.T, x)
            n[name] += x.shape[0]
        return hook
    hs = [m.register_forward_hook(mk(name)) for name, m in mods]
    try:
        with torch.no_grad():
            for c in chunks:
                model(c.unsqueeze(0).to(dev))
    finally:
        for h in hs:
            h.remove()
    return {k: v / max(1, n[k]) for k, v in H.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--calib", default="")
    ap.add_argument("--calib_chunks", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["rtn", "gptq", "awq", "nf4", "ternary", "custom", "none"], default="rtn")
    ap.add_argument("--grid", choices=["int", "q4_0"], default="int")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--custom", default="")
    ap.add_argument("--include_head", action="store_true")
    ap.add_argument("--kv_bits", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--export", default="")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--h_budget_mb", type=int, default=2048)
    a = ap.parse_args()
    t0 = time.time()
    if a.grid == "q4_0":
        a.bits, a.group, a.sym = 4, 32, True
    grid = Grid(a.grid, a.bits, a.sym)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    print(f"loaded on {dev}", flush=True)

    custom, custom_h = None, False
    if a.method == "custom":
        spec = importlib.util.spec_from_file_location("blocky_custom_quant", a.custom)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        custom = mod.quantize
        custom_h = len(inspect.signature(custom).parameters) >= 3

    linears = list(decoder_linears(model, a.include_head)) if a.method != "none" else []
    calib = []
    if a.calib and linears:
        cids = tok(open(a.calib, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
        calib = [cids[i * a.ctx:(i + 1) * a.ctx] for i in range(min(a.calib_chunks, cids.numel() // a.ctx))]

    # Quantize in groups of layers whose statistics fit the memory budget (one calibration pass per group).
    orig, quant, per_layer, per_kind = {}, {}, {}, {}
    tot = {"w": [0.0, 0.0], "o": [0.0, 0.0]}
    n_params = 0
    batch, batch_bytes, batches = [], 0, []
    for name, m in linears:
        b = m.in_features ** 2 * 4 if calib else 0
        if batch and batch_bytes + b > a.h_budget_mb * 2 ** 20:
            batches.append(batch); batch, batch_bytes = [], 0
        batch.append((name, m)); batch_bytes += b
    if batch:
        batches.append(batch)
    with torch.no_grad():
        for bi, group in enumerate(batches):
            Hs = calibrate(model, group, calib, dev) if calib else {}
            if calib:
                print(f"calibrated {bi + 1}/{len(batches)} ({len(calib)} x {a.ctx} tokens)", flush=True)
            for name, m in group:
                w, H = m.weight.data, Hs.get(name)
                if a.method == "rtn":
                    q = rtn(w, grid, a.group)
                elif a.method == "gptq":
                    if H is None:
                        raise SystemExit("GPTQ needs calibration data")
                    q = gptq(w, H, grid, a.group)
                elif a.method == "awq":
                    if H is None:
                        raise SystemExit("AWQ needs calibration data")
                    q = awq(w, H, grid, a.group)
                elif a.method == "nf4":
                    q = nf4(w, a.group)
                elif a.method == "ternary":
                    q = ternary(w, a.group)
                else:
                    q = (custom(w.clone(), name, H) if custom_h else custom(w.clone(), name)).to(w.dtype)
                    if q.shape != w.shape:
                        raise SystemExit(f"custom quantize() changed the shape of {name}")
                we, wn = float((w.float() - q.float()).norm() ** 2), float(w.float().norm() ** 2)
                oe, on = out_err(w, q, H) if H is not None else (0.0, 0.0)
                L, kind = layer_of(name), name.split(".")[-1]
                for key, e in (("w", (we, wn)), ("o", (oe, on))):
                    tot[key][0] += e[0]; tot[key][1] += e[1]
                    per_layer.setdefault(L, {"w": [0.0, 0.0], "o": [0.0, 0.0]})[key][0] += e[0]
                    per_layer[L][key][1] += e[1]
                    per_kind.setdefault(kind, {"w": [0.0, 0.0], "o": [0.0, 0.0]})[key][0] += e[0]
                    per_kind[kind][key][1] += e[1]
                n_params += w.numel()
                orig[name], quant[name] = w.cpu(), q.cpu()
            del Hs
            if dev == "cuda":
                torch.cuda.empty_cache()
    mods = dict(model.named_modules())
    mods["lm_head"] = model.get_output_embeddings()
    print(f"quantized {len(quant)} matrices ({n_params / 1e6:.0f}M weights)", flush=True)

    def use(which):
        with torch.no_grad():
            for name, w in which.items():
                mods[name].weight.data.copy_(w.to(dev, non_blocking=True))

    rel = lambda e: round(math.sqrt(e[0] / e[1]), 5) if e[1] else None  # noqa: E731
    bpw = {"rtn": grid.bpw(a.group), "gptq": grid.bpw(a.group), "awq": grid.bpw(a.group), "nf4": 4 + 16 / a.group,
           "ternary": math.log2(3) + 16 / a.group}.get(a.method)
    res = {
        "method": a.method, "grid": a.grid, "bits": a.bits if a.method in ("rtn", "gptq", "awq") else None, "group": a.group, "sym": a.sym,
        "include_head": a.include_head, "kv_bits": a.kv_bits or None, "bits_per_weight": round(bpw, 3) if bpw else None,
        "calibration": {"chunks": len(calib), "tokens": len(calib) * a.ctx, "passes": len(batches) if calib else 0} if calib else None,
        "weight_rel_error": rel(tot["w"]), "output_rel_error": rel(tot["o"]), "weights_changed": n_params,
        "per_layer": [{"layer": L, "rel_error": rel(e["w"]), "out_error": rel(e["o"])} for L, e in sorted(per_layer.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))],
        "per_kind": {k: rel(e["w"]) for k, e in sorted(per_kind.items())},
        "per_kind_out": {k: rel(e["o"]) for k, e in sorted(per_kind.items())},
        "device": dev,
        "note": "Fake quantization: the model computes with the quantized values, so quality is real; export to a GGUF and profile it for real speed and memory.",
    }

    if not a.no_eval:
        ids = tok(open(a.text, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
        n = min(a.chunks, ids.numel() // a.ctx)
        chunks = [ids[i * a.ctx:(i + 1) * a.ctx] for i in range(n)]
        kls, same, nll_b, nll_q, cnt = [], 0, 0.0, 0.0, 0
        with torch.no_grad():
            for i, c in enumerate(chunks):
                x = c.unsqueeze(0).to(dev)
                use(orig)
                lb = torch.log_softmax(model(x).logits[0, :-1].float(), -1)
                use(quant)
                hooks = kv_hooks(model, a.kv_bits) if a.kv_bits else []
                try:
                    lq = torch.log_softmax(model(x).logits[0, :-1].float(), -1)
                finally:
                    for h in hooks:
                        h.remove()
                tgt = x[0, 1:]
                kls.append((lb.exp() * (lb - lq)).sum(-1).cpu())
                same += int((lb.argmax(-1) == lq.argmax(-1)).sum())
                nll_b -= float(lb.gather(1, tgt[:, None]).sum())
                nll_q -= float(lq.gather(1, tgt[:, None]).sum())
                cnt += tgt.numel()
                print(f"chunk {i + 1}/{n}", flush=True)
        kl = torch.cat(kls)
        res.update({"tokens": cnt, "kld": round(float(kl.mean()), 6), "kld_sd": round(float(kl.std() / math.sqrt(kl.numel())), 6),
                    "kld_p99": round(float(kl.quantile(0.99)), 5), "same_top": round(100 * same / cnt, 2),
                    "ppl_base": round(math.exp(nll_b / cnt), 4), "ppl": round(math.exp(nll_q / cnt), 4)})

    if a.export:
        use(quant)
        model.save_pretrained(a.export, safe_serialization=True)
        tok.save_pretrained(a.export)
        print("exported", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(res, f)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
