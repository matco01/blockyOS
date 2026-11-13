"""Technique playground: simulate a weight / KV-cache quantization method on a Hugging Face checkpoint and measure what
it does to the model, before anyone writes a kernel for it (run inside the ML venv).

"Fake quantization": weights are quantized and immediately dequantized back to fp16, so the model computes with exactly
the values the method would store. Quality effects are real; speed and memory are not measured (that needs a kernel).

Methods (weights of the decoder's linear layers; embeddings and the output head stay fp16 unless --include_head):
  rtn       round-to-nearest, `bits`, group-wise scales (`group` weights per scale), symmetric or asymmetric
  nf4       4-bit NormalFloat code book (QLoRA), absmax scale per group
  ternary   {-1, 0, +1} × absmean scale per group (BitNet b1.58 style)
  custom    your own Python: a file defining  quantize(w: torch.Tensor, name: str) -> torch.Tensor  (returns dequantized w)
KV: --kv_bits N fake-quantizes the key/value projection outputs per token (asymmetric, per head-dim group of 32). It acts
before rotary embeddings, so it approximates a quantized KV cache rather than reproducing one exactly.

Measured on held-out text (WikiText-2 test), against the unmodified checkpoint: KL divergence per token (mean, p99),
same-top-token %, perplexity of both; per-layer relative weight error ||W - Q(W)|| / ||W||; effective bits per weight.
Output JSON goes to --out; progress lines go to stdout.
"""
import argparse
import importlib.util
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


def _groups(w: torch.Tensor, g: int) -> tuple[torch.Tensor, tuple]:
    shape = w.shape
    flat = w.reshape(-1)
    pad = (-flat.numel()) % g
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.view(-1, g), (shape, pad)


def _ungroup(x: torch.Tensor, meta) -> torch.Tensor:
    shape, pad = meta
    flat = x.reshape(-1)
    return (flat[:-pad] if pad else flat).view(shape)


def rtn(w, bits, g, sym=True):
    x, meta = _groups(w.float(), g)
    if sym:
        qmax = 2 ** (bits - 1) - 1
        s = x.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
        q = (x / s).round().clamp(-qmax - 1, qmax) * s
    else:
        lo, hi = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
        s = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
        q = ((x - lo) / s).round().clamp(0, 2 ** bits - 1) * s + lo
    return _ungroup(q, meta).to(w.dtype)


def nf4(w, g):
    x, meta = _groups(w.float(), g)
    s = x.abs().amax(1, keepdim=True).clamp_min(1e-8)
    code = NF4.to(x.device)
    idx = ((x / s).unsqueeze(-1) - code).abs().argmin(-1)
    return _ungroup(code[idx] * s, meta).to(w.dtype)


def ternary(w, g):
    x, meta = _groups(w.float(), g)
    s = x.abs().mean(1, keepdim=True).clamp_min(1e-8)
    return _ungroup((x / s).round().clamp(-1, 1) * s, meta).to(w.dtype)


def bits_per_weight(a) -> float | None:
    """Storage cost including fp16 scales (and zero points for asymmetric RTN)."""
    if a.method == "rtn":
        return a.bits + 16 / a.group * (1 if a.sym else 2)
    if a.method == "nf4":
        return 4 + 16 / a.group
    if a.method == "ternary":
        return math.log2(3) + 16 / a.group
    return None


def decoder_linears(model, include_head: bool):
    head = model.get_output_embeddings()
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and (include_head or m is not head) and ".layers." in f".{name}.".replace(".h.", ".layers."):
            yield name, m


def layer_of(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts[:-1]):
        if p in ("layers", "h") and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def kv_hooks(model, bits: int):
    """Fake-quantize k/v projection outputs per token, asymmetric, groups of 32 along the feature dimension."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["rtn", "nf4", "ternary", "custom", "none"], default="rtn")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--custom", default="")
    ap.add_argument("--include_head", action="store_true")
    ap.add_argument("--kv_bits", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=512)
    a = ap.parse_args()
    t0 = time.time()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    ids = tok(open(a.text, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
    n = min(a.chunks, ids.numel() // a.ctx)
    chunks = [ids[i * a.ctx:(i + 1) * a.ctx] for i in range(n)]
    print(f"loaded on {dev}; {n} chunks x {a.ctx} tokens", flush=True)

    custom = None
    if a.method == "custom":
        spec = importlib.util.spec_from_file_location("blocky_custom_quant", a.custom)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        custom = mod.quantize

    # Quantize once; keep both versions of every changed weight on the CPU and swap them in per pass.
    orig, quant, per_layer, per_kind = {}, {}, {}, {}
    total_err = total_norm = 0.0
    n_params = 0
    with torch.no_grad():
        for name, m in decoder_linears(model, a.include_head):
            if a.method == "none":
                break
            w = m.weight.data
            if a.method == "rtn":
                q = rtn(w, a.bits, a.group, a.sym)
            elif a.method == "nf4":
                q = nf4(w, a.group)
            elif a.method == "ternary":
                q = ternary(w, a.group)
            else:
                q = custom(w.clone(), name).to(w.dtype)
                if q.shape != w.shape:
                    raise SystemExit(f"custom quantize() changed the shape of {name}")
            err, norm = float((w.float() - q.float()).norm() ** 2), float(w.float().norm() ** 2)
            total_err, total_norm, n_params = total_err + err, total_norm + norm, n_params + w.numel()
            L, kind = layer_of(name), name.split(".")[-1]
            per_layer.setdefault(L, [0.0, 0.0])
            per_layer[L][0] += err
            per_layer[L][1] += norm
            per_kind.setdefault(kind, [0.0, 0.0])
            per_kind[kind][0] += err
            per_kind[kind][1] += norm
            orig[name], quant[name] = w.cpu(), q.cpu()
    mods = dict(model.named_modules())
    print(f"quantized {len(quant)} matrices ({n_params / 1e6:.0f}M weights)", flush=True)

    def use(which):
        with torch.no_grad():
            for name, w in which.items():
                mods[name].weight.data.copy_(w.to(dev, non_blocking=True))

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
    rel = lambda e: round(math.sqrt(e[0] / e[1]), 5) if e[1] else None  # noqa: E731
    res = {
        "method": a.method, "bits": a.bits if a.method == "rtn" else None, "group": a.group, "sym": a.sym, "include_head": a.include_head,
        "kv_bits": a.kv_bits or None, "bits_per_weight": round(bits_per_weight(a), 3) if bits_per_weight(a) else None,
        "tokens": cnt, "kld": round(float(kl.mean()), 6), "kld_sd": round(float(kl.std() / math.sqrt(kl.numel())), 6),
        "kld_p99": round(float(kl.quantile(0.99)), 5), "same_top": round(100 * same / cnt, 2),
        "ppl_base": round(math.exp(nll_b / cnt), 4), "ppl": round(math.exp(nll_q / cnt), 4),
        "weight_rel_error": round(math.sqrt(total_err / total_norm), 5) if total_norm else None, "weights_changed": n_params,
        "per_layer": [{"layer": L, "rel_error": rel(e)} for L, e in sorted(per_layer.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))],
        "per_kind": {k: rel(e) for k, e in sorted(per_kind.items())},
        "device": dev, "seconds": round(time.time() - t0, 1),
        "note": "Fake quantization: the model computes with the quantized values, so quality is real; speed and memory need a kernel and aren't measured.",
    }
    with open(a.out, "w") as f:
        json.dump(res, f)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
