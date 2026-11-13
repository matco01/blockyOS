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
            quantized, through the inverse of H (block size 128, 1% damping, no act-order: the reference defaults).
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

Modes (--mode):
  layerwise (default) one decoder layer at a time, loaded straight from the checkpoint files: GPU memory ~ one layer plus
            the output head, RAM ~ one file shard, so 7-14B models fit on an 8 GB GPU. Sequential, as in the GPTQ paper:
            each layer is calibrated on the outputs of the already-quantized layers. The reference model is evaluated
            alongside, layer by layer.
  whole     the whole model on the GPU at once (small models; calibration inputs from the full-precision model).
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
    # Memory matters here (H is 0.6-1.2 GB for the largest layers of 7-14B models): keep at most two n x n matrices alive.
    Hd = H.clone()
    dead = torch.diag(Hd) == 0
    Hd[dead, dead] = 1
    W[:, dead] = 0
    mean_diag = torch.mean(torch.diag(Hd))
    Hd.diagonal().add_(damp * mean_diag)
    try:
        L = torch.linalg.cholesky(Hd)
    except RuntimeError:                                   # badly conditioned: more damping
        Hd.diagonal().add_(0.1 * mean_diag)
        L = torch.linalg.cholesky(Hd)
    del Hd
    Linv = torch.cholesky_inverse(L)
    del L
    Hinv = torch.linalg.cholesky(Linv, upper=True)
    del Linv
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


class Quantizer:
    """Applies the chosen method to one linear layer and keeps the error statistics."""

    def __init__(self, a, grid: Grid):
        self.a, self.grid = a, grid
        self.custom, self.custom_h = None, False
        if a.method == "custom":
            spec = importlib.util.spec_from_file_location("blocky_custom_quant", a.custom)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.custom = mod.quantize
            self.custom_h = len(inspect.signature(self.custom).parameters) >= 3
        self.per_layer, self.per_kind = {}, {}
        self.tot = {"w": [0.0, 0.0], "o": [0.0, 0.0]}
        self.n_params = 0

    def __call__(self, name: str, w: torch.Tensor, H) -> torch.Tensor:
        a, grid = self.a, self.grid
        if a.method in ("gptq", "awq") and H is None:
            raise SystemExit(f"{a.method.upper()} needs calibration data")
        if a.method == "rtn":
            q = rtn(w, grid, a.group)
        elif a.method == "gptq":
            q = gptq(w, H, grid, a.group)
        elif a.method == "awq":
            q = awq(w, H, grid, a.group)
        elif a.method == "nf4":
            q = nf4(w, a.group)
        elif a.method == "ternary":
            q = ternary(w, a.group)
        else:
            q = (self.custom(w.clone(), name, H) if self.custom_h else self.custom(w.clone(), name)).to(w.dtype)
            if q.shape != w.shape:
                raise SystemExit(f"custom quantize() changed the shape of {name}")
        we, wn = float((w.float() - q.float()).norm() ** 2), float(w.float().norm() ** 2)
        oe, on = out_err(w, q, H) if H is not None else (0.0, 0.0)
        L, kind = layer_of(name), name.split(".")[-1]
        for key, e in (("w", (we, wn)), ("o", (oe, on))):
            self.tot[key][0] += e[0]; self.tot[key][1] += e[1]
            for bucket, k2 in ((self.per_layer, L), (self.per_kind, kind)):
                b = bucket.setdefault(k2, {"w": [0.0, 0.0], "o": [0.0, 0.0]})
                b[key][0] += e[0]; b[key][1] += e[1]
        self.n_params += w.numel()
        return q

    def report(self) -> dict:
        a, grid = self.a, self.grid
        rel = lambda e: round(math.sqrt(e[0] / e[1]), 5) if e[1] else None  # noqa: E731
        bpw = {"rtn": grid.bpw(a.group), "gptq": grid.bpw(a.group), "awq": grid.bpw(a.group), "nf4": 4 + 16 / a.group,
               "ternary": math.log2(3) + 16 / a.group}.get(a.method)
        return {
            "method": a.method, "grid": a.grid, "bits": a.bits if a.method in ("rtn", "gptq", "awq") else None, "group": a.group, "sym": a.sym,
            "include_head": a.include_head, "kv_bits": a.kv_bits or None, "bits_per_weight": round(bpw, 3) if bpw else None,
            "weight_rel_error": rel(self.tot["w"]), "output_rel_error": rel(self.tot["o"]), "weights_changed": self.n_params,
            "per_layer": [{"layer": L, "rel_error": rel(e["w"]), "out_error": rel(e["o"])}
                          for L, e in sorted(self.per_layer.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))],
            "per_kind": {k: rel(e["w"]) for k, e in sorted(self.per_kind.items())},
            "per_kind_out": {k: rel(e["o"]) for k, e in sorted(self.per_kind.items())},
            "note": "Fake quantization: the model computes with the quantized values, so quality is real; export to a GGUF and profile it for real speed and memory.",
        }


class Scorer:
    """KL divergence, same top token and perplexity between reference and quantized logits, chunk by chunk."""

    def __init__(self):
        self.kls, self.same, self.nll_b, self.nll_q, self.cnt = [], 0, 0.0, 0.0, 0

    def add(self, logits_b, logits_q, tgt, step: int = 128):
        # Slices of tokens: full-vocabulary fp32 tables for a whole chunk can take gigabytes.
        for s in range(0, tgt.numel(), step):
            lb = torch.log_softmax(logits_b[s:s + step].float(), -1)
            lq = torch.log_softmax(logits_q[s:s + step].float(), -1)
            t = tgt[s:s + step]
            self.kls.append((lb.exp() * (lb - lq)).sum(-1).cpu())
            self.same += int((lb.argmax(-1) == lq.argmax(-1)).sum())
            self.nll_b -= float(lb.gather(1, t[:, None]).sum())
            self.nll_q -= float(lq.gather(1, t[:, None]).sum())
            self.cnt += t.numel()
            del lb, lq

    def result(self) -> dict:
        kl = torch.cat(self.kls)
        return {"tokens": self.cnt, "kld": round(float(kl.mean()), 6), "kld_sd": round(float(kl.std() / math.sqrt(kl.numel())), 6),
                "kld_p99": round(float(kl.quantile(0.99)), 5), "same_top": round(100 * self.same / self.cnt, 2),
                "ppl_base": round(math.exp(self.nll_b / self.cnt), 4), "ppl": round(math.exp(self.nll_q / self.cnt), 4)}


def chunks_of(tok, path: str, n: int, ctx: int) -> list:
    ids = tok(open(path, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
    return [ids[i * ctx:(i + 1) * ctx] for i in range(min(n, ids.numel() // ctx))]


# ---------- Checkpoint files: read tensors lazily, map model names to checkpoint names, write the export ----------

class Checkpoint:
    SKIP = ("visual", "vision", "mtp.", "audio")

    def __init__(self, d: str):
        import glob
        from pathlib import Path
        from safetensors import safe_open
        self.dir = Path(d)
        self.files = {}
        self.where = {}
        for f in sorted(glob.glob(str(self.dir / "*.safetensors"))):
            h = safe_open(f, "pt")
            self.files[f] = h
            for k in h.keys():
                self.where[k] = f

    def key_for(self, model_key: str) -> str | None:
        if model_key in self.where:
            return model_key
        tail = model_key.split(".", 1)[1] if model_key.startswith("model.") else model_key
        hits = [k for k in self.where if (k == tail or k.endswith("." + tail)) and not any(s in k for s in self.SKIP)]
        return min(hits, key=len) if hits else None

    def get(self, model_key: str) -> torch.Tensor:
        k = self.key_for(model_key)
        if k is None:
            raise KeyError(model_key)
        return self.files[self.where[k]].get_tensor(k)

    def export(self, out_dir: str, replaced: dict, shard_bytes: int = 2 * 2 ** 30):
        """Copy the checkpoint to out_dir with `replaced` (model key -> tensor) swapped in, keeping the original names."""
        import shutil
        from pathlib import Path
        from safetensors.torch import save_file
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for f in self.dir.iterdir():
            if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
                shutil.copy(f, out / f.name)
        swap = {}
        for mk, t in replaced.items():
            ck = self.key_for(mk)
            if ck is None:
                print(f"note: {mk} isn't in the checkpoint (tied weights?); left unchanged", flush=True)
            else:
                swap[ck] = t
        weight_map, shard, size, n = {}, {}, 0, 0

        def flush():
            nonlocal shard, size, n
            if shard:
                n += 1
                name = f"model-{n:05d}.safetensors"
                save_file({k: v.contiguous() for k, v in shard.items()}, str(out / name), metadata={"format": "pt"})
                for k in shard:
                    weight_map[k] = name
                shard, size = {}, 0

        for k, f in self.where.items():
            t = swap[k] if k in swap else self.files[f].get_tensor(k)
            shard[k] = t
            size += t.numel() * t.element_size()
            if size >= shard_bytes:
                flush()
        flush()
        (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
        print(f"exported {len(swap)} quantized tensors into {n} shards", flush=True)


# ---------- Whole model on the GPU (small models) ----------

def run_whole(a, grid, tok, dev) -> dict:
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    print(f"loaded on {dev} (whole model)", flush=True)
    qz = Quantizer(a, grid)
    linears = list(decoder_linears(model, a.include_head)) if a.method != "none" else []
    calib = chunks_of(tok, a.calib, a.calib_chunks, a.ctx) if a.calib and linears else []
    orig, quant = {}, {}
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
                print(f"calibrated {bi + 1}/{len(batches)}", flush=True)
            for name, m in group:
                q = qz(name, m.weight.data, Hs.get(name))
                orig[name], quant[name] = m.weight.data.cpu(), q.cpu()
            del Hs
            if dev == "cuda":
                torch.cuda.empty_cache()
    print(f"quantized {len(quant)} matrices ({qz.n_params / 1e6:.0f}M weights)", flush=True)
    mods = dict(model.named_modules())
    mods["lm_head"] = model.get_output_embeddings()

    def use(which):
        with torch.no_grad():
            for name, w in which.items():
                mods[name].weight.data.copy_(w.to(dev, non_blocking=True))

    res = {**qz.report(), "mode": "whole model (calibration inputs from the full-precision model)", "device": dev,
           "calibration": {"chunks": len(calib), "tokens": len(calib) * a.ctx} if calib else None}
    if not a.no_eval:
        sc = Scorer()
        chunks = chunks_of(tok, a.text, a.chunks, a.ctx)
        with torch.no_grad():
            for i, c in enumerate(chunks):
                x = c.unsqueeze(0).to(dev)
                use(orig)
                lb = model(x).logits[0, :-1]
                use(quant)
                hooks = kv_hooks(model, a.kv_bits) if a.kv_bits else []
                try:
                    lq = model(x).logits[0, :-1]
                finally:
                    for h in hooks:
                        h.remove()
                sc.add(lb, lq, x[0, 1:])
                print(f"chunk {i + 1}/{len(chunks)}", flush=True)
        res.update(sc.result())
    if a.export:
        del model
        Checkpoint(a.model).export(a.export, {f"{n}.weight": t.half() for n, t in quant.items()})
    return res


# ---------- One layer at a time (models bigger than the GPU) ----------

class Catcher(torch.nn.Module):
    """Stands in for a decoder layer during a cheap forward pass: records the layer's inputs, passes them through."""

    def __init__(self, i, store):
        super().__init__()
        self.i, self.store = i, store

    def forward(self, hidden_states, *args, **kwargs):
        if args:
            raise RuntimeError("positional layer arguments aren't supported")
        e = self.store.setdefault(self.i, {"kw": kwargs, "h": []})
        if self.i == 0:                                    # only the first layer's inputs are needed; later ones are computed
            e["h"].append(hidden_states)
        return hidden_states


def _find_layers(model):
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "model.decoder.layers"):
        parent, obj = None, model
        try:
            for p in path.split("."):
                parent, obj = obj, getattr(obj, p)
            return parent, path.rsplit(".", 1)[-1], obj
        except AttributeError:
            continue
    raise RuntimeError("can't find the decoder layers of this architecture")


def _to(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, tuple):
        return tuple(_to(v, dev) for v in x)
    if isinstance(x, list):
        return [_to(v, dev) for v in x]
    if isinstance(x, dict):
        return {k: _to(v, dev) for k, v in x.items()}
    return x


def _owner(module, name):
    *path, leaf = name.split(".")
    return (module.get_submodule(".".join(path)) if path else module), leaf


def _materialize(module, prefix: str, ck: Checkpoint, dev, dtype, keep32=()):
    """Give a module real weights from the checkpoint, on `dev` (its parameters start out without memory)."""
    for n, b in list(module.named_buffers()):
        owner, leaf = _owner(module, n)
        if b.device.type == "meta":                        # a stored buffer
            owner._buffers[leaf] = ck.get(prefix + n).to(dev)
        else:                                              # computed at construction (e.g. rotary tables)
            owner._buffers[leaf] = b.to(dev)
    for n, _ in list(module.named_parameters()):
        owner, leaf = _owner(module, n)
        t = ck.get(prefix + n).to(dev, torch.float32 if any(k in prefix + n for k in keep32) else dtype)
        owner._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)


def _release(module):
    """Drop a module's weights again (back to no memory)."""
    for n, p in list(module.named_parameters()):
        owner, leaf = _owner(module, n)
        owner._parameters[leaf] = torch.nn.Parameter(torch.empty(p.shape, dtype=p.dtype, device="meta"), requires_grad=False)


def run_layerwise(a, grid, tok, dev) -> dict:
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    t_dtype = torch.float16 if dev == "cuda" else torch.float32
    cfg = AutoConfig.from_pretrained(a.model)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=t_dtype)
    model.eval()
    ck = Checkpoint(a.model)
    keep32 = tuple(getattr(model, "_keep_in_fp32_modules", None) or ())
    parent, attr, layers = _find_layers(model)
    base = parent
    prefix_of = {id(m): n for n, m in model.named_modules()}
    layer_prefix = prefix_of[id(layers)] + "."
    n_layers = len(layers)
    print(f"loading layer by layer ({n_layers} layers) on {dev}", flush=True)

    # Everything around the layers: embeddings, final norm, rotary tables, output head.
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    norm = getattr(base, "norm", None) or getattr(base, "final_layernorm", None)
    _materialize(emb, prefix_of[id(emb)] + ".", ck, dev, t_dtype)
    if norm is not None:
        _materialize(norm, prefix_of[id(norm)] + ".", ck, dev, t_dtype, keep32)
    for n, m in base.named_children():
        if m is not layers and m is not emb and m is not norm:
            _materialize(m, prefix_of[id(m)] + ".", ck, dev, t_dtype, keep32)  # e.g. rotary embedding

    calib = chunks_of(tok, a.calib, a.calib_chunks, a.ctx) if a.calib and a.method != "none" else []
    evals = [] if a.no_eval else chunks_of(tok, a.text, a.chunks, a.ctx)

    # 1) Cheap pass with stand-in layers: each layer's keyword inputs (masks, positions) and layer 0's hidden states.
    store: dict = {}
    real = list(layers)
    for i in range(n_layers):
        layers[i] = Catcher(i, store)
    with torch.no_grad():
        for c in calib + evals:
            base(input_ids=c.unsqueeze(0).to(dev), use_cache=False)
    for i in range(n_layers):
        layers[i] = real[i]
    kws = [store[i]["kw"] for i in range(n_layers)]        # same length for every chunk, so one set per layer
    h0 = store[0]["h"]
    h_cal = h0[:len(calib)]
    h_ev_ref = h0[len(calib):]
    h_ev_q = list(h_ev_ref)
    del store
    _release(emb)                                          # no longer needed: free its memory
    if dev == "cuda":
        torch.cuda.empty_cache()

    qz = Quantizer(a, grid)
    replaced = {}
    with torch.no_grad():
        for i, layer in enumerate(real):
            lp = f"{layer_prefix}{i}."
            _materialize(layer, lp, ck, dev, t_dtype, keep32)
            kw = _to(kws[i], dev)
            # Reference stream: the unmodified layer on unmodified inputs.
            h_ev_ref = [layer(h, **kw) for h in h_ev_ref]
            # Calibration on the quantized stream (inputs come from the already-quantized layers: sequential GPTQ).
            lins = [(f"{lp}{n}", m) for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)] if a.method != "none" else []
            Hs = {}
            if calib and lins:
                H, cnt = {}, {}
                def mk(name):
                    def hook(m, inp, out):
                        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                        if name not in H:
                            H[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device); cnt[name] = 0
                        H[name].addmm_(x.T, x); cnt[name] += x.shape[0]
                    return hook
                hs = [m.register_forward_hook(mk(n)) for n, m in lins]
                try:
                    for h in h_cal:
                        layer(h, **kw)
                finally:
                    for hk in hs:
                        hk.remove()
                Hs = {k: v / max(1, cnt[k]) for k, v in H.items()}
            for name, m in lins:
                q = qz(name, m.weight.data, Hs.get(name))
                m._parameters["weight"] = torch.nn.Parameter(q, requires_grad=False)
                if a.export:
                    replaced[f"{name}.weight"] = q.half().cpu()
            del Hs
            # Advance the quantized streams through the quantized layer.
            h_cal = [layer(h, **kw) for h in h_cal]
            hooks = kv_hooks(layer, a.kv_bits) if a.kv_bits else []
            try:
                h_ev_q = [layer(h, **kw) for h in h_ev_q]
            finally:
                for hk in hooks:
                    hk.remove()
            _release(layer)
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"layer {i + 1}/{n_layers}", flush=True)

    res = {**qz.report(), "mode": "layer by layer (sequential: each layer is calibrated on the outputs of the already-quantized layers)",
           "device": dev, "calibration": {"chunks": len(calib), "tokens": len(calib) * a.ctx} if calib else None}
    if evals:
        if dev == "cuda":
            torch.cuda.empty_cache()
        if head is not None and next(head.parameters()).device.type == "meta":
            try:
                _materialize(head, prefix_of[id(head)] + ".", ck, dev, t_dtype)
            except KeyError:                               # tied to the embeddings
                head._parameters["weight"] = torch.nn.Parameter(ck.get(prefix_of[id(emb)] + ".weight").to(dev, t_dtype), requires_grad=False)
        sc = Scorer()
        with torch.no_grad():
            for c, hb, hq in zip(evals, h_ev_ref, h_ev_q):
                if norm is not None:
                    hb, hq = norm(hb), norm(hq)
                sc.add(head(hb)[0, :-1], head(hq)[0, :-1], c[1:].to(dev))
        res.update(sc.result())
    if a.export:
        del model
        Checkpoint(a.model).export(a.export, replaced)
    return res


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
    ap.add_argument("--mode", choices=["layerwise", "whole"], default="layerwise")
    a = ap.parse_args()
    t0 = time.time()
    if a.grid == "q4_0":
        a.bits, a.group, a.sym = 4, 32, True
    if a.include_head and a.mode == "layerwise":
        print("note: the output head isn't quantized in layer-by-layer mode", flush=True)
        a.include_head = False
    grid = Grid(a.grid, a.bits, a.sym)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    res = run_layerwise(a, grid, tok, dev) if a.mode == "layerwise" else run_whole(a, grid, tok, dev)
    res["seconds"] = round(time.time() - t0, 1)
    if dev == "cuda":
        res["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    with open(a.out, "w") as f:
        json.dump(res, f)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
