"""Activations of a Hugging Face checkpoint on one prompt (run inside the ML venv).

Output JSON:
  tokens          the prompt's tokens
  hidden_norm     [layer][token]  L2 norm of each layer's output (residual stream) per token
  delta_norm      [layer][token]  how much each layer changed the residual stream (||h_l - h_{l-1}||)
  logit_lens      [layer]         what the model would predict next if it stopped at that layer (top 5 + prob)
  attn_entropy    [layer][token]  attention entropy per query token (mean over heads), when the model exposes it
  top_dims        [layer]         residual dimensions with the largest mean |activation| ("massive activations")
  timing          per layer: GPU milliseconds for the prompt (prefill) and for one next token (decode),
                  split into attention / feed-forward / rest (CUDA events around each module)
  knockout        per layer: skip that layer (its output = its input) and measure how the next-token
                  prediction changes (KL divergence, probability of the original answer)
"""
import argparse
import json
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return list(obj)
        except AttributeError:
            continue
    return []


def parts(layer):
    """The attention-like and feed-forward submodules of a decoder layer (names differ between architectures)."""
    attn = next((getattr(layer, n) for n in ("self_attn", "linear_attn", "attention", "attn", "mixer") if hasattr(layer, n)), None)
    ffn = next((getattr(layer, n) for n in ("mlp", "feed_forward", "ffn", "block_sparse_moe") if hasattr(layer, n)), None)
    return attn, ffn


class Timer:
    """CUDA-event timing of module forwards: total per layer and per part."""

    def __init__(self, layers):
        self.handles, self.events, self.layers = [], [], layers

    def _wrap(self, mod, key):
        def pre(m, a, kw=None):
            e = torch.cuda.Event(enable_timing=True); e.record(); m._blocky_t0 = e
        def post(m, a, *rest):
            e = torch.cuda.Event(enable_timing=True); e.record(); self.events.append((key, m._blocky_t0, e))
        self.handles.append(mod.register_forward_pre_hook(pre))
        self.handles.append(mod.register_forward_hook(post))

    def start(self):
        for i, L in enumerate(self.layers):
            self._wrap(L, (i, "layer"))
            a, f = parts(L)
            if a is not None:
                self._wrap(a, (i, "attention"))
            if f is not None:
                self._wrap(f, (i, "ffn"))

    def result(self):
        torch.cuda.synchronize()
        out = [{"layer": i, "total": 0.0, "attention": 0.0, "ffn": 0.0} for i in range(len(self.layers))]
        for (i, k), a, b in self.events:
            out[i]["total" if k == "layer" else k] += a.elapsed_time(b)
        for o in out:
            o["other"] = max(0.0, o["total"] - o["attention"] - o["ffn"])
            for k in ("total", "attention", "ffn", "other"):
                o[k] = round(o[k], 4)
        self.events = []
        return out

    def stop(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--knockout", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = {"torch_dtype": dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="eager", **kw).to(dev).eval()
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(a.model, **kw).to(dev).eval()

    ids = tok(a.prompt, return_tensors="pt").input_ids[:, : a.max_tokens].to(dev)
    with torch.no_grad():
        try:
            out = model(ids, output_hidden_states=True, output_attentions=True)
        except Exception:  # some architectures (e.g. linear-attention hybrids) can't return attention maps
            out = model(ids, output_hidden_states=True)
    hs = [h[0].float() for h in out.hidden_states]  # [embeddings, layer1, ..., layerN], each [T, D]
    tokens = [tok.decode([t]) for t in ids[0].tolist()]
    base_logp = torch.log_softmax(out.logits[0, -1].float(), -1)
    answer = int(base_logp.argmax())

    # Logit lens: final norm + LM head applied to each layer's residual at the last position.
    norm = getattr(getattr(model, "model", model), "norm", None)
    head = model.get_output_embeddings()
    lens = []
    for h in hs:
        x = h[-1:].to(dtype)
        if norm is not None:
            x = norm(x)
        p = torch.softmax(head(x).float()[0], -1)
        top = torch.topk(p, 5)
        lens.append([{"token": tok.decode([i]), "p": round(v, 4)} for v, i in zip(top.values.tolist(), top.indices.tolist())])

    ent = None
    if getattr(out, "attentions", None):
        ent = []
        for att in out.attentions:
            if att is None:
                ent.append(None)
                continue
            p = att[0].float().clamp_min(1e-12)  # [H, T, T]
            ent.append((-(p * p.log()).sum(-1)).mean(0).tolist())

    res = {
        "model": a.model, "device": dev, "tokens": tokens, "n_layers": len(hs) - 1,
        "hidden_norm": [h.norm(dim=-1).tolist() for h in hs],
        "delta_norm": [None] + [(hs[i] - hs[i - 1]).norm(dim=-1).tolist() for i in range(1, len(hs))],
        "logit_lens": lens,
        "attn_entropy": ent,
        "top_dims": [[{"dim": int(d), "mean_abs": round(float(v), 3)} for v, d in zip(*torch.topk(h.abs().mean(0), 5))] for h in hs],
        "max_entropy": math.log(max(2, len(tokens))),
        "answer": tok.decode([answer]),
    }
    layers = decoder_layers(model)

    # Per-layer GPU time: warm up, then time a prompt pass and a single next-token step (with the KV cache).
    if a.timing and dev == "cuda" and layers:
        t = Timer(layers)
        with torch.no_grad():
            for _ in range(2):                             # warm-up both paths: JIT kernels compile on first use
                w = model(ids, use_cache=True)
                model(w.logits[:, -1:].argmax(-1), past_key_values=w.past_key_values, use_cache=True)
            torch.cuda.synchronize()
            t.start()
            o = model(ids, use_cache=True)
            prefill = t.result()
            nxt = o.logits[:, -1:].argmax(-1)
            model(nxt, past_key_values=o.past_key_values, use_cache=True)
            decode = t.result()
            t.stop()
        import importlib.util
        missing = [name for mod, name in (("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d"))
                   if importlib.util.find_spec(mod) is None and any(t in str(type(model)).lower() for t in ("qwen3_5", "qwen3next", "mamba", "gated"))]
        slow = (f" Some linear-attention parts ran slow reference code ({', '.join(missing)} not installed), which inflates their share." if missing else "")
        res["timing"] = {"prefill": prefill, "decode": decode, "prompt_tokens": int(ids.shape[1]),
                         "note": "PyTorch eager execution (fp16): shows the relative cost of layers and parts; absolute speed differs from llama.cpp/vLLM." + slow}

    # Knockout: skip one layer at a time and see how much the next-token prediction moves.
    if a.knockout and layers:
        ko = []
        for i, L in enumerate(layers):
            stash = {}
            def pre(m, args, kwargs, stash=stash):
                stash["h"] = args[0] if args else kwargs.get("hidden_states")
            def post(m, args, kwargs, output, stash=stash):
                return (stash["h"],) + tuple(output[1:]) if isinstance(output, tuple) else stash["h"]
            h1 = L.register_forward_pre_hook(pre, with_kwargs=True)
            h2 = L.register_forward_hook(post, with_kwargs=True)
            try:
                with torch.no_grad():
                    lp = torch.log_softmax(model(ids).logits[0, -1].float(), -1)
                kl = float((base_logp.exp() * (base_logp - lp)).sum())
                top = int(lp.argmax())
                ko.append({"layer": i, "kl": round(kl, 5), "p_answer": round(float(lp[answer].exp()), 5), "new_top": tok.decode([top]), "changed": top != answer})
            finally:
                h1.remove(); h2.remove()
        res["knockout"] = {"answer": tok.decode([answer]), "p_answer": round(float(base_logp[answer].exp()), 5), "layers": ko}

    with open(a.out, "w") as f:
        json.dump(res, f)
    print("ok", len(tokens), "tokens,", len(hs) - 1, "layers", file=sys.stderr)


if __name__ == "__main__":
    main()
