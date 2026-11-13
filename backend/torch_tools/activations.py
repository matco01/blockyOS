"""Activations of a Hugging Face checkpoint on one prompt (run inside the ML venv).

Output JSON:
  tokens          the prompt's tokens
  hidden_norm     [layer][token]  L2 norm of each layer's output (residual stream) per token
  delta_norm      [layer][token]  how much each layer changed the residual stream (||h_l - h_{l-1}||)
  logit_lens      [layer]         what the model would predict next if it stopped at that layer (top 5 + prob)
  attn_entropy    [layer][token]  attention entropy per query token (mean over heads), when the model exposes it
  top_dims        [layer]         residual dimensions with the largest mean |activation| ("massive activations")
"""
import argparse
import json
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=64)
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

    # Logit lens: final norm + LM head applied to each layer's residual at the last position.
    norm = getattr(getattr(model, "model", model), "norm", None)
    head = model.get_output_embeddings()
    lens = []
    for li, h in enumerate(hs):
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
    }
    with open(a.out, "w") as f:
        json.dump(res, f)
    print("ok", len(tokens), "tokens,", len(hs) - 1, "layers", file=sys.stderr)


if __name__ == "__main__":
    main()
