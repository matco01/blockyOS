"""Minimal GGUF metadata reader (header only, no tensor data).

Reads the key/value section of a .gguf file so the UI can show what a model
really is (architecture, parameters, quantization, layers, experts, context)
instead of guessing from the filename. Large arrays (tokenizer vocab) are
skipped without being stored.
"""
import re
import struct
from functools import lru_cache
from pathlib import Path

# GGUF value types
_SIMPLE = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9

# llama.cpp LLAMA_FTYPE values -> quantization name
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE",
}

_QUANT_IN_NAME = re.compile(r"(UD-)?(IQ\d_[A-Z0-9_]+|Q\d_K(?:_[SMLX]+)?|Q\d_\d|BF16|F16|F32|MXFP4(?:_MOE)?)", re.I)


class _Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self):
        n = self.unpack("<Q")
        if n > 1 << 24:
            raise ValueError("implausible string length")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, vtype, keep_arrays: bool):
        if vtype in _SIMPLE:
            return self.unpack(_SIMPLE[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            item_type = self.unpack("<I")
            count = self.unpack("<Q")
            if not keep_arrays or count > 64:
                self._skip_array(item_type, count)
                return f"<array of {count}>"
            return [self.value(item_type, False) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {vtype}")

    def _skip_array(self, item_type, count):
        if item_type in _SIMPLE:
            self.f.seek(struct.calcsize(_SIMPLE[item_type]) * count, 1)
        elif item_type == _STRING:
            for _ in range(count):
                self.f.seek(self.unpack("<Q"), 1)
        else:
            for _ in range(count):
                self.value(item_type, False)


@lru_cache(maxsize=64)
def _read(path: str, mtime: float) -> dict:
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        r = _Reader(f)
        version = r.unpack("<I")
        tensor_count = r.unpack("<Q")
        kv_count = r.unpack("<Q")
        kv = {}
        for _ in range(kv_count):
            key = r.string()
            kv[key] = r.value(r.unpack("<I"), keep_arrays=True)
    kv["_gguf_version"] = version
    kv["_tensor_count"] = tensor_count
    return kv


GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
              12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
              20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
              29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}


@lru_cache(maxsize=16)
def _tensors(path: str, mtime: float) -> dict:
    """Every tensor's name, shape, type and size in bytes (from the header's tensor table)."""
    import os
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        r = _Reader(f)
        r.unpack("<I")
        n_tensors = r.unpack("<Q")
        n_kv = r.unpack("<Q")
        align = 32
        for _ in range(n_kv):
            key = r.string()
            val = r.value(r.unpack("<I"), keep_arrays=False)
            if key == "general.alignment" and isinstance(val, int):
                align = val
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
            ttype = r.unpack("<I")
            offset = r.unpack("<Q")
            infos.append({"name": name, "shape": dims, "type": GGML_TYPES.get(ttype, str(ttype)), "offset": offset})
        data_start = -(-f.tell() // align) * align
    size = os.path.getsize(path)
    order = sorted(range(len(infos)), key=lambda i: infos[i]["offset"])
    for j, i in enumerate(order):  # size = distance to the next tensor (exact, includes alignment padding)
        end = infos[order[j + 1]]["offset"] if j + 1 < len(order) else size - data_start
        infos[i]["bytes"] = end - infos[i]["offset"]
        n = 1
        for d in infos[i]["shape"]:
            n *= d
        infos[i]["params"] = n
    return {"tensors": infos}


def _part(name: str) -> str:
    """Which part of the network a tensor belongs to, in plain terms."""
    n = name.split(".", 2)[-1] if name.startswith("blk.") else name
    if "exps" in n or "ffn_gate_inp" in n or "shexp" in n:
        return "experts"
    if n.startswith(("ssm_", "time_mix", "attn_qkv_lin", "linear_attn")) or ".ssm_" in name:
        return "linear"
    if n.startswith("attn_") and "norm" not in n:
        return "attention"
    if n.startswith("ffn_") and "norm" not in n:
        return "ffn"
    return "other"


def anatomy(path: Path) -> dict:
    """The network's real structure: embedding, each layer's parts (sized by bytes), output."""
    t = _tensors(str(path), path.stat().st_mtime)["tensors"]
    layers: dict[int, dict] = {}
    embedding = {"bytes": 0, "params": 0, "types": set()}
    output = {"bytes": 0, "params": 0, "types": set()}
    for x in t:
        if x["name"].startswith("blk."):
            idx = int(x["name"].split(".")[1])
            L = layers.setdefault(idx, {"index": idx, "parts": {}})
            p = L["parts"].setdefault(_part(x["name"]), {"bytes": 0, "params": 0, "types": set(), "tensors": 0})
            p["bytes"] += x["bytes"]; p["params"] += x["params"]; p["types"].add(x["type"]); p["tensors"] += 1
        elif x["name"].startswith("token_embd"):
            embedding["bytes"] += x["bytes"]; embedding["params"] += x["params"]; embedding["types"].add(x["type"])
        else:  # output, output_norm, rope tables…
            output["bytes"] += x["bytes"]; output["params"] += x["params"]; output["types"].add(x["type"])
    out_layers = []
    for idx in sorted(layers):
        raw = layers[idx]["parts"]
        if "linear" in raw and "attention" in raw:
            # Hybrid layers name their linear-attention projections attn_qkv/attn_gate: they belong to the linear block.
            a, l = raw.pop("attention"), raw["linear"]
            l["bytes"] += a["bytes"]; l["params"] += a["params"]; l["types"] |= a["types"]; l["tensors"] += a["tensors"]
        parts = {k: {**v, "types": sorted(v["types"])} for k, v in raw.items()}
        mixer = "linear" if "linear" in parts else "attention"
        out_layers.append({"index": idx, "mixer": mixer, "parts": parts,
                           "bytes": sum(p["bytes"] for p in parts.values()), "params": sum(p["params"] for p in parts.values())})
    info = metadata(path)
    return {
        "file": path.name, "name": info.get("name"), "architecture": info.get("architecture"),
        "layers": out_layers,
        "embedding": {**embedding, "types": sorted(embedding["types"])},
        "output": {**output, "types": sorted(output["types"])},
        "total_params": sum(x["params"] for x in t), "total_bytes": sum(x["bytes"] for x in t),
        "experts": info.get("experts"), "experts_used": info.get("experts_used"),
        "heads": info.get("heads"), "kv_heads": info.get("kv_heads"), "embedding_length": info.get("embedding"),
        "tensor_count": len(t),
        "tied_output": not any(x["name"] == "output.weight" for x in t),  # output reuses the embedding matrix
    }


def metadata(path: Path) -> dict:
    """Summary of a model file. Never raises: errors are reported in the result."""
    info: dict = {"file": path.name, "size_bytes": path.stat().st_size}
    q = _QUANT_IN_NAME.search(path.name)
    info["quant"] = q.group(0).upper() if q else None
    try:
        kv = _read(str(path), path.stat().st_mtime)
    except (OSError, ValueError) as e:
        info["error"] = str(e)
        return info

    arch = kv.get("general.architecture", "unknown")
    a = lambda key: kv.get(f"{arch}.{key}")  # noqa: E731
    heads = a("attention.head_count")
    embd = a("embedding_length")
    kv_heads = a("attention.head_count_kv")
    if isinstance(heads, list):  # some models store per-layer head counts
        heads = max(heads)
    if isinstance(kv_heads, list):
        kv_heads = max(kv_heads)
    head_dim = a("attention.key_length") or (embd // heads if embd and heads else None)

    info.update({
        "name": kv.get("general.name") or path.stem,
        "architecture": arch,
        "size_label": kv.get("general.size_label"),
        "quant": FILE_TYPES.get(kv.get("general.file_type"), info["quant"]),
        "context_length": a("context_length"),
        "layers": a("block_count"),
        "embedding": embd,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "experts": a("expert_count"),
        "experts_used": a("expert_used_count"),
        "license": kv.get("general.license"),
        "tensors": kv.get("_tensor_count"),
        "tokenizer": tokenizer_signature(kv),
    })
    return info


def tokenizer_signature(kv: dict) -> str | None:
    """Tokenizer identity: type / pre-tokenizer / vocabulary size. Two models can only be compared token-by-token
    (KL divergence, perplexity) when this matches."""
    toks = kv.get("tokenizer.ggml.tokens")
    n = None
    if isinstance(toks, list):
        n = len(toks)
    elif isinstance(toks, str) and (m := re.search(r"\d+", toks)):
        n = int(m.group(0))
    if not (kv.get("tokenizer.ggml.model") or n):
        return None
    return f"{kv.get('tokenizer.ggml.model')}/{kv.get('tokenizer.ggml.pre')}/{n}"


def kv_bytes_per_token(info: dict, cache_type: str = "q8_0") -> float:
    """Rough KV-cache cost per token of context (upper bound for hybrid models)."""
    bytes_per = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}.get(cache_type, 2.0)
    layers, kv_heads, head_dim = info.get("layers"), info.get("kv_heads"), info.get("head_dim")
    if not (layers and kv_heads and head_dim):
        # Not downloaded yet: typical modern GQA model is ~0.14 MB/token at f16.
        return 0.14 * 2**20 * bytes_per / 2.0
    return 2 * layers * kv_heads * head_dim * bytes_per
