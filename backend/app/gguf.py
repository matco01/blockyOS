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
    })
    return info


def kv_bytes_per_token(info: dict, cache_type: str = "q8_0") -> float:
    """Rough KV-cache cost per token of context (upper bound for hybrid models)."""
    bytes_per = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}.get(cache_type, 2.0)
    layers, kv_heads, head_dim = info.get("layers"), info.get("kv_heads"), info.get("head_dim")
    if not (layers and kv_heads and head_dim):
        # Not downloaded yet: typical modern GQA model is ~0.14 MB/token at f16.
        return 0.14 * 2**20 * bytes_per / 2.0
    return 2 * layers * kv_heads * head_dim * bytes_per
