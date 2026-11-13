"""GPU kernel profile of a run: which operations the GPU spends its time on, for prefill and for decode.

Runs llama-bench under NVIDIA Nsight Systems (`nsys`) with CUDA tracing (CUDA graphs traced per node, so fused
graph launches still show their kernels), then groups kernel names into operation categories. Needs `nsys` and a
system where CUDA kernel tracing works: regular Linux installs do; WSL2 doesn't record kernels, and the run then
says so instead of reporting anything.
"""
import csv
import io
import re
import shutil
from pathlib import Path

from . import quantlab

# Kernel name patterns -> operation category (first match wins).
CATEGORIES = [
    ("Matrix multiply (weights)", re.compile(r"mul_mat|mmq|mmvq|mmf|gemm|gemv|cutlass|cublas|sgemm|hgemm|turing_|ampere_|sm\d+_xmma", re.I)),
    ("Attention", re.compile(r"flash_attn|fattn|soft_max|softmax", re.I)),
    ("Linear attention / state", re.compile(r"ssm|gated_delta|delta_net|rwkv|conv", re.I)),
    ("Normalization", re.compile(r"norm", re.I)),
    ("Positional (RoPE)", re.compile(r"rope", re.I)),
    ("(De)quantization", re.compile(r"quantize|dequant|convert|q8_1", re.I)),
    ("Data movement", re.compile(r"cpy|copy|concat|get_rows|set_rows|dup|transpose|memcpy|memset|pad", re.I)),
    ("Elementwise", re.compile(r"bin_bcast|add|mul|silu|gelu|glu|swiglu|scale|unary|sigmoid|exp|tanh|relu|sum|argmax|topk", re.I)),
]


def available() -> bool:
    return shutil.which("nsys") is not None


def categorize(name: str) -> str:
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return "Other"


def parse_kern_sum(text: str) -> dict:
    """nsys `cuda_gpu_kern_sum` CSV -> totals per category and the top kernels (times in ms)."""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return {"total_ms": 0.0, "categories": [], "top": []}
    def num(r, *keys):
        for k in keys:
            if k in r and r[k] not in (None, ""):
                return float(str(r[k]).replace(",", ""))
        return 0.0
    kernels = [{"name": r.get("Name", "?"), "ms": num(r, "Total Time (ns)", "Total Time") / 1e6, "calls": int(num(r, "Instances", "Count"))} for r in rows]
    total = sum(k["ms"] for k in kernels) or 1e-9
    cats: dict[str, dict] = {}
    for k in kernels:
        c = cats.setdefault(categorize(k["name"]), {"category": categorize(k["name"]), "ms": 0.0, "calls": 0, "kernels": 0})
        c["ms"] += k["ms"]; c["calls"] += k["calls"]; c["kernels"] += 1
    out_cats = sorted(({**c, "ms": round(c["ms"], 3), "share": round(c["ms"] / total, 4)} for c in cats.values()), key=lambda c: -c["ms"])
    top = sorted(kernels, key=lambda k: -k["ms"])[:12]
    return {"total_ms": round(total, 3), "categories": out_cats,
            "top": [{**k, "ms": round(k["ms"], 3), "share": round(k["ms"] / total, 4), "category": categorize(k["name"]), "name": _short(k["name"])} for k in top]}


def _short(name: str) -> str:
    """Kernel names can be huge C++ template signatures; keep the readable head."""
    name = re.sub(r"\s+", " ", name)
    head = name.split("(")[0]
    return (head if len(head) > 3 else name)[:140]


async def profile(bench: Path, model: Path, extra_args: list[str], work: Path, log: list[str], run_id: int) -> dict:
    """Two short traced llama-bench runs: prefill only (512 tokens) and decode only (64 tokens)."""
    if not available():
        return {"available": False, "note": "Nsight Systems (nsys) isn't installed"}
    work.mkdir(parents=True, exist_ok=True)
    out = {"available": True}
    for phase, pn in (("prefill", ["-p", "512", "-n", "0"]), ("decode", ["-p", "0", "-n", "64"])):
        rep = work / f"kern-{phase}"
        for f in work.glob(f"kern-{phase}*"):
            f.unlink()
        cmd = ["nsys", "profile", "-t", "cuda", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none", "-f", "true", "-o", str(rep),
               str(bench), "-m", str(model), *pn, "-r", "1", "-o", "jsonl", *extra_args]
        await quantlab._proc(cmd, log, run_id, timeout=900)
        await quantlab._proc(["nsys", "stats", "-r", "cuda_gpu_kern_sum", "-f", "csv", "--force-overwrite", "true", "-o", str(rep), str(rep) + ".nsys-rep"],
                             log, run_id, timeout=600)
        csv_file = work / f"kern-{phase}_cuda_gpu_kern_sum.csv"
        if not csv_file.is_file() or csv_file.stat().st_size == 0:
            return {"available": False, "note": "This system doesn't record CUDA kernels (for example WSL2). Kernel profiles work on a regular Linux install."}
        out[phase] = parse_kern_sum(csv_file.read_text())
        for f in work.glob(f"kern-{phase}*"):
            f.unlink()
    return out
