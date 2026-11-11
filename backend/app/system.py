"""Hardware monitoring: CPU, RAM, disk, and NVIDIA GPU (if present)."""
import shutil
import subprocess

import psutil

from . import config

_GPU_FIELDS = [
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
]


def _num(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None  # nvidia-smi prints "[N/A]" for unsupported fields


def gpu_stats() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, util, used, total, temp, power = [p.strip() for p in line.split(",")]
        gpus.append({
            "name": name,
            "util_percent": _num(util),
            "vram_used_mb": _num(used),
            "vram_total_mb": _num(total),
            "temp_c": _num(temp),
            "power_w": _num(power),
        })
    return gpus


def snapshot() -> dict:
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(config.DATA_DIR) if config.DATA_DIR.exists() else "/")
    return {
        "cpu": {
            "percent": psutil.cpu_percent(interval=None),
            "cores": psutil.cpu_count(logical=True),
        },
        "ram": {"used_mb": (mem.total - mem.available) / 2**20, "total_mb": mem.total / 2**20},
        "disk": {"used_gb": disk.used / 2**30, "total_gb": disk.total / 2**30},
        "gpus": gpu_stats(),
    }
