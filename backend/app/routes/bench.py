"""llama-bench runs and established benchmark datasets."""
import csv
import io
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response

from .. import benchmarks, lab
from .common import _bad, _spawn, state

router = APIRouter()


@router.get("/api/bench")
def list_bench():
    return lab.list_bench()


@router.post("/api/bench")
async def start_bench(body: dict):  # async: starts a background task on the event loop
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        run_id = lab.create_bench(body)
    except (ValueError, TypeError, KeyError) as e:
        _bad(e)
    _spawn(lab.run_bench(run_id, state.llm))
    return {"id": run_id}


@router.get("/api/bench/export")
def export_bench(format: str = "csv"):
    runs = [r for r in lab.list_bench() if r["results"]]
    if format == "json":
        return JSONResponse(runs, headers={"Content-Disposition": 'attachment; filename="blocky-benchmarks.json"'})
    cols = ["run_id", "date", "model", "test", "n_prompt", "n_gen", "n_depth", "avg_ts", "stddev_ts", "power_w",
            "tokens_per_joule", "peak_vram_mb", "n_gpu_layers", "n_cpu_moe", "type_k", "flash_attn", "n_threads",
            "gpu", "driver", "cpu", "ram_mb", "os", "wsl", "llama_cpp"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for run in runs:
        hw = run["hardware"] or {}
        gpu = (hw.get("gpus") or [{}])[0]
        for r in run["results"]:
            w.writerow([run["id"], datetime.fromtimestamp(run["created_at"]).isoformat(timespec="seconds"), run["model"],
                        r.get("test"), r.get("n_prompt"), r.get("n_gen"), r.get("n_depth"), r.get("avg_ts"), r.get("stddev_ts"),
                        r.get("power_w"), r.get("tokens_per_joule"), r.get("peak_vram_mb"), r.get("n_gpu_layers"),
                        r.get("n_cpu_moe"), r.get("type_k"), r.get("flash_attn"), r.get("n_threads"),
                        gpu.get("name"), gpu.get("driver"), hw.get("cpu"), round(hw.get("ram_mb") or 0),
                        hw.get("os"), hw.get("wsl"), hw.get("llama_cpp")])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="blocky-benchmarks.csv"'})


@router.get("/api/bench/{run_id}")
def get_bench(run_id: int):
    run = lab.get_bench(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@router.post("/api/bench/{run_id}/cancel")
def cancel_bench(run_id: int):
    lab.cancel_bench(run_id)
    return {"ok": True}


@router.delete("/api/bench/{run_id}")
def delete_bench(run_id: int):
    lab.delete_bench(run_id)
    return {"ok": True}


@router.get("/api/benchmarks")
def bench_catalog():
    return benchmarks.summary()


@router.post("/api/benchmarks/{key}/download")
async def bench_download(key: str):  # async: starts a background task on the event loop
    if key not in benchmarks.catalog():
        raise HTTPException(404, "Unknown benchmark")
    if (benchmarks.downloads.get(key) or {}).get("status") == "downloading":
        return {"ok": True}
    _spawn(benchmarks.download(key))
    return {"ok": True}


@router.delete("/api/benchmarks/{key}")
def bench_delete(key: str):
    benchmarks.delete(key)
    return {"ok": True}


@router.get("/api/hf/dataset")
async def hf_dataset_preview(repo: str):
    try:
        return await benchmarks.hf_preview(repo.strip())
    except ValueError as e:
        _bad(e)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Couldn't reach Hugging Face: {e.__class__.__name__}")


@router.post("/api/benchmarks/custom")
async def bench_add_custom(body: dict):  # async: starts the download task
    try:
        key = benchmarks.add_custom(body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(benchmarks.download(key))
    return {"key": key}
