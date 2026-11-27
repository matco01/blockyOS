"""The `blocky` command: measure models with the blockyOS protocol from a terminal or a script.

    blocky measure Qwen3.5-0.8B-F16.gguf Qwen3.5-0.8B-Q4_K_M.gguf   # baseline + a variant, waits, prints a table
    blocky measure base.gguf --repeat 1 --csv results.csv            # with a rerun of the baseline, exported
    blocky results 13                                                # an existing experiment's table
    blocky check 13                                                  # instrument checks (exit code 1 on a failure)
    blocky status | models | experiments | projects

Talks to http://localhost:8000 unless --url or BLOCKY_URL says otherwise; another device needs its remote-access token
(--token or BLOCKY_TOKEN). Exit codes: 0 done, 1 a run or check failed, 2 usage or connection error.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

from blocky import BlockyError, Client, Experiment

POLL_S = 10.0


def _fmt(v, sd=None, digits=2, scale=1.0, unit=""):
    if v is None:
        return "–"
    s = f"{v * scale:.{digits}f}{unit}"
    return f"{s} ± {sd * scale:.{digits}f}" if sd is not None else s


def rows(exp: Experiment) -> list[dict]:
    """One line per finished run: the numbers a reader needs, each with its spread or interval."""
    out = []
    for r in exp.runs():
        if r["status"] != "done":
            continue
        x, st = r["results"], r.get("stats") or {}
        m, s = x.get("mmlu") or {}, x.get("speed") or {}
        ci = m.get("ci95") or [None, None]
        diff = (st.get("mmlu_diff") or {}).get("diff")
        p = st.get("mmlu_p_holm", st.get("mmlu_p"))
        out.append({
            "run": r["id"], "version": r["label"],
            "mmlu_pro": _fmt(m.get("acc"), None, 1, 100, "%") + (f" [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]" if ci[0] is not None else ""),
            "vs_baseline": (f"{diff * 100:+.2f} pts" + (f" (p{'<0.001' if p < 0.001 else f'={p:.2g}'})" if p is not None else "")) if diff is not None else "",
            "decode_tok_s": _fmt(s.get("tg"), s.get("tg_sd"), 1), "prefill_tok_s": _fmt(s.get("pp"), s.get("pp_sd"), 0),
            "perplexity": _fmt(x.get("ppl"), x.get("ppl_sd"), 3), "kld": _fmt(x.get("kld"), None, 4),
            "same_top": _fmt(x.get("same_top"), None, 1, unit="%"),
        })
    return out


def table(data: list[dict]) -> str:
    if not data:
        return "No finished runs yet."
    cols = list(data[0])
    width = {c: max(len(c), *(len(str(r[c])) for r in data)) for c in cols}
    line = "  ".join(c.ljust(width[c]) for c in cols)
    return "\n".join([line, "  ".join("-" * width[c] for c in cols)] + ["  ".join(str(r[c]).ljust(width[c]) for c in cols) for r in data])


def wait(exp: Experiment, out=sys.stderr, poll: float = POLL_S) -> bool:
    """Until nothing is queued or running. Returns True when every run finished without an error."""
    last = None
    while True:
        rs = exp.runs()
        busy = [r for r in rs if r["status"] in ("queued", "running")]
        running = next((r for r in busy if r["status"] == "running"), None)
        now = (running["label"] + (f" · {running['phase']}" if running and running.get("phase") else "") if running else None, len(busy))
        if now != last:
            print(f"  {now[0] or 'waiting for the lab'} · {now[1]} left" if busy else "  all runs finished", file=out, flush=True)
            last = now
        if not busy:
            return all(r["status"] == "done" for r in rs)
        status = exp.c.status()
        if status.get("gpu_pause"):
            print(f"  GPU work is paused: {status['gpu_pause']['reason']}. Resume it in Settings → GPU safety.", file=out, flush=True)
        time.sleep(poll)


def _project_id(c: Client, name: str) -> int:
    for p in c.get("/api/projects")["projects"]:
        if p["name"].lower() == name.lower():
            return p["id"]
    return c.post("/api/projects", {"name": name})["id"]


def cmd_measure(c: Client, a) -> int:
    models = {m["file"] for m in c.models()}
    missing = [m for m in a.models if m not in models]
    if missing:
        print(f"Not on the device: {', '.join(missing)}. See `blocky models`.", file=sys.stderr)
        return 2
    body = {"name": a.name or a.models[0].removesuffix(".gguf"), "model": a.models[0]}
    if a.project:
        body["project_id"] = _project_id(c, a.project)
    exp = Experiment(c, c.post("/api/experiments", body)["id"])
    if a.ppl_chunks:
        c._req("PUT", f"/api/experiments/{exp.id}/protocol", {"ppl_chunks": a.ppl_chunks})
    base = c.post(f"/api/experiments/{exp.id}/runs", {"config": {"model": a.models[0]}})["id"]
    for _ in range(a.repeat):
        c.post(f"/api/experiments/{exp.id}/runs", {"config": {"model": a.models[0]}, "parent_id": base})
    for m in a.models[1:]:
        c.post(f"/api/experiments/{exp.id}/runs", {"config": {"model": m}, "parent_id": base})
    total = 1 + a.repeat + len(a.models) - 1
    print(f"Experiment {exp.id}: {total} run{'s' if total != 1 else ''} queued. Open it at {c.url}/#/experiments?exp={exp.id}", file=sys.stderr)
    if a.no_wait:
        print(exp.id)
        return 0
    ok = wait(exp, poll=a.poll)
    return _report(exp, a) or (0 if ok else 1)


def _report(exp: Experiment, a) -> int:
    data = rows(exp)
    if getattr(a, "json", False):
        print(json.dumps(data, indent=1))
    else:
        print(table(data))
    if getattr(a, "csv", None):
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            raw = exp.results()
            items = raw.reset_index().to_dict("records") if hasattr(raw, "reset_index") else raw
            if items:
                w = csv.DictWriter(f, fieldnames=sorted({k for r in items for k in r}))
                w.writeheader()
                w.writerows(items)
        print(f"Every metric of every run: {a.csv}", file=sys.stderr)
    return 0


def cmd_results(c: Client, a) -> int:
    return _report(c.experiment(int(a.experiment) if a.experiment.isdigit() else a.experiment), a)


def cmd_check(c: Client, a) -> int:
    exp = c.experiment(int(a.experiment) if a.experiment.isdigit() else a.experiment)
    report = c.get(f"/api/experiments/{exp.id}/instrument-check")
    if a.json:
        print(json.dumps(report, indent=1))
    else:
        for ch in report["checks"]:
            print(f"[{ch['status']:<7}] {ch['id']:<16} {ch['detail']}")
    return 1 if any(ch["status"] == "fail" for ch in report["checks"]) else 0


def cmd_status(c: Client, a) -> int:
    s = c.status()
    g = c.get("/api/gpu-safety")
    print(f"Chat model: {s.get('model') or 'not loaded'}{' (ready)' if s.get('model_ready') else ''}")
    print(f"Lab: {s.get('busy') or 'idle'}")
    print(f"GPU: {g['temp_c'] if g.get('temp_c') is not None else '–'} °C · safety stop at {g['limit_c']} °C"
          + (f" · PAUSED: {g['paused']['reason']}" if g.get("paused") else ""))
    return 0


def cmd_models(c: Client, a) -> int:
    print(table([{"file": m["file"], "size": f"{m['size_bytes'] / 2**30:.2f} GB", "quant": m.get("quant") or ""} for m in c.models()]))
    return 0


def cmd_experiments(c: Client, a) -> int:
    print(table([{"id": e["id"], "name": e["name"], "model": e["model"], "runs": e.get("runs", "")} for e in c.experiments()]))
    return 0


def cmd_projects(c: Client, a) -> int:
    ps = c.get("/api/projects")["projects"]
    print(table([{"id": p["id"], "name": p["name"], "models": len(p["experiments"]), "versions": sum(e["done"] for e in p["experiments"]),
                  "goal": p["goal"][:60]} for p in ps]))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="blocky", description="Measure AI models with the blockyOS protocol.")
    p.add_argument("--url", default=os.environ.get("BLOCKY_URL", "http://localhost:8000"), help="the blockyOS device (default: this one)")
    p.add_argument("--token", default=os.environ.get("BLOCKY_TOKEN"), help="remote-access token of another device")
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="measure a baseline model and optional variants, wait, print the results")
    m.add_argument("models", nargs="+", help="model files on the device: the first is the baseline")
    m.add_argument("--name", help="experiment name")
    m.add_argument("--project", help="put the experiment in this project (created if needed)")
    m.add_argument("--repeat", type=int, default=0, help="reruns of the baseline (for repeatability)")
    m.add_argument("--ppl-chunks", type=int, choices=(32, 64, 128, 256), help="perplexity/KLD sample size")
    m.add_argument("--no-wait", action="store_true", help="queue and print the experiment id")
    m.add_argument("--poll", type=float, default=POLL_S, help=argparse.SUPPRESS)
    for name, fn_help in (("results", "an experiment's results"), ("check", "instrument checks of an experiment")):
        s = sub.add_parser(name, help=fn_help)
        s.add_argument("experiment", help="id or (part of the) name")
    for s in (m, sub.choices["results"]):
        s.add_argument("--csv", help="also write every metric of every run to this CSV file")
    for s in (m, sub.choices["results"], sub.choices["check"]):
        s.add_argument("--json", action="store_true", help="print JSON instead of a table")
    for name, fn_help in (("status", "the device: chat model, lab, GPU safety"), ("models", "model files on the device"),
                          ("experiments", "all experiments"), ("projects", "all projects")):
        sub.add_parser(name, help=fn_help)
    return p


COMMANDS = {"measure": cmd_measure, "results": cmd_results, "check": cmd_check, "status": cmd_status, "models": cmd_models,
            "experiments": cmd_experiments, "projects": cmd_projects}


def main(argv: list[str] | None = None, client: Client | None = None) -> int:
    a = parser().parse_args(argv)
    c = client or Client(a.url, token=a.token)
    try:
        return COMMANDS[a.command](c, a)
    except BlockyError as e:
        print(f"blocky: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"blocky: can't reach {c.url} ({e}). Is blockyOS running? Another device needs --url and --token.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped waiting; the runs continue on the device.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
