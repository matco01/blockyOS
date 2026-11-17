#!/usr/bin/env python3
"""Re-run the experiment in this bundle on a blockyOS device and compare the results with the original.

Usage: python reproduce.py [--url http://localhost:8000] [--token TOKEN] [--dry-run]

Needs only the Python standard library. The device must have the same model files (same SHA-256) and, for custom
engines, an engine with the same name. Speed and energy depend on the hardware; accuracy, KL divergence and perplexity
should match closely on any device with the same model, engine commit and protocol.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = [  # name, path in results, path of an interval or spread in results (or None)
    ("MMLU-Pro accuracy", ("mmlu", "acc"), ("mmlu", "ci95")),
    ("KL divergence", ("kld",), ("kld_sd",)),
    ("Same top token %", ("same_top",), ("same_top_sd",)),
    ("Perplexity", ("ppl",), ("ppl_sd",)),
    ("Long-context retrieval", ("needle", "acc"), None),
    ("Decode tok/s (llama-bench)", ("speed", "tg"), ("speed", "tg_sd")),
    ("Prefill tok/s (llama-bench)", ("speed", "pp"), ("speed", "pp_sd")),
    ("Serving output tok/s", ("serving", "tg"), ("serving", "tg_sd")),
]
HARDWARE_BOUND = {"Decode tok/s (llama-bench)", "Prefill tok/s (llama-bench)", "Serving output tok/s"}


def get(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


class Api:
    def __init__(self, url, token):
        self.url, self.token = url.rstrip("/"), token

    def call(self, method, path, body=None):
        req = urllib.request.Request(self.url + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.token}"} if self.token else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            sys.exit(f"{method} {path}: {e.code} {e.read().decode(errors='replace')[:300]}")


def plan(exp):
    """Runs to repeat, in the original order, with parents mapped (baseline first)."""
    runs = sorted((r for r in exp["runs_list"] if r["status"] == "done" and not (r.get("results") or {}).get("imported_from")),
                  key=lambda r: r["id"])
    return [{"original": r["id"], "parent": r.get("parent_id"), "config": r["config"], "label": r.get("label") or ""} for r in runs]


def compare(orig, new):
    rows = []
    po, pn = orig.get("provenance") or {}, new.get("provenance") or {}
    for name, path in (("protocol", ("protocol",)), ("model SHA-256", ("model", "sha256")), ("engine commit", ("llama_cpp", "commit")),
                       ("GPU", ("gpu", "name"))):
        if get(po, path) != get(pn, path):
            rows.append({"condition": name, "original": get(po, path), "reproduced": get(pn, path)})
    out = []
    for name, path, spread in METRICS:
        a, b = get(orig, path), get(new, path)
        if a is None and b is None:
            continue
        s = get(new, spread) if spread else None
        inside = None
        if isinstance(s, list) and len(s) == 2 and a is not None:
            inside = s[0] <= a <= s[1]
        elif isinstance(s, (int, float)) and a is not None and b is not None:
            inside = abs(a - b) <= 2 * s + 1e-12  # within two of the reproduction's reported spread
        out.append({"metric": name, "original": a, "reproduced": b, "within_interval": inside,
                    "hardware_dependent": name in HARDWARE_BOUND})
    return rows, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--token", default="")
    ap.add_argument("--dry-run", action="store_true", help="show what would be queued, then stop")
    ap.add_argument("--poll", type=float, default=30)
    a = ap.parse_args()
    exp = json.loads((HERE / "experiment.json").read_text())
    steps = plan(exp)
    if not steps:
        sys.exit("The bundle has no completed runs to reproduce")
    print(f"{len(steps)} run(s) to reproduce from '{exp['name']}' (model {exp['model']})")
    for s in steps:
        print(f"  #{s['original']} {s['label']}  parent={s['parent']}")
    if a.dry_run:
        return
    api = Api(a.url, a.token)
    new_exp = api.call("POST", "/api/experiments", {"name": f"{exp['name']} (reproduction)"[:80], "model": exp["model"]})["id"]
    api.call("PUT", f"/api/experiments/{new_exp}/protocol", exp.get("protocol") or {})
    mapping = {}
    for s in steps:
        parent = mapping.get(s["parent"])
        mapping[s["original"]] = api.call("POST", f"/api/experiments/{new_exp}/runs", {"config": s["config"], "parent_id": parent})["id"]
    print(f"Queued in experiment {new_exp}; waiting (Ctrl+C stops waiting, not the runs)")
    while True:
        e = api.call("GET", f"/api/experiments/{new_exp}")
        states = {r["id"]: r["status"] for r in e["runs_list"]}
        pending = [i for i in mapping.values() if states.get(i) in ("queued", "running")]
        print(f"  {len(mapping) - len(pending)} of {len(mapping)} finished", flush=True)
        if not pending:
            break
        time.sleep(a.poll)
    by_new = {r["id"]: r for r in e["runs_list"]}
    by_orig = {r["id"]: r for r in exp["runs_list"]}
    report = {"original_experiment": exp["id"], "reproduction_experiment": new_exp, "url": a.url, "runs": []}
    for orig_id, new_id in mapping.items():
        new = by_new[new_id]
        conditions, metrics = compare(by_orig[orig_id].get("results") or {}, new.get("results") or {})
        report["runs"].append({"original": orig_id, "reproduced": new_id, "status": new["status"], "error": new.get("error"),
                               "condition_differences": conditions, "metrics": metrics})
        print(f"\n#{orig_id} -> #{new_id} ({new['status']})")
        for c in conditions:
            print(f"  ! {c['condition']} differs: {c['original']} vs {c['reproduced']}")
        for m in metrics:
            flag = "" if m["within_interval"] is None else ("ok" if m["within_interval"] else "OUTSIDE")
            print(f"  {m['metric']:<30} {m['original']!s:>12} {m['reproduced']!s:>12}  {flag}{'  (hardware-dependent)' if m['hardware_dependent'] else ''}")
    (HERE / "reproduction-report.json").write_text(json.dumps(report, indent=1))
    print("\nWrote reproduction-report.json")


if __name__ == "__main__":
    main()
