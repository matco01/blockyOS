"""Persisted alternating A/B repeats. Conclusions concern one preselected metric only."""
import json
import math
import statistics
import time

from . import db

METHOD = "paired-median-sign/1"
METRICS = {
    "tg": ("Decode speed", "tok/s", ("speed", "tg"), 1),
    "pp": ("Prefill speed", "tok/s", ("speed", "pp"), 1),
    "srv_tg": ("Serving output throughput", "tok/s", ("serving", "tg"), 1),
    "jpt": ("Energy per output token", "J", (), -1),
}
SCHEMA = """
CREATE TABLE IF NOT EXISTS validations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    plan TEXT NOT NULL
);
"""


def get(d, path):
    for key in path:
        d = d.get(key) if isinstance(d, dict) else None
    return d


def create(exp_id: int, body: dict) -> dict:
    from . import profile
    exp = profile.get_experiment(exp_id)
    if not exp:
        raise ValueError("Experiment not found")
    pair_count = int(body.get("pairs", 6))
    metric = body.get("metric", "tg")
    threshold = float(body.get("min_effect_pct", 3))
    if not 3 <= pair_count <= 12 or metric not in METRICS:
        raise ValueError("Choose 3–12 pairs and a supported measured metric")
    if not math.isfinite(threshold) or not 0 <= threshold <= 100:
        raise ValueError("Minimum improvement must be between 0 and 100 percent")
    sources = []
    for key in ("reference_id", "candidate_id"):
        source = next((r for r in exp["runs_list"] if r["id"] == body.get(key)), None)
        if not source or source["status"] != "done" or source["results"].get("imported_from"):
            raise ValueError("Choose two completed local runs in this experiment")
        sources.append(source)
    if sources[0]["id"] == sources[1]["id"] or sources[0]["config"] == sources[1]["config"]:
        raise ValueError("Choose two different setups")
    # Runs are new measurements; historical source runs are never counted as repetitions.
    configs = [profile.normalize_config(exp, r["config"]) for r in sources]
    if configs[0] == configs[1]:
        raise ValueError("Choose two different setups")
    baseline = next(r for r in exp["runs_list"] if r["status"] == "done")
    if get(baseline["results"], ("provenance", "protocol")) != profile.PROTOCOL:
        raise ValueError("Create a baseline with the current measurement protocol before validating")
    if metric in ("tg", "pp") and any(c.get("engine") == "vllm" for c in configs):
        raise ValueError("Use serving output throughput when comparing vLLM")
    plan = {"method": METHOD, "protocol": profile.PROTOCOL, "tests": exp.get("protocol") or {},
            "pairs": pair_count, "metric": metric, "min_effect_pct": threshold,
            "sources": [{"id": r["id"], "label": r["label"], "config": cfg} for r, cfg in zip(sources, configs)], "run_ids": []}
    with db.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        vid = c.execute("INSERT INTO validations (experiment_id, created_at, plan) VALUES (?, ?, ?)",
                        (exp_id, time.time(), json.dumps(plan))).lastrowid
        for pair in range(pair_count):
            ids = []
            for arm, (source, cfg) in enumerate(zip(sources, configs)):
                mark = {"id": vid, "pair": pair + 1, "arm": "reference" if arm == 0 else "candidate"}
                label = f"Validate #{vid} · pair {pair + 1} · {'A' if arm == 0 else 'B'}"
                ids.append(c.execute("INSERT INTO exp_runs (experiment_id, created_at, status, label, config, results, parent_id, validation, protocol_snapshot) "
                                     "VALUES (?, ?, 'queued', ?, ?, '{}', ?, ?, ?)",
                                     (exp_id, time.time(), label, json.dumps(cfg), source["id"], json.dumps(mark),
                                      json.dumps({"version": profile.PROTOCOL, "tests": plan["tests"]}))).lastrowid)
            plan["run_ids"].append(ids)
        c.execute("UPDATE validations SET plan = ? WHERE id = ?", (json.dumps(plan), vid))
    return {"id": vid, "run_ids": [rid for pair in plan["run_ids"] for rid in pair]}


def paired_stats(changes: list[float]) -> dict:
    """Exact sign test and distribution-free 95% interval for the median paired percent change.

    Order-statistic coverage follows Binomial(n, 1/2). With fewer than six pairs the interval is unbounded.
    Assumes independent pairs and a continuous common median; ties make the interval conservative.
    """
    xs = sorted(changes)
    n = len(xs)
    if not n:
        return {"n": 0, "median_pct": None, "sd_pct": None, "range_pct": None, "ci95_pct": None, "p": None}
    positive, negative = sum(x > 0 for x in xs), sum(x < 0 for x in xs)
    nonzero = positive + negative
    p = min(1.0, 2 * sum(math.comb(nonzero, i) for i in range(min(positive, negative) + 1)) / 2**nonzero) if nonzero else 1.0
    k = 0
    for candidate in range(1, n // 2 + 1):
        tail = 2 * sum(math.comb(n, i) for i in range(candidate)) / 2**n
        if tail <= 0.05:
            k = candidate
    return {"n": n, "median_pct": statistics.median(xs), "sd_pct": statistics.stdev(xs) if n > 1 else None,
            "range_pct": [xs[0], xs[-1]], "ci95_pct": [xs[k - 1], xs[n - k]] if k else None, "p": p}


def summarize(plan: dict, runs: list[dict]) -> dict:
    by_id = {r["id"]: r for r in runs}
    title, unit, path, direction = METRICS[plan["metric"]]
    pairs, changes, problems = [], [], []
    fingerprints = [None, None]
    common = None
    pending = False
    for index, ids in enumerate(plan["run_ids"]):
        values, states = [], []
        for arm, rid in enumerate(ids):
            run = by_id.get(rid)
            state = run["status"] if run else "missing"
            states.append(state)
            pending |= state in ("queued", "running")
            value = None
            if state == "done":
                res = run["results"]
                prov = res.get("provenance") or {}
                if prov.get("protocol") != plan["protocol"] or run["config"] != plan["sources"][arm]["config"]:
                    problems.append("A repeat used a different protocol or setup")
                if (not get(prov, ("model", "sha256")) or not get(prov, ("host", "device"))
                        or not prov.get("data") or not get(prov, ("llama_cpp", "version")) and not get(prov, ("llama_cpp", "commit"))):
                    problems.append("Provenance is incomplete")
                # Hardware and evaluation inputs must be constant. Engine/model may differ between arms, not within one arm.
                shared = [prov.get(k) for k in ("protocol", "data", "host", "gpu", "extra_tests")]
                if common is None:
                    common = shared
                elif shared != common:
                    problems.append("Hardware, datasets or test settings changed between repeats")
                fingerprint = [prov.get(k) for k in ("model", "llama_cpp", "framework")]
                if fingerprints[arm] is None:
                    fingerprints[arm] = fingerprint
                elif fingerprints[arm] != fingerprint:
                    problems.append("Model weights or engine changed within a setup")
                if plan["metric"] == "jpt":
                    energy = res.get("energy") or {}
                    value = get(energy, ("sources", energy.get("primary"), "j_per_token"))
                    source = energy.get("primary")
                    if source != plan.get("_energy_source", source):
                        problems.append("Power sources changed between repeats")
                    plan = {**plan, "_energy_source": source}
                else:
                    value = get(res, path)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    value = None
                    problems.append("A repeat has no valid measurement for the selected metric")
            elif state not in ("queued", "running"):
                problems.append("A planned repeat failed, was cancelled or was deleted")
            values.append(value)
        change = direction * (values[1] / values[0] - 1) * 100 if all(v is not None for v in values) else None
        if change is not None:
            changes.append(change)
        pairs.append({"pair": index + 1, "run_ids": ids, "states": states, "reference": values[0], "candidate": values[1], "improvement_pct": change})
    stats = paired_stats(changes)
    conclusion = "pending" if pending else "insufficient"
    complete = len(changes) == plan["pairs"] and not problems and not pending
    interval = stats["ci95_pct"]
    if complete and interval and stats["p"] < 0.05:
        if interval[0] > plan["min_effect_pct"]:
            conclusion = "candidate_better"
        elif interval[1] < -plan["min_effect_pct"]:
            conclusion = "reference_better"
    return {"metric_label": title, "unit": unit, "pairs": pairs, "stats": stats, "conclusion": conclusion,
            "problems": sorted(set(problems)), "complete": complete,
            "note": "Only the preselected metric is tested. Repeated pairs assume stable conditions; fixed A/B order cannot rule out order effects. This does not establish overall model quality or equivalence."}


def list_batches(exp_id: int, runs: list[dict]) -> list[dict]:
    with db.connect() as c:
        rows = list(c.execute("SELECT * FROM validations WHERE experiment_id = ? ORDER BY id DESC", (exp_id,)))
    return [{"id": r["id"], "created_at": r["created_at"], "plan": json.loads(r["plan"]),
             "summary": summarize(json.loads(r["plan"]), runs)} for r in rows]


def cancel(exp_id: int, vid: int) -> None:
    from . import profile
    with db.connect() as c:
        row = c.execute("SELECT plan FROM validations WHERE id = ? AND experiment_id = ?", (vid, exp_id)).fetchone()
    if not row:
        raise ValueError("Validation not found")
    for pair in json.loads(row["plan"])["run_ids"]:
        for rid in pair:
            profile.cancel(rid)
