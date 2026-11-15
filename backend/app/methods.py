"""Registered research evaluators as first-class experiment configurations.

Registration pins a successful job's source and launch settings. No arbitrary events are reinterpreted as native
profile measurements: shared metrics retain their own evaluation contracts.
"""
import hashlib
import json
import math
import time
from pathlib import Path

from . import db, research_jobs

PREFIX = "method:"
SCHEMA = """
CREATE TABLE IF NOT EXISTS research_methods (
 id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, created_at REAL NOT NULL,
 source_job_id INTEGER NOT NULL REFERENCES research_jobs(id), definition TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS method_profile_jobs (
 run_id INTEGER PRIMARY KEY REFERENCES exp_runs(id) ON DELETE CASCADE,
 job_id INTEGER NOT NULL REFERENCES research_jobs(id)
);
"""


def init():
    with db.connect() as c:
        c.executescript(research_jobs.SCHEMA)
        c.executescript(SCHEMA)


def get(mid):
    with db.connect() as c:
        row = c.execute("SELECT * FROM research_methods WHERE id=?", (mid,)).fetchone()
    if not row:
        raise ValueError("Registered method not found")
    return {**dict(row), "definition": json.loads(row["definition"])}


def list_methods():
    with db.connect() as c:
        ids = [r[0] for r in c.execute("SELECT id FROM research_methods ORDER BY id DESC")]
    return [get(mid) for mid in ids]


def register(jid, name):
    job = research_jobs.job(jid)
    metrics = [e for e in job["events"] if e.get("type") == "metric" and isinstance(e.get("name"), str)]
    if job["status"] != "done" or not metrics:
        raise ValueError("Register a successfully completed job with reported evaluation metrics")
    # Validate contracts using the same checks as the comparison view.
    comparison = research_jobs.compare([jid, jid])
    if any(not m["compatible"] for m in comparison["metrics"]):
        raise ValueError("Fix incomplete metric contracts before registering this evaluator")
    source = Path(job["directory"]) / "source"
    for relative, expected in job["provenance"].get("files", {}).items():
        path = research_jobs.confined(source, relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("The tested source changed after snapshotting; run the updated code before registering")
    spec = {k: v for k, v in job["spec"].items() if k not in ("parent_id", "profile_run_id")}
    latest = {e["name"]: e for e in metrics}
    definition = {"project_id": job["project_id"], "spec": spec, "source_sha256": job["provenance"]["snapshot_sha256"],
                  "capabilities": [{k: e.get(k) for k in ("name", "unit", "protocol", "task", "dataset_sha256", "scope", "kind", "better", "tokenizer_sha256")} for e in latest.values()],
                  "note": "Capabilities are observed evaluator declarations. External model/data files and Python environments must still be supplied and versioned by the researcher."}
    with db.connect() as c:
        mid = c.execute("INSERT INTO research_methods (name,created_at,source_job_id,definition) VALUES (?,?,?,?)",
                        (str(name or "").strip()[:100] or f"Research method from job {jid}", time.time(), jid, json.dumps(definition))).lastrowid
    return get(mid)


def normalize(model, supplied):
    mid = int(model.removeprefix(PREFIX))
    method = get(mid)
    definition = method["definition"]
    unsupported = {"ctx": 0, "kv": "unsupported", "ngl": "unsupported", "fa": "unsupported",
                   "threads": None, "vram_limit_mb": None, "preset": None, "cores": None, "ram_limit_mb": None}
    for key, expected in unsupported.items():
        if key in supplied and supplied[key] not in (None, "", expected):
            raise ValueError(f"Registered evaluator does not implement native setting '{key}'; use its method_config instead")
    cfg = supplied.get("method_config", definition["spec"]["config"])
    if not isinstance(cfg, dict):
        raise ValueError("Method settings must be a JSON object")
    json.dumps(cfg, allow_nan=False)
    return {"model": f"method:{mid}", "engine": "research", "method_config": cfg,
            "ctx": 0, "kv": "unsupported", "ngl": "unsupported", "fa": "unsupported", "threads": None, "vram_limit_mb": None,
            "device": "gpu" if definition["spec"]["gpu"] else "cpu", "cores": None, "ram_limit_mb": None, "preset": None}


def queue_profile(exp_id, body, cfg):
    from . import profile
    exp = profile.get_experiment(exp_id)
    parent_id = body.get("parent_id")
    if parent_id is not None and not any(r["id"] == int(parent_id) for r in exp["runs_list"]):
        raise ValueError("Parent run is not in this experiment")
    method = get(int(cfg["model"].removeprefix(PREFIX)))
    source_job = research_jobs.job(method["source_job_id"])
    for relative, expected in source_job["provenance"].get("files", {}).items():
        path = research_jobs.confined(Path(source_job["directory"]) / "source", relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("The registered source snapshot changed; register a new tested version")
    spec = {**method["definition"]["spec"], "config": cfg["method_config"]}
    label = str(body.get("label") or method["name"])[:100]
    with db.connect() as c:
        rid = c.execute("INSERT INTO exp_runs (experiment_id,created_at,status,label,config,results,parent_id) VALUES (?,?,'preparing',?,?,'{}',?)",
                        (exp_id, time.time(), label, json.dumps(cfg), parent_id)).lastrowid
    try:
        jid = research_jobs.queue(method["definition"]["project_id"], spec, source_root=Path(source_job["directory"]) / "source", profile_run_id=rid)["id"]
        with db.connect() as c:
            c.execute("INSERT INTO method_profile_jobs (run_id,job_id) VALUES (?,?)", (rid, jid))
            c.execute("UPDATE exp_runs SET status='queued',results=? WHERE id=?", (json.dumps({"research_job_id": jid, "method": {"id": method["id"], "name": method["name"]}}), rid))
    except Exception as e:
        profile._save(rid, status="error", error=str(e)[:500])
        raise
    return rid


async def run_profile(rec, llm):
    from . import profile
    rid = rec["id"]
    with db.connect() as c:
        row = c.execute("SELECT job_id FROM method_profile_jobs WHERE run_id=?", (rid,)).fetchone()
    if not row:
        profile._save(rid, status="error", error="The registered method's queued source snapshot is missing")
        return
    jid = row[0]
    profile._save(rid, status="running", phase="research")
    try:
        await research_jobs.execute(jid, llm)
    finally:
        job = research_jobs.job(jid)
        metrics = {e["name"]: e for e in job["events"] if e.get("type") == "metric" and isinstance(e.get("name"), str)
                   and isinstance(e.get("value"), (float, int)) and not isinstance(e["value"], bool) and math.isfinite(e["value"])
                   and not contract_errors(e, e)}
        result = {**rec["results"], "research_metrics": metrics, "research_provenance": job["provenance"],
                  "research_job_id": jid, "research_artifacts": [e for e in job["events"] if e.get("type") == "artifact"],
                  "config": rec["config"], "fidelity_note": "Registered evaluator: native llama-bench, MMLU and KLD phases are unavailable unless explicitly reported through a compatible evaluator contract."}
        error = job["error"]
        status = job["status"]
        if status == "done" and not metrics:
            status, error = "error", "Evaluator completed without valid metrics"
        profile._save(rid, status=status, results=result, error=error, phase=None, log=job["log"])
        profile._cancel.discard(rid)


def contract_errors(a, b):
    if not a or not b:
        return ["A repeat did not report the selected metric"]
    keys = ("unit", "protocol", "task", "dataset_sha256", "scope", "kind", "better", "tokenizer_sha256")
    errors = [f"Metric contract changed: {k}" for k in keys if a.get(k) != b.get(k)]
    if any(not a.get(k) for k in keys if k not in ("better", "tokenizer_sha256")):
        errors.append("Incomplete metric contract")
    return errors


def create_validation(exp_id, body):
    from . import profile
    import math
    exp = profile.get_experiment(exp_id)
    count = int(body.get("pairs", 6))
    minimum = float(body.get("min_effect_pct", 3))
    if not exp or not 3 <= count <= 12 or not math.isfinite(minimum) or not 0 <= minimum <= 100:
        raise ValueError("Choose an experiment, 3–12 pairs, and a finite improvement threshold between 0 and 100")
    sources = [next((r for r in exp["runs_list"] if r["id"] == body.get(key)), None) for key in ("reference_id", "candidate_id")]
    if any(not r or r["status"] != "done" or r["config"].get("engine") != "research" or r["results"].get("imported_from") for r in sources):
        raise ValueError("Shared-metric validation requires two completed local registered-method runs")
    if sources[0]["config"] == sources[1]["config"]:
        raise ValueError("Choose two different setups")
    name = body["metric"].removeprefix("research:")
    metrics = [r["results"].get("research_metrics", {}).get(name) for r in sources]
    errors = contract_errors(*metrics)
    if errors or metrics[0].get("kind") != "measured" or metrics[0].get("better") not in ("higher", "lower"):
        raise ValueError("Choose a measured shared metric with matching contracts and an improvement direction")
    plan = {"method": "paired-median-sign/1", "research_metric": name, "contract": metrics[0], "pairs": count,
            "inference_allowed": metrics[0]["unit"] in ("byte/s", "token/s", "s", "ms", "J", "J/byte", "J/token", "W"),
            "metric": body["metric"], "min_effect_pct": minimum, "sources": [{"id": r["id"], "config": r["config"], "label": r["label"]} for r in sources], "run_ids": []}
    with db.connect() as c:
        vid = c.execute("INSERT INTO validations (experiment_id,created_at,plan) VALUES (?,?,?)", (exp_id, time.time(), json.dumps(plan))).lastrowid
    created_ids = []
    try:
        for pair in range(count):
            ids = []
            for arm, source in enumerate(sources):
                rid = profile.create_run(exp_id, {"parent_id": source["id"], "label": f"Validate #{vid} pair {pair + 1} {'A' if arm == 0 else 'B'}", "config": source["config"]})
                ids.append(rid)
                created_ids.append(rid)
            plan["run_ids"].append(ids)
            with db.connect() as c:
                c.execute("UPDATE validations SET plan=? WHERE id=?", (json.dumps(plan), vid))
    except Exception:
        for rid in created_ids:
            profile.cancel(rid)
        raise
    return {"id": vid, "run_ids": [rid for pair in plan["run_ids"] for rid in pair]}


def summarize_validation(plan, runs):
    import math
    from .validation import paired_stats
    by_id = {r["id"]: r for r in runs}
    errors, pairs, changes = [], [], []
    states = []
    hardware = None
    arm_sources = [None, None]
    for number, ids in enumerate(plan["run_ids"]):
        values, pair_states = [], []
        for arm, rid in enumerate(ids):
            r = by_id.get(rid)
            state = r["status"] if r else "missing"
            pair_states.append(state); states.append(state)
            value = None
            if state == "done":
                result = r["results"]
                metric = result.get("research_metrics", {}).get(plan["research_metric"])
                errors.extend(contract_errors(plan["contract"], metric))
                if metric:
                    value = metric.get("value")
                prov = result.get("research_provenance") or {}
                if not prov.get("hardware") or not prov.get("snapshot_sha256") or not prov.get("environment_sha256"):
                    errors.append("Incomplete repeat provenance")
                if hardware is None:
                    hardware = prov.get("hardware")
                elif hardware != prov.get("hardware"):
                    errors.append("Hardware changed between repeats")
                source = [prov.get("snapshot_sha256"), prov.get("environment_sha256"), r["config"]]
                if arm_sources[arm] is None:
                    arm_sources[arm] = source
                elif source != arm_sources[arm]:
                    errors.append("Source, environment or settings changed within a setup")
            elif state not in ("queued", "running", "preparing"):
                errors.append("A repeat failed, was cancelled or was deleted")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                if state == "done": errors.append("Missing or invalid repeat value")
                value = None
            values.append(value)
        direction = 1 if plan["contract"]["better"] == "higher" else -1
        if values[0] == 0:
            errors.append("Reference value is zero; percentage improvement is undefined")
        change = direction * (values[1] / values[0] - 1) * 100 if all(v is not None for v in values) and values[0] != 0 else None
        if change is not None: changes.append(change)
        pairs.append({"pair": number + 1, "run_ids": ids, "states": pair_states, "reference": values[0], "candidate": values[1], "improvement_pct": change})
    stats = paired_stats(changes)
    if not plan.get("inference_allowed"):
        stats["p"] = None
        stats["ci95_pct"] = None
    pending = any(s in ("queued", "running", "preparing") for s in states)
    complete = len(changes) == plan["pairs"] and not pending and not errors
    conclusion = "pending" if pending else "insufficient"
    interval = stats["ci95_pct"]
    if complete and interval and stats["p"] is not None and stats["p"] < .05:
        if interval[0] > plan["min_effect_pct"]: conclusion = "candidate_better"
        elif interval[1] < -plan["min_effect_pct"]: conclusion = "reference_better"
    return {"metric_label": plan["research_metric"], "unit": plan["contract"]["unit"], "pairs": pairs, "stats": stats,
            "problems": sorted(set(errors)), "complete": complete, "conclusion": conclusion,
            "note": ("Shared runtime/energy metric only. Fixed A/B order can confound repeated measurements. " if plan.get("inference_allowed") else
                     "Descriptive repeats only: rescoring the same fixed dataset does not supply independent quality evidence. No significance test or confidence interval is inferred. ") +
                    "Matching declared contracts do not certify evaluator correctness or overall model quality."}
