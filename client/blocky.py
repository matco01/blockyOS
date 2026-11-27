"""blocky: a small Python client for a Blocky box.

    import blocky
    b = blocky.Client()                          # http://localhost:8000 by default
    exp = b.experiment("quantization study")     # by (part of the) name, or by id
    df = exp.results()                           # one row per run, every metric as a column (pandas, if installed)
    run = exp.branch(exp.run("Q4_K_M"), kv="q4_0")   # same setup as that run + one change, queued
    exp.wait()                                   # until the queue is empty
    exp.sweep({"kv": ["f16", "q8_0", "q4_0"], "ctx": [4096, 16384]})

    b.models()                                   # GGUF files on the box
    b.variant("Qwen3.5-9B-Q8_0.gguf", "Q3_K_M", imatrix=True, overrides={"attn_v": "q8_0"}).wait()
    b.scan("Qwen3.5-0.8B-Q8_0.gguf", mode="quantize", qtype="q2_k").wait()
    b.build_engine("https://github.com/you/llama.cpp", "my-branch", name="my-kernel").wait()

Only the standard library is required; pandas is used for tables when it's installed. The full REST API is
documented at http://<box>:8000/docs.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

__all__ = ["Client", "Experiment", "Run", "BlockyError"]
__version__ = "0.1"


class BlockyError(RuntimeError):
    pass


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif not isinstance(v, list) and k != "answers":
            out[key] = v
    return out


class Client:
    def __init__(self, url: str = "http://localhost:8000", timeout: float = 120, token: str | None = None):
        """`token`: the remote-access token, when the box is another device (Settings → Remote access)."""
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.token = token

    # ---- HTTP ----
    def _req(self, method: str, path: str, body=None, params: dict | None = None):
        q = f"?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}" if params else ""
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.token}"} if self.token else {})}
        r = urllib.request.Request(self.url + path + q, data, headers, method=method)
        try:
            with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail")
            except ValueError:
                detail = None
            raise BlockyError(f"{method} {path}: {e.code} {detail or e.reason}") from None
        return json.loads(raw) if raw else None

    def get(self, path, **params):
        return self._req("GET", path, params=params)

    def post(self, path, body=None):
        return self._req("POST", path, body if body is not None else {})

    # ---- Box ----
    def status(self) -> dict:
        return self.get("/api/status")

    def hardware(self) -> dict:
        return self.get("/api/hardware/limits")

    def models(self) -> list[dict]:
        return self.get("/api/models")["models"]

    def engines(self) -> list[dict]:
        return self.get("/api/engines")

    # ---- Experiments ----
    def experiments(self) -> list[dict]:
        return self.get("/api/experiments")

    def experiment(self, key: int | str) -> "Experiment":
        if isinstance(key, int):
            return Experiment(self, key)
        exps = self.experiments()
        exact = [e for e in exps if e["name"].lower() == str(key).lower()]
        hits = exact or [e for e in exps if str(key).lower() in e["name"].lower()]
        if not hits:
            raise BlockyError(f"No experiment matching {key!r}")
        if len(hits) > 1:
            raise BlockyError(f"{key!r} matches several experiments: " + ", ".join(f"{e['id']}: {e['name']}" for e in hits) + ". Use the id or the full name.")
        return Experiment(self, hits[0]["id"])

    def new_experiment(self, model: str, name: str | None = None) -> "Experiment":
        return Experiment(self, self.post("/api/experiments", {"model": model, "name": name or model.removesuffix(".gguf")})["id"])

    # ---- Weights, scans, engines ----
    def variant(self, source: str, qtype: str, *, imatrix: bool = False, overrides: dict | None = None, output_type: str | None = None,
                embedding_type: str | None = None, prune: list[int] | None = None, name: str | None = None) -> "Job":
        jid = self.post("/api/weights/variants", {"source": source, "qtype": qtype, "imatrix": imatrix, "overrides": overrides or {},
                                                   "output_type": output_type, "embedding_type": embedding_type, "prune": prune or [], "name": name})["id"]
        return Job(lambda: next((j for j in self.get("/api/weights/jobs") if j["id"] == jid), {}), ("queued", "running"))

    def scan(self, model: str, mode: str = "quantize", qtype: str = "q2_k") -> "Job":
        sid = self.post("/api/scans", {"model": model, "mode": mode, "qtype": qtype})["id"]
        return Job(lambda: next((s for s in self.get("/api/scans") if s["id"] == sid), {}), ("queued", "running"))

    def build_engine(self, repo: str, ref: str = "master", name: str | None = None) -> "Job":
        eid = self.post("/api/engines", {"repo": repo, "ref": ref, "name": name})["id"]
        return Job(lambda: next((e for e in self.engines() if e["id"] == eid), {}), ("queued", "building"))


class Job:
    """Something running on the box (a weights build, a scan, an engine build)."""

    def __init__(self, fetch, active: tuple):
        self._fetch, self._active = fetch, active

    def info(self) -> dict:
        return self._fetch()

    def wait(self, poll: float = 10, verbose: bool = True) -> dict:
        last = None
        while True:
            i = self.info()
            st = (i.get("status"), i.get("step"))
            if verbose and st != last:
                print(time.strftime("%H:%M:%S"), *[x for x in st if x])
                last = st
            if i.get("status") not in self._active:
                return i
            time.sleep(poll)


class Run(dict):
    """One profiled configuration. A dict (the raw API record) with conveniences."""

    @property
    def metrics(self) -> dict:
        return _flat({k: v for k, v in self.get("results", {}).items() if k not in ("series", "config", "commands")})


class Experiment:
    def __init__(self, client: Client, exp_id: int):
        self.c, self.id = client, exp_id

    def info(self) -> dict:
        return self.c.get(f"/api/experiments/{self.id}")

    def runs(self) -> list[Run]:
        return [Run(r) for r in self.info()["runs_list"]]

    def run(self, key: int | str) -> Run:
        for r in self.runs():
            if r["id"] == key or (isinstance(key, str) and r["label"] == key):
                return r
        raise BlockyError(f"No run {key!r}")

    def baseline(self) -> Run:
        return next(r for r in self.runs() if r["status"] == "done")

    def results(self, only_done: bool = True):
        """All runs × all metrics. A pandas DataFrame when pandas is installed, else a list of dicts."""
        rows = []
        for r in self.runs():
            if only_done and r["status"] != "done":
                continue
            label = r["label"] if r["label"] != "Baseline" else f"Baseline · {r['config']['model'].removesuffix('.gguf')}"
            rows.append({"run_id": r["id"], "label": label, "parent_id": r.get("parent_id"), "status": r["status"],
                         **_flat(r["config"], "config."), **r.metrics, **_flat(r.get("stats") or {}, "vs_baseline.")})
        try:
            import pandas as pd
            return pd.DataFrame(rows).set_index("run_id")
        except ImportError:
            return rows

    def branch(self, parent: Run | int | None = None, label: str = "", **changes) -> int:
        """Queue a run: the parent's exact setup + `changes` (model=, engine=, kv=, ctx=, vram_limit_mb=, ngl=, fa=, threads=)."""
        p = parent if isinstance(parent, dict) else (self.run(parent) if parent is not None else self.baseline())
        return self.c.post(f"/api/experiments/{self.id}/runs", {"label": label, "config": {**p["config"], **changes}, "parent_id": p["id"]})["id"]

    def sweep(self, grid: dict, parent: Run | int | None = None, repeats: int = 1) -> list[int]:
        p = parent if isinstance(parent, dict) else (self.run(parent) if parent is not None else None)
        return self.c.post(f"/api/experiments/{self.id}/sweep", {"grid": grid, "repeats": repeats, "parent_id": p["id"] if p else None})["ids"]

    def wait(self, poll: float = 15, verbose: bool = True) -> None:
        last = None
        while True:
            rs = self.runs()
            busy = [r for r in rs if r["status"] in ("queued", "running")]
            now = next((f"{r['label']}: {r.get('phase')}" for r in busy if r["status"] == "running"), None)
            if verbose and (now, len(busy)) != last:
                print(time.strftime("%H:%M:%S"), now or "", f"({len(busy)} left)" if busy else "done")
                last = (now, len(busy))
            if not busy:
                return
            time.sleep(poll)

    # ---- Experiment trackers ----
    def _tracker_rows(self):
        import re
        clean = lambda k: re.sub(r"[^A-Za-z0-9_.\- /]", "_", k)[:250]  # noqa: E731
        info = self.info()
        for r in info["runs_list"]:
            if r["status"] != "done":
                continue
            metrics = {clean(k): float(v) for k, v in Run(r).metrics.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool) and not k.startswith(("provenance.", "hardware.", "imported_from."))}
            params = {clean(k): str(v) for k, v in _flat(r["config"]).items()}
            prov = {clean(f"blocky.{k}"): str(v)[:500] for k, v in _flat({"provenance": r["results"].get("provenance") or {},
                                                                        "hardware": r["results"].get("hardware") or {}}).items() if v is not None}
            tags = {"blocky.run_id": str(r["id"]), "blocky.parent_id": str(r.get("parent_id") or ""), "blocky.experiment": info["name"], **prov}
            yield r, metrics, params, tags

    def to_mlflow(self, tracking_uri: str | None = None, experiment_name: str | None = None) -> list[str]:
        """Log every finished run to MLflow: config as params, every number as metrics, provenance + hardware as tags.

        Default: a local SQLite file `./mlflow.db` (open it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).
        Pass a tracking server URI to log there instead. Needs `pip install mlflow` (or mlflow-skinny + sqlalchemy + alembic).
        """
        import mlflow
        mlflow.set_tracking_uri(tracking_uri or "sqlite:///mlflow.db")
        exp = mlflow.set_experiment(experiment_name or f"blocky · {self.info()['name']}")
        ids = []
        for r, metrics, params, tags in self._tracker_rows():
            with mlflow.start_run(experiment_id=exp.experiment_id, run_name=r["label"]) as run:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.set_tags(tags)
                mlflow.log_dict({k: v for k, v in r["results"].items() if k != "series"}, "blocky_results.json")
                ids.append(run.info.run_id)
        return ids

    def to_wandb(self, project: str = "blocky", mode: str = "offline", entity: str | None = None) -> int:
        """Log every finished run to Weights & Biases. Offline by default: nothing leaves this machine until you
        run `wandb sync` yourself. Needs `pip install wandb`."""
        import wandb
        name = self.info()["name"]
        n = 0
        for r, metrics, params, tags in self._tracker_rows():
            run = wandb.init(project=project, entity=entity, name=r["label"], group=name, config={**params, **tags}, mode=mode, reinit="finish_previous")
            run.summary.update(metrics)
            run.finish()
            n += 1
        return n

    def export_csv(self, path: str) -> str:
        with urllib.request.urlopen(f"{self.c.url}/api/experiments/{self.id}/export.csv", timeout=self.c.timeout) as r, open(path, "wb") as f:
            f.write(r.read())
        return path
