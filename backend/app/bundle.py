"""Reproducibility bundle: one zip with everything needed to check or repeat an experiment.

  README.md                 what's inside and how to reproduce it
  experiment.json           the experiment as exported (runs, results, statistics, test settings, preregistrations)
  instrument-check.json     the instrument checks at export time
  environment.json          Blocky version and commit, Python and packages of the backend, OS
  runs/<id>/provenance.json per run: protocol, engine commit, model and dataset SHA-256, GPU, driver, noise controls
  runs/<id>/commands.txt    the exact commands each phase ran
  runs/<id>/outputs/*.txt   retained raw evaluator output (llama-bench, llama-perplexity, logs), where kept
  PROTOCOL.md               the measurement protocol document of this Blocky version
  reproduce.py              re-runs every setup on a Blocky device and compares the results with these

Nothing here leaves the device unless the user downloads the bundle and shares it.
"""
import io
import json
import platform
import sys
import time
import zipfile
from importlib import metadata
from pathlib import Path

from . import __version__, instrument, profile

REPO = Path(__file__).resolve().parents[2]
REPRODUCE = Path(__file__).resolve().parent / "reproduce_template.py"


def environment() -> dict:
    git = lambda *a: profile._sh("git", "-c", "safe.directory=*", "-C", str(REPO), *a)  # noqa: E731
    pkgs = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata.get("Name"))
    return {"blocky_version": __version__, "blocky_commit": git("rev-parse", "HEAD") or None,
            "blocky_dirty": bool(git("status", "--porcelain")), "protocol": profile.PROTOCOL,
            "python": sys.version.split()[0], "platform": platform.platform(), "backend_packages": pkgs,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def readme(exp: dict, env: dict) -> str:
    runs = [r for r in exp["runs_list"] if r["status"] == "done"]
    return f"""# Reproducibility bundle: {exp['name']}

Exported {env['exported_at']} from blockyOS {env['blocky_version']} (commit {(env['blocky_commit'] or 'unknown')[:12]}{', with local changes' if env['blocky_dirty'] else ''}).
Measurement protocol: `{env['protocol']}`. Model: `{exp['model']}`. Completed runs: {len(runs)}.

## Contents

- `experiment.json`: the full experiment, as Blocky's own export (importable with Experiments → Import).
- `instrument-check.json`: whether the measurement pipeline behaved correctly when exported.
- `runs/<id>/provenance.json`, `commands.txt`, `outputs/`: per-run provenance, commands and retained raw output.
- `environment.json`: software versions on the exporting device.
- `PROTOCOL.md`: how every number was measured.

## Reproduce

On a device running blockyOS with the same model files (check the SHA-256 in each run's provenance):

```bash
python reproduce.py --url http://localhost:8000
```

It creates a new experiment with the same test settings, queues every setup in the original order (branches keep their
parents), waits, and writes `reproduction-report.json`: each metric next to the original, with whether the original value
lies inside the new run's interval where one exists. Different hardware is expected to change speed and energy, not
accuracy or KL divergence; any mismatch in model hashes, protocol or engine commit is listed first.

The experiment can also be imported for inspection without re-running it: Experiments → Import → `experiment.json`.
"""


def build(exp_id: int) -> tuple[str, bytes]:
    exp = profile.get_experiment(exp_id)
    if not exp:
        raise ValueError("Experiment not found")
    env = environment()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", readme(exp, env))
        z.writestr("experiment.json", json.dumps(exp, indent=1, default=str))
        z.writestr("instrument-check.json", json.dumps(instrument.check(exp), indent=1, default=str))
        z.writestr("environment.json", json.dumps(env, indent=1))
        protocol_doc = REPO / "docs" / "PROTOCOL.md"
        if protocol_doc.is_file():
            z.writestr("PROTOCOL.md", protocol_doc.read_text(encoding="utf-8"))
        z.writestr("reproduce.py", REPRODUCE.read_text(encoding="utf-8"))
        for r in exp["runs_list"]:
            res = r.get("results") or {}
            base = f"runs/{r['id']}"
            z.writestr(f"{base}/config.json", json.dumps({"label": r.get("label"), "status": r["status"], "parent_id": r.get("parent_id"),
                                                          "config": r["config"]}, indent=1))
            if res.get("provenance"):
                z.writestr(f"{base}/provenance.json", json.dumps(res["provenance"], indent=1, default=str))
            if res.get("commands"):
                z.writestr(f"{base}/commands.txt", "\n".join(f"# {k}\n{v}\n" for k, v in res["commands"].items()))
            try:
                outputs = profile.evidence_outputs(r["id"])["outputs"]
            except ValueError:
                outputs = {}
            for phase, out in outputs.items():
                z.writestr(f"{base}/outputs/{phase}.txt", out["text"] + ("\n[truncated]\n" if out.get("truncated") else ""))
    name = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in exp["name"]).strip("-") or f"experiment-{exp_id}"
    return f"{name}-bundle.zip", buf.getvalue()
