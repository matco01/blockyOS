"""Dependency-free reporting SDK for Blocky research jobs. No network calls.

Runs receive BLOCKY_RUN_DIR and BLOCKY_CONFIG from the runner. Reports are JSONL on disk.
"""
import hashlib
import json
import math
import os
import time
from pathlib import Path


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Run:
    def __init__(self):
        self.directory = Path(os.environ["BLOCKY_RUN_DIR"]).resolve()
        self.config = json.loads(os.environ.get("BLOCKY_CONFIG", "{}"))

    def _event(self, event):
        event["time"] = time.time()
        encoded = json.dumps(event, allow_nan=False)
        if len(encoded.encode()) > 256_000:
            raise ValueError("Report event exceeds 256 KB; save large data as an artifact")
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(encoded + "\n")

    def metric(self, name, value, *, unit, protocol, task, dataset_sha256, scope,
               samples=None, kind="measured", better=None, note="", tokenizer_sha256=None):
        """Describe an evaluator's output. Matching contracts permit comparison, not automatic significance.

        protocol identifies the measurement procedure; task identifies what is scored; scope describes timing/
        inclusion boundaries. Cross-tokenizer comparisons should use bytes or common task scores, not tokens.
        """
        if kind not in ("measured", "estimated", "simulated") or better not in (None, "higher", "lower"):
            raise ValueError("Invalid metric kind/direction")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("Metrics must be finite numbers; omit unavailable measurements")
        if not all(isinstance(x, str) and x for x in (name, unit, protocol, task, dataset_sha256, scope)):
            raise ValueError("Metric requires a unit, protocol, task, dataset fingerprint and measurement scope")
        if "token" in unit.lower() and not tokenizer_sha256:
            raise ValueError("Token-based metrics require a tokenizer fingerprint; use byte units for cross-tokenizer comparisons")
        samples = list(samples) if samples is not None else None
        if samples is not None and any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in samples):
            raise ValueError("Samples must be finite numbers")
        self._event({"type": "metric", "name": name, "value": value, "unit": unit, "protocol": protocol,
                     "task": task, "dataset_sha256": dataset_sha256, "scope": scope, "samples": samples,
                     "kind": kind, "better": better, "note": note, "tokenizer_sha256": tokenizer_sha256})

    def artifact(self, path, description=""):
        if Path(path).is_symlink():
            raise ValueError("Artifact symlinks are not supported")
        path = Path(path).resolve()
        if not path.is_relative_to(self.directory) or not path.is_file() or path.is_symlink():
            raise ValueError("Artifacts must be files within the run directory")
        self._event({"type": "artifact", "path": str(path.relative_to(self.directory)), "description": description,
                     "sha256": fingerprint(path), "bytes": path.stat().st_size})

    def progress(self, message, step=None):
        self._event({"type": "progress", "message": str(message)[:2000], "step": step})
