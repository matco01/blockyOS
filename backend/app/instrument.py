"""Instrument checks: does this experiment's measurement pipeline behave the way a correct one must?

These checks use results the experiment already has and add no new measurement. Each one tests a property that holds
for a correct instrument, whatever the model or hardware:

- self_kld          A rerun of the baseline's exact setup scored against the baseline's logits gives KL divergence ≈ 0.
- determinism       Greedy MMLU-Pro answers of a rerun match the baseline's (the paired test must not find a change).
- speed_repeat      Decode speed of reruns stays close to the baseline's (session-to-session drift).
- quant_order       Among weight variants of one model, fewer bits per weight do not give a *lower* KL divergence.
- roofline          Measured decode traffic doesn't exceed the hardware's nominal memory bandwidth.
- reference_values  Results match values the user recorded from a cited source (DATA_DIR/reference-values.json).

A check that lacks the runs it needs reports "not_run" with the runs to add. Passing all checks is evidence that the
pipeline works on this machine, not proof that every number is right; failing one means stop and investigate.
"""
import json
import math
import re
import time

from . import config

VERSION = "instrument-checks/1"
REFERENCE_FILE = config.DATA_DIR / "reference-values.json"
REFERENCE_METRICS = {  # metric name -> path in a run's results
    "ppl": ("ppl",), "kld": ("kld",), "same_top": ("same_top",), "mmlu_acc": ("mmlu", "acc"),
    "tg": ("speed", "tg"), "pp": ("speed", "pp"),
}


def _get(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _bpw(run: dict) -> float | None:
    w = run["results"].get("weight_summary") or {}
    return w["bytes"] * 8 / w["params"] if w.get("bytes") and w.get("params") else None


def _without_model(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k != "model"}


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy) if sx and sy else None


def _result(cid: str, title: str, status: str, detail: str, **extra) -> dict:
    return {"id": cid, "title": title, "status": status, "detail": detail, **extra}


def load_references() -> list[dict]:
    """[{"model_sha256", "metric", "value", "tolerance", "source"}, ...]; malformed entries are skipped."""
    try:
        data = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for r in data if isinstance(data, list) else []:
        if (isinstance(r, dict) and isinstance(r.get("model_sha256"), str) and r.get("metric") in REFERENCE_METRICS
                and _num(r.get("value")) is not None and _num(r.get("tolerance")) is not None and str(r.get("source") or "").strip()):
            out.append(r)
    return out


def add_reference(entry: dict) -> list[dict]:
    """Records a value published for a byte-identical model file. Returns all references."""
    sha = str(entry.get("model_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("The model's SHA-256 (64 hex characters) is needed: values only count for the identical file")
    if entry.get("metric") not in REFERENCE_METRICS:
        raise ValueError(f"Metric: one of {', '.join(REFERENCE_METRICS)}")
    value, tol = _num(entry.get("value")), _num(entry.get("tolerance"))
    if value is None or tol is None or tol < 0:
        raise ValueError("Value and tolerance must be numbers (tolerance ≥ 0)")
    source = " ".join(str(entry.get("source") or "").split())[:500]
    if len(source) < 15:
        raise ValueError("Cite the source: author, title, table or figure, and a link")
    refs = load_references() + [{"model_sha256": sha, "metric": entry["metric"], "value": value, "tolerance": tol, "source": source,
                                 "added_at": time.time()}]
    _save_references(refs)
    return refs


def remove_reference(index: int) -> list[dict]:
    refs = load_references()
    if not 0 <= index < len(refs):
        raise ValueError("No such reference value")
    del refs[index]
    _save_references(refs)
    return refs


def _save_references(refs: list[dict]) -> None:
    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = REFERENCE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(refs, indent=1), encoding="utf-8")
    tmp.replace(REFERENCE_FILE)


def check(exp: dict, references: list[dict] | None = None) -> dict:
    """Run every instrument check on an enriched experiment (profile.get_experiment)."""
    done = [r for r in exp["runs_list"] if r["status"] == "done" and not r["results"].get("imported_from")
            and r["config"].get("engine") != "research"]
    out: list[dict] = []
    if not done:
        return {"version": VERSION, "checks": [], "note": "Profile a baseline first."}
    base = done[0]
    repeats = [r for r in done[1:] if r["config"] == base["config"]]
    add_repeat = "Queue “Repeat of baseline” (the baseline's exact setup) at least twice, ideally at different times."

    # 1) The baseline against itself.
    klds = [(r["id"], _num(r["results"].get("kld")), _num(r["results"].get("same_top"))) for r in repeats]
    klds = [x for x in klds if x[1] is not None]
    if not klds:
        out.append(_result("self_kld", "Baseline vs itself: KL divergence ≈ 0", "not_run", "No rerun of the baseline's setup has a KL divergence yet.", todo=add_repeat))
    else:
        worst = max(k for _, k, _ in klds)
        top = min((s for _, _, s in klds if s is not None), default=None)
        status = "pass" if worst < 0.001 and (top is None or top >= 99.5) else "warn" if worst < 0.01 else "fail"
        out.append(_result("self_kld", "Baseline vs itself: KL divergence ≈ 0", status,
                           f"Largest KLD of {len(klds)} rerun(s): {worst:.6f}" + (f"; same top token ≥ {top:.2f}%" if top is not None else "")
                           + (". Nonzero values this small come from non-deterministic GPU reductions." if status == "pass" and worst > 0 else
                              "" if status == "pass" else ". Identical setups should give identical logits: check that the model file, engine build and "
                              "placement (GPU layers, threads) are really the same, and whether the backend is deterministic."),
                           runs=[i for i, _, _ in klds], value=worst))

    # 2) Greedy answers are reproducible.
    ans = base["results"].get("mmlu", {}).get("answers")
    pairs = [(r["id"], r["results"].get("mmlu", {}).get("answers"), (r.get("stats") or {}).get("mmlu_p")) for r in repeats]
    pairs = [x for x in pairs if ans and x[1] and len(x[1]) == len(ans)]
    if not pairs:
        out.append(_result("determinism", "Reruns give the same MMLU-Pro answers", "not_run", "No rerun of the baseline's setup has MMLU-Pro answers yet.", todo=add_repeat))
    else:
        agree = min(sum(x == y for x, y in zip(ans, a)) / len(ans) for _, a, _ in pairs)
        sig = [i for i, _, p in pairs if p is not None and p < 0.05]
        status = "fail" if sig else "pass" if agree >= 0.99 else "warn"
        out.append(_result("determinism", "Reruns give the same MMLU-Pro answers", status,
                           f"Lowest agreement with the baseline: {agree * 100:.1f}% of {len(ans)} questions"
                           + (f". The paired test reports a significant change for run(s) {', '.join(map(str, sig))}, which a rerun must not produce" if sig else "")
                           + ("" if status == "pass" else ". Answers of an identical greedy setup should barely change; look for non-determinism or a changed environment."),
                           runs=[i for i, _, _ in pairs], value=agree))

    # 3) Speed drift between sessions.
    tg0 = _num(_get(base["results"], ("speed", "tg")))
    tgs = [(r["id"], _num(_get(r["results"], ("speed", "tg")))) for r in repeats]
    tgs = [x for x in tgs if x[1] is not None]
    if not tg0 or not tgs:
        out.append(_result("speed_repeat", "Decode speed is repeatable", "not_run", "No rerun of the baseline's setup has a decode speed yet.", todo=add_repeat))
    else:
        dev = max(abs(v / tg0 - 1) for _, v in tgs)
        status = "pass" if dev <= 0.03 else "warn" if dev <= 0.08 else "fail"
        out.append(_result("speed_repeat", "Decode speed is repeatable", status,
                           f"Largest difference from the baseline across {len(tgs)} rerun(s): {dev * 100:.1f}%"
                           + ("" if status == "pass" else ". Speed differences smaller than this aren't trustworthy from single runs: use Repeat & validate, "
                              "check clocks and temperature, and close other GPU programs."),
                           runs=[i for i, _ in tgs], value=dev))

    # 4) Fewer bits must not mean higher fidelity.
    same = [r for r in done if _without_model(r["config"]) == _without_model(base["config"]) and _num(r["results"].get("kld")) is not None
            and _bpw(r) and not r["results"].get("token_note")]
    by_model: dict[str, dict] = {}
    for r in same:
        by_model.setdefault(r["config"]["model"], r)
    variants = list(by_model.values())
    if len({round(_bpw(r), 2) for r in variants}) < 3:
        out.append(_result("quant_order", "Fewer bits per weight don't give a lower KL divergence", "not_run",
                           "Needs at least three weight variants of the baseline model (e.g. Q8_0, Q4_K_M, Q2_K) with otherwise identical settings.",
                           todo="Branch from the baseline and change only the weights file, for three or more quantization levels."))
    else:
        bpws = [_bpw(r) for r in variants]
        kl = [_num(r["results"]["kld"]) for r in variants]
        rho = _spearman(bpws, kl)
        inversions = [(a["config"]["model"], b["config"]["model"]) for a in variants for b in variants
                      if _bpw(a) > _bpw(b) * 1.15 and _num(a["results"]["kld"]) > _num(b["results"]["kld"]) * 1.5 + 1e-4]
        status = "fail" if inversions else "pass" if rho is not None and rho <= -0.8 else "warn"
        out.append(_result("quant_order", "Fewer bits per weight don't give a lower KL divergence", status,
                           f"Spearman correlation of bits per weight with KLD across {len(variants)} variants: {rho:.2f}" if rho is not None else "Correlation undefined",
                           runs=[r["id"] for r in variants], value=rho,
                           **({"inversions": inversions[:5], "hint": "A clearly larger file with clearly worse KLD usually means a mislabeled file, a broken "
                               "quantization, or a KLD reference mix-up."} if inversions else {})))

    # 5) Physics: decode can't read weights faster than the memory allows.
    utils = [(r["id"], _num(_get(r["results"], ("hw", "decode", "roof_util")))) for r in done]
    utils = [x for x in utils if x[1] is not None]
    if not utils:
        out.append(_result("roofline", "Measured speed stays under the memory-bandwidth limit", "not_run",
                           "No run has a bandwidth estimate (needs a known GPU or measured RAM bandwidth).",
                           todo="Add the GPU's bandwidth to the table in profile.py if it's missing."))
    else:
        over = [(i, u) for i, u in utils if u > 1.05]
        out.append(_result("roofline", "Measured speed stays under the memory-bandwidth limit", "warn" if over else "pass",
                           f"Highest share of the nominal bandwidth: {max(u for _, u in utils) * 100:.0f}%"
                           + (f". Run(s) {', '.join(str(i) for i, _ in over)} exceed it: either the spec-sheet bandwidth is wrong for this device, the "
                              "weight-bytes model doesn't fit this architecture (e.g. mixture-of-experts reads only some weights), or timing is off." if over else ""),
                           runs=[i for i, _ in utils], value=max(u for _, u in utils)))

    # 6) Values from a cited source.
    refs = load_references() if references is None else references
    matched = []
    for ref in refs:
        for r in done:
            if _get(r["results"], ("provenance", "model", "sha256")) == ref["model_sha256"]:
                v = _num(_get(r["results"], REFERENCE_METRICS[ref["metric"]]))
                if v is not None:
                    matched.append({"run": r["id"], "metric": ref["metric"], "measured": v, "reference": ref["value"],
                                    "tolerance": ref["tolerance"], "source": ref["source"],
                                    "ok": abs(v - ref["value"]) <= ref["tolerance"]})
    if not matched:
        out.append(_result("reference_values", "Results match cited reference values", "not_run",
                           f"{len(refs)} reference value(s) recorded; none match a model file in this experiment." if refs else
                           "No reference values recorded.",
                           todo=f"Add entries to {REFERENCE_FILE} (model SHA-256, metric, value, tolerance, source); see docs/VALIDATION.md."))
    else:
        bad = [m for m in matched if not m["ok"]]
        out.append(_result("reference_values", "Results match cited reference values", "fail" if bad else "pass",
                           f"{len(matched) - len(bad)} of {len(matched)} value(s) within tolerance", matches=matched))

    counts = {s: sum(c["status"] == s for c in out) for s in ("pass", "warn", "fail", "not_run")}
    return {"version": VERSION, "checks": out, "counts": counts,
            "note": "Passing checks is evidence that the measurement pipeline works on this machine, not proof that every number is right. "
                    "A failed check means: stop and investigate before trusting comparisons in this experiment."}
