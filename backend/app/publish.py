"""Ship: a model card written from the measurements, a license check, and publishing a version to Hugging Face.

Nothing leaves the device unless the owner publishes. The Hugging Face token is a secret: never returned by the API,
never available to AI assistants (the /api/publish endpoints are owner-only, see mcp_server.BLOCKED).
"""
import asyncio
import re
import time
from pathlib import Path

from . import db, gguf, lab, profile

# Licenses that allow redistribution without extra conditions; anything else needs the owner's explicit confirmation.
PERMISSIVE = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0", "unlicense"}
LICENSES = [
    ("apache-2.0", "Apache 2.0"), ("mit", "MIT"), ("cc-by-4.0", "CC BY 4.0"),
    ("llama3.1", "Llama 3.1 Community License"), ("llama3.2", "Llama 3.2 Community License"),
    ("llama3.3", "Llama 3.3 Community License"), ("gemma", "Gemma Terms of Use"),
    ("cc-by-nc-4.0", "CC BY-NC 4.0 (non-commercial)"), ("other", "Other (describe it in the card)"),
]
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
jobs: dict[str, dict] = {}


def _version(exp_id: int, run_id: int) -> tuple[dict, dict, dict | None]:
    exp = profile.get_experiment(exp_id)
    if not exp:
        raise ValueError("Experiment not found")
    run = next((r for r in exp["runs_list"] if r["id"] == run_id), None)
    if not run or run["status"] != "done":
        raise ValueError("Choose a measured version (a finished run)")
    baseline = next((r for r in exp["runs_list"] if r["status"] == "done"), None)
    return exp, run, (baseline if baseline and baseline["id"] != run_id else None)


def model_file(run: dict) -> Path:
    name = str(run["config"].get("model") or "")
    if not name.endswith(".gguf"):
        raise ValueError("Only GGUF model files can be shipped from here")
    path = lab.MODELS_DIR / Path(name).name
    if not path.is_file():
        raise ValueError(f"{name} is no longer on this device")
    return path


def license_info(exp_id: int, run_id: int) -> dict:
    """The license the file declares (or its source file, for variants built here), and whether it needs confirming."""
    _, run, _ = _version(exp_id, run_id)
    path = model_file(run)
    found = gguf.metadata(path).get("license")
    recipe = ((run["results"].get("provenance") or {}).get("model") or {}).get("recipe") or {}
    if not found and recipe.get("source") and (lab.MODELS_DIR / recipe["source"]).is_file():
        found = gguf.metadata(lab.MODELS_DIR / recipe["source"]).get("license")
    found = str(found).strip().lower() if found else None
    return {"detected": found, "permissive": found in PERMISSIVE, "options": [{"id": k, "label": v} for k, v in LICENSES],
            "file": path.name, "bytes": path.stat().st_size, "model_name": gguf.metadata(path).get("name")}


def _pm(v, sd=None, digits=2, unit=""):
    if v is None:
        return "–"
    return f"{v:.{digits}f}{' ± ' + format(sd, f'.{digits}f') if sd is not None else ''}{unit}"


def _row(name: str, r: dict | None) -> str:
    if not r:
        return ""
    x = r["results"]
    m, s = x.get("mmlu") or {}, x.get("speed") or {}
    ci = m.get("ci95")
    acc = f"{m['acc'] * 100:.1f}% [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]" if m.get("acc") is not None and ci else "–"
    needle = f"{(x.get('needle') or {}).get('acc', 0) * 100:.0f}%" if x.get("needle") else "–"
    return (f"| {name} | {acc} | {_pm(s.get('tg'), s.get('tg_sd'), 1)} | {_pm(s.get('pp'), s.get('pp_sd'), 0)} | "
            f"{_pm(x.get('ppl'), x.get('ppl_sd'), 3)} | {_pm(x.get('kld'), None, 4)} | "
            f"{_pm(x.get('same_top'), None, 1, '%')} | {needle} |\n")


def card(exp_id: int, run_id: int, license_id: str = "", base_model: str = "", title: str = "") -> str:
    """A Hugging Face model card (README.md) written only from what was measured and recorded."""
    exp, run, base = _version(exp_id, run_id)
    path = model_file(run)
    prov = run["results"].get("provenance") or {}
    model, llama, gpu = prov.get("model") or {}, prov.get("llama_cpp") or {}, prov.get("gpu") or {}
    recipe = model.get("recipe") or {}
    lic = license_id or license_info(exp_id, run_id)["detected"] or "other"
    title = title.strip() or path.stem
    front = ["---", f"license: {lic}", "library_name: gguf", "tags:", "- gguf", "- blockyos"]
    if base_model.strip():
        front += ["base_model:", f"- {base_model.strip()}"]
    front.append("---")
    stats = run.get("stats") or {}
    diff = stats.get("mmlu_diff") or {}
    lines = ["\n".join(front), "", f"# {title}", ""]
    if recipe:
        how = f"quantized to **{recipe.get('qtype')}**" + (" with an importance matrix" if recipe.get("imatrix") else "")
        if recipe.get("pruned_layers"):
            how += f", {len(recipe['pruned_layers'])} layer(s) removed"
        lines.append(f"Made from `{recipe.get('source')}` (SHA-256 `{(recipe.get('source_sha256') or '')[:16]}…`), {how}, with "
                     "llama.cpp's own tools in blockyOS.")
    else:
        lines.append(f"`{path.name}`, measured in blockyOS.")
    lines += ["", "## Measured results", "",
              f"Measured with the blockyOS measurement protocol `{(prov.get('protocol') or '').split(' (')[0]}` on "
              f"{gpu.get('name') or 'this device'} (driver {gpu.get('driver') or '?'}), llama.cpp {llama.get('version') or '?'} "
              f"(`{llama.get('commit') or '?'}`). ± is the spread over repetitions (speed: 1.4826 × MAD of 5) or the standard "
              "error (perplexity). Accuracy is a fixed 560-question MMLU-Pro sample (5-shot, option-letter scoring) with a "
              "95% interval; it is meant for paired comparisons on this setup, not for leaderboards.", "",
              "| Version | MMLU-Pro sample | Decode tok/s | Prefill tok/s | Perplexity | KL div. vs baseline | Same top token | Needle |",
              "|---|---|---|---|---|---|---|---|",
              _row(f"**This model** ({run['label']})", run).rstrip("\n")]
    if base:
        lines.append(_row(f"Baseline ({base['label']})", base).rstrip("\n"))
    if diff.get("diff") is not None:
        lo, hi = diff.get("ci95") or (None, None)
        sig = stats.get("mmlu_p_holm", stats.get("mmlu_p"))
        lines += ["", f"Accuracy against the baseline: {diff['diff'] * 100:+.2f} points"
                  + (f" (95% interval {lo * 100:+.2f} to {hi * 100:+.2f})" if lo is not None else "")
                  + (f", paired test p = {sig:.3g}" if sig is not None else "")
                  + ("; not a significant difference" if sig is not None and sig >= 0.05 else "") + "."]
    lines += ["", "Speed depends on the hardware and on how much of the model fits on the GPU; measure on your own device "
              "before relying on it.", "", "## Reproduce", "",
              f"- Model file: `{path.name}`, SHA-256 `{model.get('sha256') or '?'}`",
              "- `reproducibility-bundle.zip` (if included) holds the provenance, the exact commands, raw outputs and "
              "`reproduce.py`, which re-runs every measurement on another blockyOS device and compares.",
              "- blockyOS: https://github.com/matco01/blockyOS", ""]
    if lic not in PERMISSIVE:
        lines += ["## License", "", f"Distributed under `{lic}`. Check the base model's license terms before using it.", ""]
    return "\n".join(lines)


# ---------- Hugging Face ----------

def token() -> str:
    return str(db.get_settings().get("hf_token") or "")


def account() -> dict:
    s = db.get_settings()
    return {"connected": bool(s.get("hf_token")), "user": s.get("hf_user") or None}


def connect(tok: str) -> dict:
    """Checks the token with Hugging Face and stores it (it is never shown again)."""
    tok = str(tok or "").strip()
    if not re.match(r"^hf_[A-Za-z0-9]{20,}$", tok):
        raise ValueError("Paste a Hugging Face access token (it starts with hf_). Create one with 'write' access at "
                         "huggingface.co → Settings → Access Tokens")
    from huggingface_hub import HfApi
    try:
        who = HfApi().whoami(token=tok)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Hugging Face didn't accept this token ({e.__class__.__name__})") from None
    with db.connect() as c:
        for k, v in (("hf_token", tok), ("hf_user", who.get("name") or "")):
            c.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                      (k, __import__("json").dumps(v)))
    return account()


def disconnect() -> dict:
    with db.connect() as c:
        c.execute("DELETE FROM settings WHERE key IN ('hf_token', 'hf_user')")
    return account()


def start_hf(body: dict) -> str:
    exp_id, run_id = int(body.get("experiment_id") or 0), int(body.get("run_id") or 0)
    _, run, _ = _version(exp_id, run_id)
    path = model_file(run)
    if not token():
        raise ValueError("Connect your Hugging Face account first")
    repo = str(body.get("repo") or "").strip()
    if not _REPO.match(repo) or ".." in repo:
        raise ValueError("Repository: owner/name, letters, digits, dot, dash and underscore")
    lic = str(body.get("license") or "").strip().lower()
    if not lic:
        raise ValueError("Choose the license the model is shared under")
    if lic not in PERMISSIVE and not body.get("confirm_license"):
        raise ValueError(f"'{lic}' has conditions: confirm that you are allowed to share this model under it")
    if any(j["status"] == "running" for j in jobs.values()):
        raise ValueError("Another upload is running")
    readme = card(exp_id, run_id, lic, str(body.get("base_model") or ""), str(body.get("title") or ""))
    jid = str(int(time.time() * 1000))
    jobs[jid] = {"id": jid, "repo": repo, "status": "running", "step": "starting", "error": None, "url": None,
                 "private": body.get("private", True) is not False, "file": path.name, "bytes": path.stat().st_size,
                 "started": time.time()}
    asyncio.get_running_loop().create_task(_publish(jid, exp_id, path, readme, bool(body.get("include_bundle", True))))
    return jid


async def _publish(jid: str, exp_id: int, path: Path, readme: str, include_bundle: bool) -> None:
    j = jobs[jid]
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token())
        j["step"] = "creating the repository"
        url = await asyncio.to_thread(api.create_repo, j["repo"], private=j["private"], exist_ok=True, repo_type="model")
        j["step"] = "uploading the model card"
        await asyncio.to_thread(api.upload_file, path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=j["repo"],
                                commit_message="Model card from blockyOS measurements")
        if include_bundle:
            from . import bundle
            j["step"] = "uploading the reproducibility bundle"
            _, data = await asyncio.to_thread(bundle.build, exp_id)
            await asyncio.to_thread(api.upload_file, path_or_fileobj=data, path_in_repo="reproducibility-bundle.zip",
                                    repo_id=j["repo"], commit_message="blockyOS reproducibility bundle")
        j["step"] = f"uploading {path.name} ({path.stat().st_size / 2**30:.1f} GB)"
        await asyncio.to_thread(api.upload_file, path_or_fileobj=str(path), path_in_repo=path.name, repo_id=j["repo"],
                                commit_message=f"Upload {path.name}")
        j.update(status="done", step="published", url=str(url))
    except Exception as e:  # noqa: BLE001
        j.update(status="error", error=str(e)[:400])
