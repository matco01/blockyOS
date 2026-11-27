"""Quality tests: run question suites against a model and score them with code.

Scoring is deterministic (numbers, accepted answers, formatting rules, unit
tests): no model grades another model. Model-written code runs inside a
bubblewrap sandbox: no network, read-only system, no access to user files,
CPU and memory limits.
"""
import asyncio
import json
import logging
import random
import re
import shutil
import tempfile
import time
from pathlib import Path

import httpx

from . import arena, benchmarks, db, fresh, generate, lab
from .eval_suites import ANSWER_HINT, CODE_HINT, SUITES

log = logging.getLogger("blocky.evals")
_cancel: set[int] = set()

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   REAL NOT NULL,
    finished_at  REAL,
    status       TEXT NOT NULL,
    model        TEXT NOT NULL,
    think        INTEGER NOT NULL DEFAULT 0,
    suites       TEXT NOT NULL,
    results      TEXT NOT NULL,
    summary      TEXT,
    error        TEXT
);
CREATE TABLE IF NOT EXISTS custom_suites (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    name        TEXT NOT NULL,
    questions   TEXT NOT NULL
);
"""


def init() -> None:
    with db.connect() as c:
        c.executescript(SCHEMA)
        if "options" not in {r[1] for r in c.execute("PRAGMA table_info(eval_runs)")}:
            c.execute("ALTER TABLE eval_runs ADD COLUMN options TEXT")
        c.execute("UPDATE eval_runs SET status = 'error', error = 'Interrupted' WHERE status IN ('queued', 'running')")


_bench_cache: dict[str, tuple[float, list]] = {}


def bench_questions(key: str) -> list[dict]:
    p = benchmarks.path(key)
    mtime = p.stat().st_mtime
    if key not in _bench_cache or _bench_cache[key][0] != mtime:
        _bench_cache[key] = (mtime, benchmarks.load(key))
    return _bench_cache[key][1]


def parse_spec(spec: str) -> tuple[str, int | None]:
    """'bench:gsm8k@50' -> ('bench:gsm8k', 50); 'math' -> ('math', None)."""
    key, _, n = spec.partition("@")
    return key, (int(n) if n.isdigit() else None)


def questions_for(spec: str, seed: int = 0) -> list[dict]:
    key, limit = parse_spec(spec)
    if key.startswith("fresh:"):
        return fresh.generate(key[6:], limit or 25, seed)
    qs = all_suites()[key]["questions"] if not key.startswith("bench:") else bench_questions(key[6:])
    return qs[:limit] if limit else qs


# ---------- Suites ----------

def custom_suites() -> dict:
    with db.connect() as c:
        rows = c.execute("SELECT * FROM custom_suites ORDER BY id").fetchall()
    return {f"custom:{r['id']}": {"title": r["name"], "icon": "user", "color": "#a78bfa", "custom": True,
                                  "about": "Your own questions.", "questions": json.loads(r["questions"])} for r in rows}


def all_suites() -> dict:
    """Built-in + custom suites (with questions). Benchmarks are listed separately (they're big)."""
    return {**SUITES, **custom_suites()}


def suite_title(key: str) -> str:
    k, _ = parse_spec(key)
    if k.startswith("bench:"):
        return benchmarks.catalog().get(k[6:], {}).get("title", k)
    if k.startswith("fresh:"):
        return fresh.SETS.get(k[6:], {}).get("title", k)
    return all_suites().get(k, {}).get("title", k)


def suites_summary() -> list[dict]:
    return [{"key": k, "title": s["title"], "icon": s["icon"], "color": s["color"], "about": s["about"],
             "count": len(s["questions"]), "custom": s.get("custom", False),
             "available": k != "coding" or sandbox_available()}
            for k, s in all_suites().items()]


def create_custom(name: str, lines: list[dict]) -> int:
    name = name.strip()[:80]
    if not name:
        raise ValueError("Give the test set a name")
    qs = []
    for item in lines[:200]:
        q, a = str(item.get("q", "")).strip(), str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        num = a.replace(",", "")
        if re.fullmatch(r"-?\d+(\.\d+)?", num):
            qs.append({"q": q[:2000], "kind": "number", "answer": float(num)})
        else:
            qs.append({"q": q[:2000], "kind": "text", "accept": [x.strip().lower() for x in a.split("/") if x.strip()]})
    if not qs:
        raise ValueError("Add at least one 'question | answer' line")
    with db.connect() as c:
        return c.execute("INSERT INTO custom_suites (created_at, name, questions) VALUES (?, ?, ?)",
                         (time.time(), name, json.dumps(qs))).lastrowid


def delete_custom(suite_id: int) -> None:
    with db.connect() as c:
        c.execute("DELETE FROM custom_suites WHERE id = ?", (suite_id,))


# ---------- Scoring ----------

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _unlatex(text: str) -> str:
    """Models often wrap answers in LaTeX: \\text{Answer: } 12, \\boxed{12}, $$ … $$."""
    t = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", text)
    t = re.sub(r"\\(?:text|mathrm|textbf)\{([^{}]*)\}", r"\1", t)
    t = t.replace("$$", "\n").replace("$", "").replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    return t


def final_answer(text: str) -> tuple[str, bool]:
    """The model's final answer: the last 'Answer: …' line, else the last line with real content."""
    text = _unlatex(text)
    matches = re.findall(r"(?im)^\W*\**\s*(?:final\s+)?answer\s*\**\s*[:：]\s*(.+)$", text)
    matches = [m for m in matches if re.search(r"\w", m)]
    if matches:
        return matches[-1].strip(), True
    lines = [l.strip() for l in text.strip().splitlines() if re.search(r"\w", l)]
    return (lines[-1] if lines else ""), False


def score_number(text: str, answer: float, tol: float = 1e-6) -> tuple[bool, str]:
    ans, explicit = final_answer(text)
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", ans).replace("$", "")
    nums = _NUM.findall(cleaned)
    if not nums:
        return False, ans
    pick = float(nums[0] if explicit else nums[-1])
    return abs(pick - float(answer)) <= tol + 1e-9 * abs(float(answer)), ans


def score_text(text: str, accept: list[str]) -> tuple[bool, str]:
    ans, _ = final_answer(text)
    low = ans.lower()
    ok = any(re.search(rf"(?<![a-z0-9]){re.escape(a.lower())}(?![a-z0-9])", low) for a in accept)
    return ok, ans


def _strip_fences(text: str) -> str:
    """Code from a fenced block. Prefers the last complete block (models often revise);
    if the answer was cut off mid-block, takes what there is after the opening fence."""
    blocks = re.findall(r"```(?:\w+)?[ \t]*\n(.*?)```", text, re.S)
    if blocks:
        return blocks[-1]
    m = re.search(r"```(?:\w+)?[ \t]*\n(.*)", text, re.S)
    return m.group(1) if m else text


def check_rules(text: str, rules: list) -> tuple[bool, str]:
    t = text.strip()
    lines = [l for l in t.splitlines() if l.strip()]
    words = re.findall(r"[A-Za-z0-9']+", t)
    bullet = re.compile(r"^\s*[-*•]\s+")
    numbered = re.compile(r"^\s*\d+[.)]\s+")
    failed = []
    for rule in rules:
        name, arg = rule[0], (rule[1] if len(rule) > 1 else None)
        ok = {
            "bullets": lambda: sum(bool(bullet.match(l)) for l in lines) == arg,
            "numbered": lambda: sum(bool(numbered.match(l)) for l in lines) == arg,
            "only_list": lambda: all(bullet.match(l) or numbered.match(l) for l in lines),
            "no_upper": lambda: t == t.lower(),
            "min_words": lambda: len(words) >= arg,
            "max_words": lambda: len(words) <= arg,
            "word_count": lambda: len(words) == arg,
            "exactly": lambda: re.sub(r"[^a-z]", "", t.lower()) == arg,
            "contains": lambda: arg.lower() in t.lower(),
            "lines": lambda: len(lines) == arg,
            "all_caps": lambda: any(ch.isalpha() for ch in t) and all(not ch.isalpha() or ch.isupper() for ch in t),
            "no_char": lambda: arg not in t,
            "starts_with": lambda: t.lower().lstrip("*_ ").startswith(arg),
            "json_keys": lambda: _json_has(t, arg),
        }[name]()
        if not ok:
            failed.append(name.replace("_", " ") + (f" {arg}" if arg is not None else ""))
    return not failed, ("all rules met" if not failed else "failed: " + ", ".join(failed))


def _json_has(t: str, keys: list) -> bool:
    try:
        obj = json.loads(_strip_fences(t).strip())
    except ValueError:
        return False
    return isinstance(obj, dict) and all(k in obj for k in keys)


# ---------- Sandbox ----------

def sandbox_available() -> bool:
    return shutil.which("bwrap") is not None


_RUNNER = """
import json
src = open('/sbx/solution.py').read()
tests = json.load(open('/sbx/tests.json'))
ns = {}
try:
    exec(compile(src, 'solution.py', 'exec'), ns)
except Exception as e:
    print(json.dumps({"passed": 0, "total": len(tests), "error": "code didn't run: " + repr(e)[:200]})); raise SystemExit
passed, errors = 0, []
for t in tests:
    try:
        exec(t, ns); passed += 1
    except Exception as e:
        errors.append(t[:80] + " -> " + (repr(e)[:80] or 'AssertionError'))
print(json.dumps({"passed": passed, "total": len(tests), "error": "; ".join(errors[:2]) or None}))
"""


async def sandbox(files: dict[str, str], script: str, total: int, cpu: int = 5, wall: int = 10, mem_mb: int = 512) -> dict:
    """Run `script` (one of `files`) with python3 inside bubblewrap: no network, read-only system,
    no user files, CPU/memory/file-size limits. The script must print one JSON line: {passed, total, error}."""
    if not sandbox_available():
        return {"passed": 0, "total": total, "error": "sandbox (bubblewrap) not installed"}
    with tempfile.TemporaryDirectory(prefix="blocky-sbx-") as d:
        for name, content in files.items():
            Path(d, name).write_text(content)
        cmd = ["prlimit", f"--as={mem_mb * 1048576}", f"--cpu={cpu}", "--fsize=1048576", "--",
               "bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
               "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               # Library lookup only (no secrets): numpy finds its BLAS library through these.
               "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache", "--ro-bind-try", "/etc/alternatives", "/etc/alternatives",
               "--ro-bind", d, "/sbx", "--chdir", "/tmp", "--unshare-all", "--die-with-parent", "--new-session",
               "--clearenv", "--setenv", "PATH", "/usr/bin", "python3", "-I", f"/sbx/{script}"]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), wall)
        except asyncio.TimeoutError:
            proc.kill()
            return {"passed": 0, "total": total, "error": "timed out (infinite loop?)"}
    try:
        return json.loads(out.decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        if proc.returncode is not None and proc.returncode < 0:
            return {"passed": 0, "total": total, "error": "stopped by the sandbox: used too much CPU time or memory (infinite loop?)"}
        return {"passed": 0, "total": total, "error": (err.decode(errors="replace").strip()[-200:] or "crashed")}


async def run_code(code: str, tests: list[str], cpu: int = 5, wall: int = 10) -> dict:
    return await sandbox({"solution.py": code, "tests.json": json.dumps(tests), "run.py": _RUNNER}, "run.py", len(tests), cpu, wall)


# LiveCodeBench: stdin/stdout programs, or LeetCode-style class Solution methods.
_LCB_RUNNER = r"""
import json, subprocess, sys
meta = json.load(open('/sbx/meta.json'))
tests = meta['tests']
passed, err = 0, None
def norm(s):
    return [l.rstrip() for l in s.strip().splitlines()]
for i, t in enumerate(tests):
    try:
        if t.get('testtype') == 'functional':
            r = subprocess.run([sys.executable, '-I', '/sbx/fn.py', str(i)], capture_output=True, text=True, timeout=6)
            ok = r.stdout.strip().endswith('OK')
            if not ok and err is None: err = (r.stdout + r.stderr).strip()[-160:] or 'wrong answer'
        else:
            r = subprocess.run([sys.executable, '-I', '/sbx/solution.py'], input=t['input'], capture_output=True, text=True, timeout=6)
            ok = norm(r.stdout) == norm(t['output'])
            if not ok and err is None: err = ('runtime error: ' + r.stderr.strip()[-140:]) if r.returncode else 'wrong output'
    except subprocess.TimeoutExpired:
        ok = False
        if err is None: err = 'too slow (time limit 6 s)'
    if not ok:
        break  # pass@1 needs every test; stop at the first failure
    passed += 1
print(json.dumps({'passed': passed, 'total': len(tests), 'error': err}))
"""

_LCB_FN = r"""
import json, sys
meta = json.load(open('/sbx/meta.json'))
t = meta['tests'][int(sys.argv[1])]
ns = {'__name__': 'solution'}
src = open('/sbx/solution.py').read()
prelude = 'from typing import *\nimport collections, heapq, math, bisect, itertools, functools, string, re\nfrom collections import *\nfrom functools import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\n'
exec(prelude + src, ns)
args = [json.loads(line) for line in t['input'].split('\n') if line.strip()]
fn = getattr(ns['Solution'](), meta['func_name'])
out = fn(*args)
exp = json.loads(t['output'])
print('OK' if out == exp or (isinstance(out, tuple) and list(out) == exp) else f'got {str(out)[:60]} want {str(exp)[:60]}')
"""

_MBPP_RUNNER = r"""
import json
src = open('/sbx/solution.py').read()
test = open('/sbx/test.py').read()
ns = {'__name__': '__main__'}
try:
    exec(compile(src, 'solution.py', 'exec'), ns)
    exec(compile(test, 'test.py', 'exec'), ns)
    print(json.dumps({'passed': 1, 'total': 1, 'error': None}))
except Exception as e:
    print(json.dumps({'passed': 0, 'total': 1, 'error': (type(e).__name__ + ': ' + str(e))[:200]}))
"""


def _crux_equal(pred: str, gold: str) -> bool:
    import ast
    p = pred.strip().strip("`").strip()
    if "==" in p:  # "f(x) == [1, 2]" -> "[1, 2]"
        p = p.split("==")[-1].strip()
    try:
        return ast.literal_eval(p) == ast.literal_eval(gold)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return p.replace(" ", "") == gold.replace(" ", "")


async def score(q: dict, text: str) -> dict:
    kind = q["kind"]
    if kind == "number":
        ok, got = score_number(text, q["answer"], q.get("tol", 1e-6))
        return {"correct": ok, "got": got, "expected": _fmt_num(q["answer"])}
    if kind == "text":
        ok, got = score_text(text, q["accept"])
        return {"correct": ok, "got": got, "expected": " / ".join(q["accept"])}
    if kind == "rules":
        ok, detail = check_rules(text, q["rules"])
        return {"correct": ok, "got": detail, "expected": ", ".join(" ".join(str(x) for x in r) for r in q["rules"])}
    if kind == "code":
        res = await run_code(_strip_fences(text), q["tests"])
        return {"correct": res["passed"] == res["total"], "got": f"{res['passed']}/{res['total']} tests passed" + (f" · {res['error']}" if res.get("error") else ""),
                "expected": f"{len(q['tests'])} unit tests pass"}
    if kind == "choice":
        got = extract_choice(text, q["labels"])
        return {"correct": got == q["answer"], "got": got or "(no option picked)", "expected": f"{q['answer']}) {_option(q, q['answer'])}"}
    if kind == "math":
        boxed = benchmarks.last_boxed(text)
        pred = boxed if boxed is not None else final_answer(text)[0]
        return {"correct": benchmarks.math_equal(pred, q["answer"]), "got": pred[:200], "expected": q["answer"]}
    if kind == "mbpp":
        code = _strip_fences(text)
        imports = "\n".join(q.get("imports") or [])
        res = await sandbox({"solution.py": imports + "\n" + code, "test.py": q["test"], "run.py": _MBPP_RUNNER}, "run.py", 1, cpu=20, wall=30)
        ok = res["passed"] == res["total"]
        return {"correct": ok, "got": "all hidden tests passed" if ok else ("failed" + (f" · {res['error']}" if res.get("error") else "")),
                "expected": "hidden unit tests pass"}
    if kind == "crux":
        ans, _ = final_answer(text)
        return {"correct": _crux_equal(ans, q["answer"]), "got": ans[:200], "expected": q["answer"]}
    if kind == "lcb":
        code = _strip_fences(text)
        files = {"solution.py": code, "meta.json": json.dumps({"tests": q["tests"], "func_name": q.get("func_name")}),
                 "run.py": _LCB_RUNNER, "fn.py": _LCB_FN}
        res = await sandbox(files, "run.py", len(q["tests"]), cpu=120, wall=150, mem_mb=1024)
        ok = res["passed"] == res["total"]
        cut = "```" in text and text.count("```") % 2 == 1  # answer ended inside the code block
        return {"correct": ok, "got": ("answer was cut off before the code was finished" if cut and not ok else
                                       f"{res['passed']}/{res['total']} tests passed" + (f" · {res['error']}" if res.get("error") else "")),
                "expected": f"all {len(q['tests'])} tests pass"}
    if kind == "humaneval":
        code = _strip_fences(text)
        imports = "\n".join(l for l in q["q"].splitlines() if l.startswith(("import ", "from ")))
        res = await run_code(imports + "\n" + code, [q["test"], f"check({q['entry_point']})"], cpu=20, wall=30)
        ok = res["passed"] == res["total"]
        return {"correct": ok, "got": "all hidden tests passed" if ok else ("failed hidden tests" + (f" · {res['error']}" if res.get("error") else "")),
                "expected": "hidden unit tests pass"}
    return {"correct": False, "got": "", "expected": ""}


def _option(q: dict, label: str) -> str:
    try:
        return q["options"][q["labels"].index(label)][:120]
    except (ValueError, IndexError):
        return ""


def extract_choice(text: str, labels: list[str]) -> str | None:
    """The option letter the model picked: from 'Answer: C', else the last standalone label."""
    alts = "|".join(map(re.escape, labels))
    inline = re.findall(rf"(?i:answer)(?:\s+is)?\s*[:：]?\s*\**\s*\(?({alts})\)?(?![A-Za-z0-9])", _unlatex(text))
    if inline:
        return inline[-1]
    ans, explicit = final_answer(text)
    pattern = rf"(?<![A-Za-z0-9])\(?({alts})\)?(?![A-Za-z0-9])"
    m = re.search(pattern, ans)
    if m:
        return m.group(1)
    found = re.findall(pattern, _unlatex(text)[-300:])
    return found[-1] if found else None


def _fmt_num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def prompt_for(q: dict) -> str:
    if q["kind"] in ("number", "text"):
        return q["q"] + ANSWER_HINT
    if q["kind"] == "code":
        return q["q"] + CODE_HINT
    if q["kind"] == "choice":
        opts = "\n".join(f"{l}) {o}" for l, o in zip(q["labels"], q["options"]))
        return f"{q['q']}\n\nOptions:\n{opts}\n\nThink it through if needed, then give your final answer on the last line as: Answer: <letter>"
    if q["kind"] == "math":
        return q["q"] + "\n\nSolve the problem. Put your final answer in \\boxed{}."
    if q["kind"] == "humaneval":
        return ("Complete the following Python function. Reply with the complete function (signature included) "
                f"in a single ```python code block, no explanations.\n\n```python\n{q['q']}```")
    if q["kind"] == "mbpp":
        return (f"{q['q']}\nYour code should pass this test:\n{q['example']}\n\n"
                "Reply with only the complete Python function in a single ```python code block.")
    if q["kind"] == "crux":
        return (f"Here is a Python function:\n```python\n{q['q']}\n```\nWhat does `f({q['input']})` return?\n\n"
                "Think it through step by step if needed, then give the exact return value as a Python literal on the last line, "
                "in the form: Answer: <value>")
    if q["kind"] == "lcb":
        if q.get("starter"):
            how = f"Complete this starter code (keep the class and method names):\n```python\n{q['starter']}\n```"
        else:
            how = "Your program must read from standard input and print to standard output."
        return (f"Solve this programming problem in Python 3.\n\n{q['q']}\n\n{how}\n\n"
                "Reply with only the complete solution in a single ```python code block.")
    return q["q"]


# ---------- Runs ----------

def _row(r) -> dict:
    d = dict(r)
    for k in ("suites", "results", "summary", "options"):
        d[k] = json.loads(d[k]) if d.get(k) else None
    d["think"] = bool(d["think"])
    d["options"] = d.get("options") or {}
    return d


def get(run_id: int) -> dict | None:
    with db.connect() as c:
        r = c.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    return _row(r) if r else None


def list_runs(limit: int = 100) -> list[dict]:
    with db.connect() as c:
        rows = c.execute("SELECT * FROM eval_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def _save(run_id: int, **fields) -> None:
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
    with db.connect() as c:
        c.execute(f"UPDATE eval_runs SET {sets} WHERE id = ?", (*vals, run_id))


def create(body: dict) -> int:
    from . import gpuguard
    gpuguard.check_start("A quality test")
    model = Path(str(body.get("model") or "")).name
    if not (lab.MODELS_DIR / model).is_file():
        raise ValueError("Model not found")
    known = all_suites()
    keys = []
    for spec in body.get("suites") or []:
        key, limit = parse_spec(str(spec))
        if key in known or (key.startswith("bench:") and benchmarks.is_downloaded(key[6:])) \
                or (key.startswith("fresh:") and key[6:] in fresh.SETS):
            keys.append(f"{key}@{max(1, min(20000, limit))}" if limit else key)
    if not keys:
        raise ValueError("Pick at least one test set")
    code_sets = ("coding", "bench:humaneval", "bench:humaneval_plus", "bench:mbpp_plus", "bench:livecodebench", "fresh:code")
    if any(parse_spec(k)[0] in code_sets for k in keys) and not sandbox_available():
        raise ValueError("The coding tests need the sandbox (bubblewrap). Re-run the installer.")
    budget = body.get("think_budget")
    # Fresh questions are generated from this seed: new every run, but reproducible.
    seed = int(body.get("seed") or random.randint(1, 2**31 - 1))
    options = {"think_budget": int(budget) if budget else None, "seed": seed}
    with db.connect() as c:
        return c.execute(
            "INSERT INTO eval_runs (created_at, status, model, think, suites, results, options) VALUES (?, 'queued', ?, ?, ?, '[]', ?)",
            (time.time(), model, int(bool(body.get("think"))), json.dumps(keys), json.dumps(options)),
        ).lastrowid


def delete(run_id: int) -> None:
    with db.connect() as c:
        c.execute("DELETE FROM eval_runs WHERE id = ? AND status NOT IN ('queued', 'running')", (run_id,))


def cancel(run_id: int) -> None:
    _cancel.add(run_id)


def summarize(results: list[dict], suites: list[str]) -> dict:
    per = {}
    for key in suites:
        rs = [r for r in results if r["suite"] == key]
        if rs:
            per[key] = {"correct": sum(r["correct"] for r in rs), "total": len(rs),
                        "score": sum(r["correct"] for r in rs) / len(rs)}
    done = [r for r in results if r.get("seconds") is not None]
    tokens = sum(r.get("tokens") or 0 for r in done)
    secs = sum(r["seconds"] for r in done)
    energy = sum(r.get("energy_j") or 0 for r in done) or None
    return {
        "per_suite": per,
        "budget_hits": sum(1 for r in results if r.get("budget_hit")),
        "overall": sum(p["score"] for p in per.values()) / len(per) if per else None,
        "correct": sum(p["correct"] for p in per.values()), "total": sum(p["total"] for p in per.values()),
        "seconds": round(secs, 1), "tokens": tokens,
        "avg_tps": (sum((r.get("tps") or 0) * (r.get("tokens") or 0) for r in done) / tokens) if tokens else None,
        "energy_j": round(energy, 1) if energy else None,
        "seconds_per_question": round(secs / len(done), 2) if done else None,
    }


async def run(run_id: int, llm: httpx.AsyncClient) -> None:
    rec = get(run_id)
    rt = lab.read_runtime()
    use_live = rec["model"] == rt["model"]
    results: list[dict] = []
    status, error = "done", None
    think = rec["think"]
    budget = rec["options"].get("think_budget") if think else None
    # Thinking: the model's recommended sampling. Not thinking: greedy, for repeatable scores.
    sampling = generate.recommended_sampling(rec["model"], True) if think else {"temperature": 0.0}
    async with lab._lock:
        lab.busy = "eval"
        _save(run_id, status="running")
        proc = None
        client = llm
        samples: list = []
        stop = asyncio.Event()
        sampler = None
        try:
            if not use_live:
                await lab.llm_service("stop")
                proc = await arena._start_temp_server(rec["model"], rt["kv"])
                client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{arena.TEMP_PORT}", timeout=httpx.Timeout(10, read=600))
                if not await lab.wait_ready(client, timeout=240):
                    raise RuntimeError(f"Couldn't load {rec['model']}. It may not fit in memory.")
            sampler = asyncio.create_task(lab._sample_gpu(samples, stop))
            for key in rec["suites"]:
                for i, q in enumerate(questions_for(key, rec["options"].get("seed") or 0)):
                    if run_id in _cancel:
                        raise asyncio.CancelledError
                    t0 = time.monotonic()
                    long_answer = q["kind"] in ("code", "humaneval", "math", "mbpp", "lcb", "crux")
                    info = await generate.complete(
                        client, [{"role": "user", "content": prompt_for(q)}], think=think, sampling=sampling, seed=42,
                        # Contest problems need room: small models "think in comments" and ran out at 2k.
                        budget=budget, max_tokens=(16384 if think else (4096 if q["kind"] == "lcb" else 2048 if long_answer else 1024)),
                        answer_tokens=4096 if q["kind"] == "lcb" else 2048 if long_answer else 1024,
                        should_stop=lambda: run_id in _cancel)  # Stop acts immediately, mid-answer
                    text = info["content"]
                    t1 = time.monotonic()
                    timings = info["timings"] or {}
                    win = [s for s in samples if t0 <= s[0] <= t1]
                    power = sum(s[1] for s in win if s[1] is not None) / len(win) if win and win[0][1] is not None else None
                    graded = await score(q, text)
                    results.append({
                        "suite": key, "index": i, "question": q["q"][:3000], "kind": q["kind"], "tag": q.get("tag"),
                        "output": text[:6000], "reasoning_chars": len(info["reasoning"]),
                        "reasoning_tokens": info["reasoning_tokens"], "budget_hit": info["budget_hit"],
                        **graded,
                        "seconds": round(t1 - t0, 2), "tokens": timings.get("predicted_n"),
                        "tps": timings.get("predicted_per_second"),
                        "energy_j": round(power * (t1 - t0), 1) if power else None,
                        "truncated": info["finish_reason"] == "length",
                    })
                    _save(run_id, results=results, summary=summarize(results, rec["suites"]))
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as e:  # noqa: BLE001
            status, error = "error", str(e)[:400]
            log.warning("eval %s failed: %s", run_id, e)
        finally:
            stop.set()
            if sampler:
                await sampler
            if client is not llm:
                await client.aclose()
            await arena._stop(proc)
            _save(run_id, status=status, error=error, results=results,
                  summary=summarize(results, rec["suites"]), finished_at=time.time())
            _cancel.discard(run_id)
            if not use_live:
                try:
                    await lab.llm_service("start")
                    await lab.wait_ready(llm, timeout=300)
                except Exception as e:  # noqa: BLE001
                    log.error("couldn't restart chat model after eval: %s", e)
            lab.busy = None
