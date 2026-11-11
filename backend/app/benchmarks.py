"""Established benchmark sets, downloaded from Hugging Face (public, no login).

Each benchmark is converted once to our question format and stored locally
(/opt/blocky/data/benchmarks/<key>.json), so runs work offline afterwards.
Questions are shuffled with a fixed seed: "first 50" is always the same 50,
which keeps runs comparable, and a 50-question run is a prefix of a 100 run.
"""
import asyncio
import json
import random
import re
import time
import urllib.parse

import httpx

from . import config

DIR = config.DATA_DIR / "benchmarks"
API = "https://datasets-server.huggingface.co/rows"
HEADERS = {"User-Agent": "blockyOS/0.1"}

CATALOG = {
    "gsm8k": {
        "title": "GSM8K", "icon": "calculator", "color": "#7c5cff", "difficulty": "Easy",
        "about": "Grade-school math word problems. The classic; good models score 85–95%.",
        "sources": [("openai/gsm8k", "main", "test")], "baseline": None,
    },
    "arc": {
        "title": "ARC-Challenge", "icon": "flask", "color": "#22d3ee", "difficulty": "Medium",
        "about": "Hard grade-school science questions, multiple choice (4 options).",
        "sources": [("allenai/ai2_arc", "ARC-Challenge", "test")], "baseline": 0.25,
    },
    "mmlu_pro": {
        "title": "MMLU-Pro", "icon": "graduation", "color": "#ffa94d", "difficulty": "Hard",
        "about": "University-level exam questions in 14 subjects, 10 options each. Tests knowledge + reasoning.",
        "sources": [("TIGER-Lab/MMLU-Pro", "default", "test")], "baseline": 0.10,
    },
    "math500": {
        "title": "MATH-500", "icon": "sigma", "color": "#ff5c9d", "difficulty": "Hard",
        "about": "Competition math from algebra to precalculus, 5 difficulty levels. Answers in LaTeX.",
        "sources": [("HuggingFaceH4/MATH-500", "default", "test")], "baseline": None,
    },
    "humaneval": {
        "title": "HumanEval", "icon": "code", "color": "#34d399", "difficulty": "Medium",
        "about": "164 Python programming tasks, each checked by hidden unit tests in the sandbox.",
        "sources": [("openai/openai_humaneval", "openai_humaneval", "test")], "baseline": None,
    },
    "aime": {
        "title": "AIME 2024 + 2025", "icon": "trophy", "color": "#e879f9", "difficulty": "Very hard",
        "about": "Math olympiad qualifier problems (answers 0–999). Thinking makes a huge difference here.",
        "sources": [("HuggingFaceH4/aime_2024", "default", "train"), ("MathArena/aime_2025", "default", "train")], "baseline": None,
    },
    # ----- Coding -----
    "humaneval_plus": {
        "title": "HumanEval+", "icon": "code", "color": "#34d399", "difficulty": "Medium", "group": "coding",
        "about": "HumanEval's 164 tasks with ~80× more hidden tests. Catches code that only looks right.",
        "sources": [("evalplus/humanevalplus", "default", "test")], "baseline": None,
    },
    "mbpp_plus": {
        "title": "MBPP+", "icon": "code", "color": "#22d3ee", "difficulty": "Easy", "group": "coding",
        "about": "378 everyday Python tasks (strings, lists, math) with extra hidden tests.",
        "sources": [("evalplus/mbppplus", "default", "test")], "baseline": None,
    },
    "cruxeval": {
        "title": "CRUXEval (output)", "icon": "brain", "color": "#a78bfa", "difficulty": "Medium", "group": "coding",
        "about": "800 short functions: predict exactly what they return. Tests understanding code, not writing it.",
        "sources": [("cruxeval-org/cruxeval", "default", "test")], "baseline": None,
    },
    "livecodebench": {
        "title": "LiveCodeBench", "icon": "trophy", "color": "#f97316", "difficulty": "Hard", "group": "coding",
        "about": "Recent contest problems (LeetCode, AtCoder, Codeforces) with hidden tests. Newer problems = less chance the model memorized them.",
        "file": "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl", "baseline": None,
    },
}

CUSTOM_FILE = DIR / "custom.json"  # user-imported benchmarks (Hugging Face datasets with a column mapping)


def custom_catalog() -> dict:
    try:
        return json.loads(CUSTOM_FILE.read_text())
    except (OSError, ValueError):
        return {}


def catalog() -> dict:
    return {**CATALOG, **custom_catalog()}

downloads: dict[str, dict] = {}  # key -> {"done": rows, "total": rows, "status": ..., "error": ...}


def path(key: str):
    return DIR / f"{key}.json"


def is_downloaded(key: str) -> bool:
    return path(key).is_file()


def load(key: str) -> list[dict]:
    return json.loads(path(key).read_text())


_counts: dict[str, tuple[float, int, dict]] = {}


def _info(key: str) -> tuple[int, dict]:
    """Question count + extra facts (e.g. date range), cached by file mtime."""
    p = path(key)
    m = p.stat().st_mtime
    if key not in _counts or _counts[key][0] != m:
        qs = load(key)
        extra = {}
        dates = sorted(q["date"] for q in qs if q.get("date"))
        if dates:
            extra = {"date_from": dates[0][:10], "date_to": dates[-1][:10]}
        _counts[key] = (m, len(qs), extra)
    return _counts[key][1], _counts[key][2]


def summary() -> list[dict]:
    out = []
    for key, b in catalog().items():
        have = is_downloaded(key)
        n, extra = _info(key) if have else (None, {})
        out.append({"key": f"bench:{key}", "id": key, "title": b["title"], "icon": b["icon"], "color": b["color"],
                    "about": b["about"], "difficulty": b["difficulty"], "baseline": b.get("baseline"),
                    "group": b.get("group", "custom" if b.get("custom") else "general"), "custom": bool(b.get("custom")),
                    "downloaded": have, "count": n, "download": downloads.get(key), **extra})
    return out


# ---------- Converters: dataset row -> our question format ----------

def _gsm8k(row):
    ans = row["answer"].split("####")[-1].strip().replace(",", "")
    return {"q": row["question"], "kind": "number", "answer": float(ans)}


def _arc(row):
    ch = row["choices"]
    labels = list(ch["label"])
    return {"q": row["question"], "kind": "choice", "options": list(ch["text"]), "labels": labels, "answer": row["answerKey"]}


def _mmlu_pro(row):
    opts = row["options"] if isinstance(row["options"], list) else json.loads(row["options"].replace("'", '"'))
    labels = [chr(ord("A") + i) for i in range(len(opts))]
    return {"q": row["question"], "kind": "choice", "options": opts, "labels": labels, "answer": row["answer"], "tag": row.get("category")}


def _math500(row):
    return {"q": row["problem"], "kind": "math", "answer": row["answer"], "tag": f"{row.get('subject')} · level {row.get('level')}"}


def _humaneval(row):
    return {"q": row["prompt"], "kind": "humaneval", "test": row["test"], "entry_point": row["entry_point"], "tag": row.get("task_id")}


def _aime(row):
    return {"q": row["problem"], "kind": "number", "answer": float(str(row["answer"]).strip())}


def _humaneval_plus(row):
    return {"q": row["prompt"], "kind": "humaneval", "test": row["test"], "entry_point": row["entry_point"], "tag": row.get("task_id")}


def _mbpp_plus(row):
    tests = row["test_list"] if isinstance(row["test_list"], list) else ast_literal(row["test_list"])
    imports = row.get("test_imports") or []
    if isinstance(imports, str):
        imports = ast_literal(imports) or []
    return {"q": row["prompt"], "kind": "mbpp", "example": tests[0] if tests else "", "test": row["test"],
            "imports": imports, "tag": f"MBPP/{row.get('task_id')}"}


def _cruxeval(row):
    return {"q": row["code"], "kind": "crux", "input": row["input"], "answer": row["output"], "tag": row.get("id")}


def _aime_safe(row):
    return _aime(row)


def ast_literal(s):
    import ast
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


CONVERT = {"gsm8k": _gsm8k, "arc": _arc, "mmlu_pro": _mmlu_pro, "math500": _math500, "humaneval": _humaneval, "aime": _aime,
           "humaneval_plus": _humaneval_plus, "mbpp_plus": _mbpp_plus, "cruxeval": _cruxeval}


# ---------- LiveCodeBench (direct file; hidden tests are compressed) ----------

class _DataOnlyUnpickler:
    """Hidden tests are pickled. Pickle can run code, so we only allow plain data:
    any attempt to load a class or function is refused."""
    @staticmethod
    def loads(data: bytes):
        import io
        import pickle

        class U(pickle.Unpickler):
            def find_class(self, module, name):
                raise pickle.UnpicklingError(f"refused to load {module}.{name}")
        return U(io.BytesIO(data)).load()


def _lcb_tests(raw: str) -> list:
    import base64
    import zlib
    try:
        return json.loads(raw)
    except ValueError:
        obj = _DataOnlyUnpickler.loads(zlib.decompress(base64.b64decode(raw)))
        return json.loads(obj) if isinstance(obj, str) else []


MAX_LCB_TESTS = 15          # public + a sample of hidden tests per problem
MAX_TEST_BYTES = 200_000    # skip giant inputs (keeps the local file small)


def _lcb(row) -> dict:
    tests = [t for t in json.loads(row["public_test_cases"]) if len(t.get("input", "")) < MAX_TEST_BYTES]
    hidden = [t for t in _lcb_tests(row["private_test_cases"]) if len(t.get("input", "")) < MAX_TEST_BYTES]
    tests += hidden[: max(0, MAX_LCB_TESTS - len(tests))]
    meta = json.loads(row.get("metadata") or "{}")
    return {"q": row["question_content"], "kind": "lcb", "title": row.get("question_title"), "starter": row.get("starter_code") or "",
            "tests": tests, "func_name": meta.get("func_name"), "date": row.get("contest_date"),
            "tag": f"{row.get('platform')} · {row.get('difficulty')} · {(row.get('contest_date') or '')[:10]}"}


async def _download_file(url: str, prog: dict) -> list[dict]:
    rows = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=300), headers=HEADERS, follow_redirects=True) as client:
        async with client.stream("GET", url) as r:
            r.raise_for_status()
            prog["total"] = int(r.headers.get("content-length") or 1)
            buf = b""
            async for chunk in r.aiter_bytes(1 << 20):
                prog["done"] += len(chunk)
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if line.strip():
                        try:
                            rows.append(_lcb(json.loads(line)))
                        except (ValueError, KeyError, TypeError):
                            continue
            if buf.strip():
                try:
                    rows.append(_lcb(json.loads(buf)))
                except (ValueError, KeyError, TypeError):
                    pass
    return rows


# ---------- User-imported benchmarks ----------

def _letter(answer, options: list, labels: list) -> str | None:
    a = str(answer).strip()
    if a in labels:
        return a
    if a.isdigit() and int(a) < len(labels):
        return labels[int(a)]
    for l, o in zip(labels, options):
        if str(o).strip().lower() == a.lower():
            return l
    return None


def make_converter(m: dict):
    """Generic converter from a user's column mapping."""
    def conv(row):
        q = str(row[m["question"]])
        ans = row[m["answer"]]
        kind = m.get("kind", "auto")
        opts = row.get(m["options"]) if m.get("options") else None
        if isinstance(opts, str):
            opts = ast_literal(opts) or json.loads(opts)
        if isinstance(opts, dict):  # e.g. {"text": [...], "label": [...]}
            opts = opts.get("text") or list(opts.values())[0]
        if kind == "auto":
            if opts:
                kind = "choice"
            elif re.fullmatch(r"\s*-?[\d,]+(\.\d+)?\s*", str(ans)):
                kind = "number"
            elif "\\" in str(ans) or "$" in str(ans):
                kind = "math"
            else:
                kind = "text"
        if kind == "choice":
            labels = [chr(65 + i) for i in range(len(opts))]
            letter = _letter(ans, opts, labels)
            if letter is None:
                raise ValueError("answer not among options")
            return {"q": q, "kind": "choice", "options": [str(o) for o in opts], "labels": labels, "answer": letter}
        if kind == "number":
            return {"q": q, "kind": "number", "answer": float(str(ans).replace(",", "").strip())}
        if kind == "math":
            return {"q": q, "kind": "math", "answer": str(ans)}
        return {"q": q, "kind": "text", "accept": [s.strip().lower() for s in str(ans).split("/") if s.strip()]}
    return conv


def add_custom(body: dict) -> str:
    repo = str(body.get("repo", "")).strip()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Use owner/dataset, e.g. openai/gsm8k")
    mapping = {k: body.get(k) for k in ("question", "answer", "options", "kind")}
    if not mapping["question"] or not mapping["answer"]:
        raise ValueError("Pick which column holds the question and which holds the answer")
    if mapping["kind"] not in ("auto", "number", "text", "choice", "math"):
        mapping["kind"] = "auto"
    key = "custom_" + re.sub(r"[^a-z0-9]+", "_", f"{repo}_{body.get('config', '')}_{body.get('split', '')}".lower()).strip("_")[:60]
    entry = {"title": (body.get("name") or repo.split("/")[-1])[:60], "icon": "user", "color": "#a78bfa", "difficulty": "Custom",
             "about": f"Imported from huggingface.co/datasets/{repo} ({body.get('config')}/{body.get('split')}).",
             "sources": [[repo, body.get("config") or "default", body.get("split") or "test"]], "baseline": None,
             "custom": True, "mapping": mapping, "limit": int(body.get("limit") or 2000)}
    cat = custom_catalog()
    cat[key] = entry
    DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_FILE.write_text(json.dumps(cat, indent=1))
    return key


async def hf_preview(repo: str) -> dict:
    """Configs/splits and a sample row, so the user can map columns."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Use owner/dataset, e.g. openai/gsm8k")
    q = urllib.parse.quote(repo)
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as c:
        r = await c.get(f"https://datasets-server.huggingface.co/splits?dataset={q}")
        if r.status_code in (401, 403, 404):
            raise ValueError("Dataset not found, or it needs a login (gated datasets aren't supported).")
        r.raise_for_status()
        splits = [{"config": s["config"], "split": s["split"]} for s in r.json().get("splits", [])]
        if not splits:
            raise ValueError("No downloadable splits found")
        s0 = splits[0]
        r2 = await c.get(f"https://datasets-server.huggingface.co/first-rows?dataset={q}&config={urllib.parse.quote(s0['config'])}&split={s0['split']}")
        r2.raise_for_status()
        rows = [x["row"] for x in r2.json().get("rows", [])[:3]]
    return {"splits": splits, "columns": list(rows[0].keys()) if rows else [],
            "sample": [{k: str(v)[:300] for k, v in row.items()} for row in rows]}


async def _fetch_all(client: httpx.AsyncClient, ds: str, cfg: str, split: str, prog: dict, max_rows: int = 50_000) -> list[dict]:
    rows, offset, total = [], 0, None
    while (total is None or offset < total) and offset < max_rows:
        url = f"{API}?dataset={urllib.parse.quote(ds)}&config={urllib.parse.quote(cfg)}&split={split}&offset={offset}&length=100"
        data = None
        for attempt in range(4):
            try:
                r = await client.get(url)
                if r.status_code in (429, 500, 502, 503, 504):  # throttled or busy: back off and retry
                    raise httpx.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                data = r.json()  # can fail on an empty/HTML body when the server is overloaded
                break
            except (httpx.HTTPError, ValueError):
                if attempt < 3:
                    await asyncio.sleep(2 + attempt * 3)
        await asyncio.sleep(0.25)  # be polite: big sets are ~100+ requests
        if data is None:
            # One page the server can't serve (usually a malformed row): skip it rather than lose everything.
            if total is None:
                raise RuntimeError("Hugging Face couldn't serve this dataset (try again later)")
            prog["skipped"] = prog.get("skipped", 0) + 100
            if prog["skipped"] > max(500, total * 0.1):
                raise RuntimeError(f"Too many pages failed (from row {offset}); try again later")
            offset += 100
            prog["done"] += 100
            continue
        total = data.get("num_rows_total", 0)
        batch = [x["row"] for x in data.get("rows", [])]
        if not batch:
            break
        rows += batch
        offset += len(batch)
        prog["done"] += len(batch)
        prog["total"] = max(prog["total"], prog["done"] + max(0, total - offset))
    return rows


async def download(key: str) -> None:
    b = catalog()[key]
    prog = downloads[key] = {"done": 0, "total": 1, "status": "downloading", "error": None, "started": time.time()}
    try:
        if b.get("file"):  # single big file (LiveCodeBench): converted while streaming
            qs = await _download_file(b["file"], prog)
        else:
            async with httpx.AsyncClient(timeout=60, headers=HEADERS) as client:
                rows = []
                for ds, cfg, split in b["sources"]:
                    rows += await _fetch_all(client, ds, cfg, split, prog, b.get("limit") or 50_000)
                    if b.get("limit") and len(rows) >= b["limit"]:
                        rows = rows[: b["limit"]]
                        break
            conv = make_converter(b["mapping"]) if b.get("custom") else CONVERT[key]
            qs = []
            for row in rows:
                try:
                    qs.append(conv(row))
                except (KeyError, ValueError, TypeError, AttributeError):
                    continue  # skip malformed rows
        if not qs:
            raise RuntimeError("No usable questions found. Check the column mapping.")
        random.Random(f"blocky-{key}").shuffle(qs)
        DIR.mkdir(parents=True, exist_ok=True)
        tmp = path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(qs))
        tmp.replace(path(key))
        prog["status"] = "done"
    except Exception as e:  # noqa: BLE001
        prog["status"], prog["error"] = "error", str(e)[:300]


def delete(key: str) -> None:
    path(key).unlink(missing_ok=True)
    downloads.pop(key, None)
    cat = custom_catalog()
    if key in cat:  # imported benchmarks disappear entirely
        del cat[key]
        CUSTOM_FILE.write_text(json.dumps(cat, indent=1))


# ---------- Grading helpers for the new question kinds ----------

def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, handling nested braces."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def normalize_math(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{\s*([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|textbf|mbox)\{([^{}]*)\}", r"\1", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("\\displaystyle", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""), ("\\$", ""),
                 ("degrees", ""), ("°", "")]:
        s = s.replace(a, b)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)  # "x = 5" -> "5"
    s = s.replace(" ", "").rstrip(".")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)  # \frac12 -> \frac{1}{2}
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)  # 1,000 -> 1000
    return s


def _to_number(s: str) -> float | None:
    """Evaluate simple LaTeX numbers: 3, -2.5, \\frac{1}{2}, 2\\sqrt{3}, \\pi/4. Safe: no eval()."""
    import ast
    import math
    t = s
    for _ in range(3):
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", t)
        t = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", t)
    t = t.replace("\\pi", "pi").replace("^", "**").replace("\\cdot", "*").replace("\\times", "*")
    t = re.sub(r"(\d)(\(|sqrt|pi)", r"\1*\2", t)
    t = re.sub(r"\)(\(|sqrt|pi|\d)", r")*\1", t)
    try:
        node = ast.parse(t, mode="eval").body
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id == "pi":
            return math.pi
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ValueError
            return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b, ast.Div: a / b if b else math.inf, ast.Pow: a ** b}[type(n.op)]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "sqrt" and len(n.args) == 1:
            return math.sqrt(ev(n.args[0]))
        raise ValueError
    try:
        return ev(node)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


def math_equal(pred: str, gold: str) -> bool:
    a, b = normalize_math(pred), normalize_math(gold)
    if not a:
        return False
    if a == b:
        return True
    na, nb = _to_number(a), _to_number(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= 1e-6 * max(1.0, abs(nb))
    return False
