"""Projects: what a person or team is making (a model or a whole AI system), and everything that belongs to it.

A project holds experiments (a model and its measured versions) and code (research projects). Both keep working on
their own; a project only groups them. Deleting a project unlinks its contents, it never deletes them.
"""
import time

from . import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    goal        TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
"""
# What can belong to a project -> the table that holds it.
MEMBERS = {"experiment": "experiments", "code": "research_projects"}


def init() -> None:
    """Run after the experiment and research-project tables exist."""
    with db.connect() as c:
        c.executescript(SCHEMA)
        for table in MEMBERS.values():
            if "project_id" not in [r[1] for r in c.execute(f"PRAGMA table_info({table})")]:
                c.execute(f"ALTER TABLE {table} ADD COLUMN project_id INTEGER")


def _clean(name, goal) -> tuple[str, str]:
    name = " ".join(str(name or "").split())[:80]
    if not name:
        raise ValueError("Give the project a name")
    return name, str(goal or "").strip()[:500]


def _experiments(c, where: str, args: tuple) -> list[dict]:
    rows = c.execute(f"""SELECT e.id, e.name, e.model, e.created_at, COUNT(r.id) AS runs,
                                COALESCE(SUM(r.status = 'done'), 0) AS done,
                                COALESCE(SUM(r.status IN ('queued', 'running')), 0) AS active,
                                MAX(r.created_at) AS last_run
                         FROM experiments e LEFT JOIN exp_runs r ON r.experiment_id = e.id
                         WHERE {where} GROUP BY e.id ORDER BY e.id DESC""", args).fetchall()
    out = []
    # Only the three numbers a card shows, extracted by SQLite: a run's full results are megabytes of JSON.
    pick = ("SELECT id, label, json_extract(results, '$.mmlu.acc') AS mmlu, json_extract(results, '$.speed.tg') AS tg, "
            "json_extract(results, '$.ppl') AS ppl FROM exp_runs WHERE experiment_id = ? AND status = 'done'{} ORDER BY id {} LIMIT 1")
    for row in rows:
        e = dict(row)
        first = c.execute(pick.format("", "ASC"), (e["id"],)).fetchone()
        # "Latest" is the latest change: reruns of a setup ("Repeat of …") measure noise, not a new version.
        last = c.execute(pick.format(" AND label NOT LIKE 'Repeat of%'", "DESC"), (e["id"],)).fetchone()
        e["baseline"] = {"label": first["label"], "results": _key_numbers(first)} if first else None
        e["latest"] = {"label": last["label"], "results": _key_numbers(last)} if last and last["id"] != first["id"] else None
        out.append(e)
    return out


def _key_numbers(row) -> dict | None:
    out = {"mmlu": row["mmlu"], "tg": row["tg"], "ppl": row["ppl"]}
    return out if any(v is not None for v in out.values()) else None


def _code(c, where: str, args: tuple) -> list[dict]:
    return [dict(r) for r in c.execute(f"SELECT id, name, created_at FROM research_projects WHERE {where} ORDER BY id DESC", args)]


def _summary(c, row) -> dict:
    p = dict(row)
    p["experiments"] = _experiments(c, "e.project_id = ?", (p["id"],))
    p["code"] = _code(c, "project_id = ?", (p["id"],))
    p["last_activity"] = max([p["updated_at"]] + [e["last_run"] or e["created_at"] for e in p["experiments"]])
    return p


def list_all() -> dict:
    with db.connect() as c:
        projects = [_summary(c, r) for r in c.execute("SELECT * FROM projects")]
        unassigned = {"experiments": _experiments(c, "e.project_id IS NULL", ()), "code": _code(c, "project_id IS NULL", ())}
    projects.sort(key=lambda p: -p["last_activity"])
    return {"projects": projects, "unassigned": unassigned}


def get(pid: int) -> dict | None:
    with db.connect() as c:
        row = c.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
        return _summary(c, row) if row else None


def create(name, goal="") -> int:
    name, goal = _clean(name, goal)
    now = time.time()
    with db.connect() as c:
        return c.execute("INSERT INTO projects (name, goal, created_at, updated_at) VALUES (?, ?, ?, ?)", (name, goal, now, now)).lastrowid


def update(pid: int, body: dict) -> dict:
    current = get(pid)
    if not current:
        raise ValueError("Project not found")
    name, goal = _clean(body.get("name", current["name"]), body.get("goal", current["goal"]))
    with db.connect() as c:
        c.execute("UPDATE projects SET name = ?, goal = ?, updated_at = ? WHERE id = ?", (name, goal, time.time(), pid))
    return get(pid)


def delete(pid: int) -> None:
    """Removes the project; its experiments and code stay, outside any project."""
    with db.connect() as c:
        for table in MEMBERS.values():
            c.execute(f"UPDATE {table} SET project_id = NULL WHERE project_id = ?", (pid,))
        c.execute("DELETE FROM projects WHERE id = ?", (pid,))


def assign(kind: str, item_id: int, pid: int | None) -> None:
    """Puts an experiment or code into a project (or takes it out with pid=None)."""
    table = MEMBERS.get(kind)
    if not table:
        raise ValueError(f"Kind must be one of {', '.join(MEMBERS)}")
    with db.connect() as c:
        if pid is not None and not c.execute("SELECT 1 FROM projects WHERE id = ?", (pid,)).fetchone():
            raise ValueError("Project not found")
        if not c.execute(f"SELECT 1 FROM {table} WHERE id = ?", (item_id,)).fetchone():
            raise ValueError(f"{kind.capitalize()} not found")
        c.execute(f"UPDATE {table} SET project_id = ? WHERE id = ?", (pid, item_id))
        if pid is not None:
            c.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (time.time(), pid))
