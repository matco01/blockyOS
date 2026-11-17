"""SQLite storage for conversations and messages.

One small file on disk, no server. A new connection per call keeps it
thread-safe without any locking code."""
import json
import re
import sqlite3
import time
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    reasoning        TEXT,
    tokens           INTEGER,
    tokens_per_sec   REAL,
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);

-- Long-term memory: short facts about the user, visible and editable in the UI.
CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    content          TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'auto' CHECK (source IN ('auto', 'user')),
    conversation_id  INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);

-- Benchmark runs (llama-bench) with hardware fingerprint and per-test results.
CREATE TABLE IF NOT EXISTS bench_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   REAL NOT NULL,
    finished_at  REAL,
    status       TEXT NOT NULL,
    model        TEXT NOT NULL,
    params       TEXT NOT NULL,
    hardware     TEXT,
    results      TEXT,
    log          TEXT,
    error        TEXT
);

-- Device-wide settings (key -> JSON value).
CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

-- Full-text search indexes (built into SQLite, no extra software).
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, content='messages', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, content='memories', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
"""

DEFAULT_SETTINGS = {
    "memory_enabled": True,  # learn facts from conversations automatically
    # Energy & cost (experiments): optional wall-power meter and prices, so runs report energy and cost per token.
    "power_meter_url": "",      # e.g. http://192.168.1.50/rpc/Switch.GetStatus?id=0 (Shelly), a Tasmota or any JSON URL
    "power_meter_field": "",    # dotted path to the watts in the reply, when it isn't a known meter
    "device_price": None,       # what this device cost
    "electricity_price_kwh": None,
    "currency": "",
    "remote_token": "",
    "rpc_worker": {"enabled": False, "expose": False, "port": 50052},  # this device as a llama.cpp RPC worker (rpcworker.py)
    "anthropic_api_key": "",    # the agent's Claude key (optional); never returned by the API
    "agent_model": "claude-opus-5-5",         # set = other devices may control this one with this token (Settings → Remote access)
}


@contextmanager
def connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as c:
        c.execute("PRAGMA journal_mode = WAL")
        c.executescript(SCHEMA)
        # Migrations for databases created by older versions.
        cols = {r[1] for r in c.execute("PRAGMA table_info(messages)")}
        if "stats" not in cols:
            c.execute("ALTER TABLE messages ADD COLUMN stats TEXT")
        # A crash mid-benchmark would leave runs stuck as 'running'.
        c.execute("UPDATE bench_runs SET status = 'error', error = 'Interrupted' WHERE status IN ('running', 'queued')")
        # Index messages written before search existed (cheap at this scale).
        c.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")


def create_conversation(title: str) -> int:
    now = time.time()
    with connect() as c:
        cur = c.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        return cur.lastrowid


def conversation_exists(conv_id: int) -> bool:
    with connect() as c:
        return c.execute("SELECT 1 FROM conversations WHERE id = ?", (conv_id,)).fetchone() is not None


def list_conversations() -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def rename_conversation(conv_id: int, title: str) -> None:
    with connect() as c:
        c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))


def delete_conversation(conv_id: int) -> None:
    with connect() as c:
        c.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


def get_messages(conv_id: int, limit: int | None = None) -> list[dict]:
    with connect() as c:
        if limit:
            rows = c.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE conversation_id = ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                (conv_id, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conv_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["stats"] = json.loads(d["stats"]) if d.get("stats") else None
            out.append(d)
        return out


def add_message(
    conv_id: int,
    role: str,
    content: str,
    reasoning: str | None = None,
    tokens: int | None = None,
    tokens_per_sec: float | None = None,
    stats: dict | None = None,
) -> int:
    now = time.time()
    with connect() as c:
        cur = c.execute(
            "INSERT INTO messages (conversation_id, role, content, reasoning, tokens, tokens_per_sec, stats, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (conv_id, role, content, reasoning, tokens, tokens_per_sec, json.dumps(stats) if stats else None, now),
        )
        c.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
        return cur.lastrowid


# ---------- Settings ----------

def get_settings() -> dict:
    with connect() as c:
        stored = {r["key"]: json.loads(r["value"]) for r in c.execute("SELECT key, value FROM settings")}
    return {**DEFAULT_SETTINGS, **stored}


def update_settings(values: dict) -> dict:
    with connect() as c:
        for key, value in values.items():
            if key in DEFAULT_SETTINGS:
                c.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, json.dumps(value)),
                )
    return get_settings()


# ---------- Memories ----------

def list_memories() -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM memories ORDER BY id ASC")]


def count_memories() -> int:
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


def add_memory(content: str, source: str = "auto", conversation_id: int | None = None) -> dict:
    now = time.time()
    with connect() as c:
        cur = c.execute(
            "INSERT INTO memories (content, source, conversation_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (content, source, conversation_id, now, now),
        )
        return dict(c.execute("SELECT * FROM memories WHERE id = ?", (cur.lastrowid,)).fetchone())


def update_memory(mem_id: int, content: str) -> bool:
    with connect() as c:
        cur = c.execute(
            "UPDATE memories SET content = ?, source = 'user', updated_at = ? WHERE id = ?",
            (content, time.time(), mem_id),
        )
        return cur.rowcount > 0


def delete_memory(mem_id: int) -> None:
    with connect() as c:
        c.execute("DELETE FROM memories WHERE id = ?", (mem_id,))


def delete_all_memories() -> None:
    with connect() as c:
        c.execute("DELETE FROM memories")


def _fts_query(text: str) -> str | None:
    """Turn free text into a safe FTS5 query: any of the words, each quoted."""
    words = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2]
    return " OR ".join(f'"{w}"' for w in words[:30]) or None


def search_memories(text: str, limit: int) -> list[dict]:
    q = _fts_query(text)
    if not q:
        return []
    with connect() as c:
        rows = c.execute(
            "SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.rowid "
            "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
            (q, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def recent_memories(limit: int) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


# ---------- Search past conversations ----------

def search_messages(text: str, limit: int = 30) -> list[dict]:
    q = _fts_query(text)
    if not q:
        return []
    with connect() as c:
        rows = c.execute(
            "SELECT m.conversation_id, c.title, m.role, "
            "snippet(messages_fts, 0, '[', ']', '…', 12) AS snippet "
            "FROM messages_fts f JOIN messages m ON m.id = f.rowid "
            "JOIN conversations c ON c.id = m.conversation_id "
            "WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
            (q, limit * 3),
        ).fetchall()
    # One result per conversation (best match first).
    seen, results = set(), []
    for r in rows:
        if r["conversation_id"] not in seen:
            seen.add(r["conversation_id"])
            results.append(dict(r))
    return results[:limit]
