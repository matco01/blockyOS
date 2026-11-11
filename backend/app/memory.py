"""Long-term memory.

Two jobs:
  1. recall():  pick which stored facts go into the prompt.
  2. extract(): after each exchange, ask the model whether there's a new lasting
                fact about the user worth keeping.

No embedding model or vector DB: with few memories we include them all (most
accurate, ~free); beyond that, SQLite full-text search picks relevant ones.
Keeps VRAM free for the chat model, which matters on 4 GB cards.
"""
import json
import logging
import re

import httpx

from . import db

log = logging.getLogger("blocky.memory")

INCLUDE_ALL_UP_TO = 40   # memories; ~1–2k tokens
SEARCH_LIMIT = 15
RECENT_LIMIT = 10
MAX_MEMORY_CHARS = 300

EXTRACT_PROMPT = """You maintain a long-term memory of facts about the user of a personal AI assistant.

Read the latest exchange below. Decide if it reveals any NEW, LASTING information about the user worth remembering in future conversations:
- personal details (name, location, job, family, pets)
- preferences (likes, dislikes, how they want answers)
- ongoing projects, goals, plans
- anything the user explicitly asks you to remember

Do NOT save:
- temporary things (today's mood, a one-off question, small talk)
- facts about the world, or anything the assistant said
- anything already in "Known facts"
- passwords, card numbers or other secrets

Write each fact as one short sentence in third person, starting with "User" (e.g. "User's name is Sam.", "User prefers short answers.").
Most exchanges contain nothing worth saving: then return an empty list.

Known facts:
{known}

Latest exchange:
User: {user}
Assistant: {assistant}"""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"memories": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["memories"],
}


def recall(user_message: str) -> list[dict]:
    """Memories to show the model for this message."""
    if db.count_memories() <= INCLUDE_ALL_UP_TO:
        return db.list_memories()
    chosen = {m["id"]: m for m in db.search_memories(user_message, SEARCH_LIMIT)}
    for m in db.recent_memories(RECENT_LIMIT):
        chosen.setdefault(m["id"], m)
    return sorted(chosen.values(), key=lambda m: m["id"])  # stable order helps prompt caching


def format_for_prompt(memories: list[dict]) -> str:
    if not memories:
        return ""
    lines = "\n".join(f"- {m['content']}" for m in memories)
    return (
        "\n\nWhat you remember about the user from earlier conversations "
        "(use it naturally when relevant; don't recite it):\n" + lines
    )


def _normalize(text: str) -> str:
    return re.sub(r"[^\w ]", "", text.lower()).strip()


async def extract(llm: httpx.AsyncClient, user: str, assistant: str, conversation_id: int) -> list[dict]:
    """Learn new facts from one exchange. Returns the memories that were added."""
    existing = db.list_memories()
    known = "\n".join(f"- {m['content']}" for m in existing[-60:]) or "(none yet)"
    prompt = EXTRACT_PROMPT.format(known=known, user=user[:4000], assistant=assistant[:2000])

    try:
        r = await llm.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False},
            # Grammar-constrained output: the model can only produce valid JSON in this shape.
            "response_format": {"type": "json_schema", "json_schema": {"schema": EXTRACT_SCHEMA}},
        }, timeout=60)
        r.raise_for_status()
        facts = json.loads(r.json()["choices"][0]["message"]["content"]).get("memories", [])
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        log.warning("memory extraction failed: %s", e)
        return []

    seen = {_normalize(m["content"]) for m in existing}
    added = []
    for fact in facts:
        fact = " ".join(str(fact).split())[:MAX_MEMORY_CHARS]
        key = _normalize(fact)
        if len(key) < 8 or key in seen:
            continue
        seen.add(key)
        added.append(db.add_memory(fact, "auto", conversation_id))
    return added
