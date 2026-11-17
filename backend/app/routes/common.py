"""Helpers shared by the route modules, and the app state they use (set up in main's lifespan)."""
import asyncio
import json

from fastapi import HTTPException


class _State:
    """The shared llama-server client and the set that keeps background tasks referenced (so they aren't collected)."""
    llm = None
    background: set = set()


state = _State()


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    state.background.add(task)
    task.add_done_callback(state.background.discard)


def _bad(e: Exception):
    raise HTTPException(400, str(e))
