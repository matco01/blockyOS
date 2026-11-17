"""Conversations, long-term memory and streaming chat."""
import asyncio
import time
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import config, db, generate, lab, memory
from .common import _sse, state

router = APIRouter()


@router.get("/api/conversations")
def list_conversations():
    return db.list_conversations()


@router.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: int):
    if not db.conversation_exists(conv_id):
        raise HTTPException(404, "Conversation not found")
    return db.get_messages(conv_id)


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)


@router.patch("/api/conversations/{conv_id}")
def rename_conversation(conv_id: int, body: RenameIn):
    db.rename_conversation(conv_id, body.title.strip())
    return {"ok": True}


@router.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: int):
    db.delete_conversation(conv_id)
    return {"ok": True}


@router.get("/api/search")
def search(q: str):
    return db.search_messages(q)


class MemoryIn(BaseModel):
    content: str = Field(min_length=1, max_length=memory.MAX_MEMORY_CHARS)


@router.get("/api/memories")
def list_memories():
    return db.list_memories()


@router.post("/api/memories")
def add_memory(body: MemoryIn):
    return db.add_memory(body.content.strip(), "user")


@router.patch("/api/memories/{mem_id}")
def update_memory(mem_id: int, body: MemoryIn):
    if not db.update_memory(mem_id, body.content.strip()):
        raise HTTPException(404, "Memory not found")
    return {"ok": True}


@router.delete("/api/memories/{mem_id}")
def delete_memory(mem_id: int):
    db.delete_memory(mem_id)
    return {"ok": True}


@router.delete("/api/memories")
def delete_all_memories():
    db.delete_all_memories()
    return {"ok": True}


class Sampling(BaseModel):
    """Generation parameters a researcher may want to control. None = model/server default."""
    temperature: float | None = Field(None, ge=0, le=5)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=0, le=1000)
    min_p: float | None = Field(None, ge=0, le=1)
    repeat_penalty: float | None = Field(None, ge=0.5, le=3)
    presence_penalty: float | None = Field(None, ge=-2, le=2)
    max_tokens: int | None = Field(None, ge=1, le=262_144)
    seed: int | None = Field(None, ge=-1)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=50_000)
    conversation_id: int | None = None
    think: bool = False  # let the model "reason" first (slower, sometimes smarter)
    sampling: Sampling | None = None
    system_prompt: str | None = Field(None, max_length=20_000)  # replaces the default persona
    think_budget: int | None = Field(4096, ge=64, le=65_536)  # max thinking tokens before forcing an answer
    confidence: bool = False  # stream per-word probabilities (for the confidence view)


_token_cache: dict[str, int] = {}


async def _count_tokens(text: str) -> int | None:
    """Token count via llama-server's tokenizer (cached; the system prompt rarely changes)."""
    if text in _token_cache:
        return _token_cache[text]
    try:
        r = await state.llm.post("/tokenize", json={"content": text}, timeout=3)
        n = len(r.json().get("tokens", []))
    except (httpx.HTTPError, ValueError):
        return None
    if len(_token_cache) > 64:
        _token_cache.clear()
    _token_cache[text] = n
    return n


def _system_prompt(user_message: str, use_memory: bool, base: str | None = None) -> str:
    prompt = (base.strip() if base and base.strip() else config.SYSTEM_PROMPT) + (
        f"\n\nCurrent date (for your reference; only mention it if asked or relevant): "
        f"{datetime.now():%A, %B %d, %Y}."
    )
    if use_memory:
        prompt += memory.format_for_prompt(memory.recall(user_message))
    return prompt


@router.post("/api/chat")
async def chat(body: ChatIn):
    conv_id = body.conversation_id
    if conv_id is None:
        conv_id = db.create_conversation(body.message.strip()[:60] or "New chat")
    elif not db.conversation_exists(conv_id):
        raise HTTPException(404, "Conversation not found")

    settings = db.get_settings()
    use_memory = settings["memory_enabled"]

    db.add_message(conv_id, "user", body.message)
    history = db.get_messages(conv_id, limit=config.MAX_HISTORY_MESSAGES)
    system_text = _system_prompt(body.message, use_memory, body.system_prompt)
    system_n = await _count_tokens(system_text)
    messages = [{"role": "system", "content": system_text}] + [
        {"role": m["role"], "content": m["content"]} for m in history
    ]
    sampling = body.sampling.model_dump(exclude_none=True) if body.sampling else {}
    # No explicit settings: use the model's recommended ones (makes small models' thinking far less erratic).
    used_sampling = sampling or generate.recommended_sampling(lab.read_runtime().get("model"), body.think)
    budget = body.think_budget if body.think else None

    async def stream():
        yield _sse({"type": "meta", "conversation_id": conv_id})
        if lab.busy:
            msg = {"benchmark": "A benchmark is running. Chat is paused until it finishes.",
                   "arena": "An Arena match is running. Chat is paused until it finishes.",
                   "eval": "A quality test is running. Chat is paused until it finishes.",
                   "quantlab": "The quantization lab is running. Chat is paused until it finishes.",
                   "throughput": "A throughput test is running. Chat is paused until it finishes.",
                   "experiment": "An experiment profile is running. Chat is paused until it finishes.",
                   "weights": "New model weights are being built. Chat is paused until it finishes.",
                   "engine": "An inference engine is being compiled. Chat is paused until it finishes.",
                   "scan": "A layer sensitivity scan is running. Chat is paused until it finishes.",
                   "torch": "A PyTorch job (activations or fine-tuning) is using the GPU. Chat is paused until it finishes."}.get(
                lab.busy, "The model is being switched. Try again in a moment.")
            yield _sse({"type": "error", "message": msg})
            return
        content, reasoning = [], []
        timings: dict = {}
        started = time.monotonic()
        first_token: float | None = None
        finished = False
        budget_hit = False

        def stats() -> dict:
            s = {
                "ttft_ms": round((first_token - started) * 1000) if first_token else None,
                "budget_hit": budget_hit, "think_budget": budget,
                "total_ms": round((time.monotonic() - started) * 1000),
                "prompt_n": timings.get("prompt_n"),
                "prompt_per_second": timings.get("prompt_per_second"),
                "cache_n": timings.get("cache_n"),
                "predicted_n": timings.get("predicted_n"),
                "predicted_per_second": timings.get("predicted_per_second"),
                "think": body.think,
                "system_n": system_n,  # system prompt + memories (for the context meter)
            }
            if sampling:
                s["sampling"] = sampling
            elif used_sampling:
                s["sampling_recommended"] = used_sampling
            if body.system_prompt:
                s["custom_system_prompt"] = True
            return s

        try:
            extra = {"logprobs": True, "top_logprobs": 5} if body.confidence else None
            async for kind, val in generate.stream(state.llm, messages, think=body.think, sampling=used_sampling,
                                                   budget=budget, answer_tokens=4096, extra=extra):
                if kind == "probs":
                    yield _sse({"type": "probs", "items": val})
                    continue
                if kind in ("reasoning", "content") and first_token is None:
                    first_token = time.monotonic()
                if kind == "reasoning":
                    reasoning.append(val)
                    yield _sse({"type": "reasoning", "text": val})
                elif kind == "content":
                    content.append(val)
                    yield _sse({"type": "token", "text": val})
                elif kind == "budget":
                    budget_hit = True
                    yield _sse({"type": "budget", "tokens": val})
                elif kind == "done":
                    timings = val["timings"] or {}
            finished = True
        except httpx.ConnectError:
            yield _sse({"type": "error", "message": "The AI model isn't running. Is blocky-llm started?"})
            return
        except httpx.HTTPStatusError as e:
            msg = "The model is still loading. Try again in a few seconds." if "503" in str(e) else str(e)[:300]
            yield _sse({"type": "error", "message": msg})
            return
        except httpx.HTTPError as e:
            yield _sse({"type": "error", "message": f"Lost connection to the model: {e.__class__.__name__}"})
            return
        finally:
            # Save whatever we got, even if the user navigated away mid-answer.
            if content or reasoning:
                db.add_message(
                    conv_id, "assistant", "".join(content),
                    reasoning="".join(reasoning) or None,
                    tokens=timings.get("predicted_n"),
                    tokens_per_sec=timings.get("predicted_per_second"),
                    stats=stats(),
                )

        yield _sse({
            "type": "done",
            "tokens": timings.get("predicted_n"),
            "tokens_per_sec": timings.get("predicted_per_second"),
            "seconds": round(time.monotonic() - started, 2),
            "stats": stats(),
        })

        # Learn from this exchange. Runs as its own task so it completes even if
        # the browser disconnects; we just stop waiting for it in that case.
        if finished and use_memory and content and not lab.busy:
            task = asyncio.create_task(
                memory.extract(state.llm, body.message, "".join(content), conv_id)
            )
            state.background.add(task)
            task.add_done_callback(state.background.discard)
            added = await asyncio.shield(task)
            if added:
                yield _sse({"type": "memory", "added": added})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
