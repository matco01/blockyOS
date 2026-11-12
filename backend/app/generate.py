"""One generation engine for chat, Arena and Tests, with a thinking budget.

Budget forcing: if the model is still thinking after `budget` tokens, we stop
it, append its reasoning plus a short "time's up" note and the end-of-thinking
tag to the exact prompt (via llama-server's /apply-template), and let it
continue straight into the answer with the raw /completion endpoint.

Also applies model-recommended sampling when the caller didn't set any:
small models' thinking gets much more erratic with generic settings.
"""
import json
import math
import time
from collections.abc import AsyncIterator

import httpx

BUDGET_NOTE = "\n\nI'm out of thinking time, so I'll stop here and give my best final answer now.\n</think>\n\n"

# Qwen's published recommendations (thinking / non-thinking).
QWEN_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
QWEN_PLAIN = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


def recommended_sampling(model_name: str | None, think: bool) -> dict:
    if model_name and "qwen" in model_name.lower():
        return dict(QWEN_THINK if think else QWEN_PLAIN)
    return {}


async def stream(client: httpx.AsyncClient, messages: list[dict], *, think: bool, sampling: dict | None = None,
                 max_tokens: int | None = None, budget: int | None = None, answer_tokens: int = 2048,
                 seed: int | None = None, extra: dict | None = None,
                 should_stop=None) -> AsyncIterator[tuple[str, object]]:
    """Yields ("reasoning", text), ("content", text), ("budget", n_tokens), and finally ("done", info).
    `should_stop()` is checked on every chunk: returning True aborts at once (raises CancelledError)."""
    import asyncio

    def check():
        if should_stop and should_stop():
            raise asyncio.CancelledError
    sampling = dict(sampling or {})
    if seed is not None:
        sampling["seed"] = seed
    payload = {"messages": messages, "stream": True, "chat_template_kwargs": {"enable_thinking": think}, **sampling, **(extra or {})}
    if think and budget:
        payload["max_tokens"] = budget + answer_tokens
    elif max_tokens:
        payload["max_tokens"] = max_tokens

    started = time.monotonic()
    first = first_content = None
    reasoning: list[str] = []
    content: list[str] = []
    r_tokens = 0
    timings: dict = {}
    finish = None
    hit_budget = False

    async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode(errors="replace")[:300]
            raise httpx.HTTPStatusError(f"model server error {resp.status_code}: {body}", request=resp.request, response=resp)
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            check()
            chunk = json.loads(data)
            timings = chunk.get("timings") or timings
            for ch in chunk.get("choices", []):
                finish = ch.get("finish_reason") or finish
                lp = (ch.get("logprobs") or {}).get("content")
                d = ch.get("delta") or {}
                if lp and d.get("content"):  # per-word confidence, answer text only (not thinking)
                    yield "probs", [{"t": x["token"], "p": round(math.exp(x["logprob"]), 4),
                                     "alts": [[a["token"], round(math.exp(a["logprob"]), 4)] for a in (x.get("top_logprobs") or [])
                                              if a["token"] != x["token"]][:4]} for x in lp]
                if d.get("reasoning_content"):
                    first = first or time.monotonic()
                    reasoning.append(d["reasoning_content"])
                    r_tokens += 1
                    yield "reasoning", d["reasoning_content"]
                if d.get("content"):
                    first = first or time.monotonic()
                    first_content = first_content or time.monotonic()
                    content.append(d["content"])
                    yield "content", d["content"]
            if think and budget and r_tokens >= budget and not content:
                hit_budget = True
                break  # closing the stream makes llama-server stop generating

    phase1_end = time.monotonic()
    # Ran out of room (budget or max_tokens) while still thinking: force the answer.
    if think and not content and reasoning and (hit_budget or finish == "length"):
        hit_budget = True
        yield "budget", r_tokens
        t = await client.post("/apply-template", json={"messages": messages, "chat_template_kwargs": {"enable_thinking": True}})
        prompt = t.json()["prompt"]
        if not prompt.rstrip().endswith("<think>"):
            prompt += "<think>\n"
        prompt += "".join(reasoning) + BUDGET_NOTE
        body = {"prompt": prompt, "n_predict": answer_tokens, "stream": True, "cache_prompt": True,
                **{k: v for k, v in sampling.items() if k in ("temperature", "top_p", "top_k", "min_p", "seed", "repeat_penalty", "presence_penalty")}}
        p1 = timings
        async with client.stream("POST", "/completion", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                check()
                chunk = json.loads(line[6:])
                if chunk.get("content"):
                    first_content = first_content or time.monotonic()
                    content.append(chunk["content"])
                    yield "content", chunk["content"]
                if chunk.get("stop"):
                    t2 = chunk.get("timings") or {}
                    n1, n2 = p1.get("predicted_n") or r_tokens, t2.get("predicted_n") or 0
                    # When we cut phase 1 short, the server never sent its timings: use the wall clock.
                    ms1 = p1.get("predicted_ms") or ((phase1_end - (first or started)) * 1000)
                    ms2 = t2.get("predicted_ms") or 0
                    timings = {**p1, "predicted_n": n1 + n2, "predicted_ms": ms1 + ms2,
                               "predicted_per_second": (n1 + n2) / ((ms1 + ms2) / 1000) if ms1 + ms2 else p1.get("predicted_per_second")}
                    finish = "length" if chunk.get("stopped_limit") else "stop"
                    break

    yield "done", {
        "content": "".join(content), "reasoning": "".join(reasoning), "timings": timings,
        "reasoning_tokens": r_tokens, "budget_hit": hit_budget, "finish_reason": finish,
        "ttft_ms": round((first - started) * 1000) if first else None,
        "think_ms": round(((first_content or time.monotonic()) - first) * 1000) if first and reasoning else None,
        "total_ms": round((time.monotonic() - started) * 1000),
    }


async def complete(client: httpx.AsyncClient, messages: list[dict], **kw) -> dict:  # same kwargs as stream()
    """Non-streaming convenience wrapper: returns the final info dict."""
    info: dict = {}
    async for kind, val in stream(client, messages, **kw):
        if kind == "done":
            info = val  # type: ignore[assignment]
    return info
