"""Long-context suite modeled on RULER (Hsieh et al., 2024): synthetic tasks that need more than finding one needle.

  niah_multikey    4 needles with different keys in the text; retrieve the value of one (the others are distractors)
  niah_multivalue  one key with 4 values spread through the text; retrieve all of them (recall)
  vt               variable tracking: a chain of assignments (VAR A = 12345, VAR B = VAR A, ...) spread through the
                   text, plus a decoy chain; name every variable that holds the value (recall)
  cwe              common words: a long list of words where 10 appear far more often; name those 10 (recall)

Every sample is generated from a fixed seed, so every run of an experiment sees identical prompts. Scores are
per task and length (mean of per-sample scores in [0, 1], with a 95% CI); the effective context length is the
longest tested length whose average score is still >= 85.6%, RULER's threshold.
"""
import asyncio
import math
import random
import re

import httpx

LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]
TASKS = ("niah_multikey", "niah_multivalue", "vt", "cwe")
SEED = 7777
THRESHOLD = 0.856
_WORDS = ("amber anchor apple arrow aspen badge banjo basil beacon birch bishop blossom bonnet bramble breeze brick bronze "
          "cabin cactus camel canyon carbon cedar cello chalk cherry cinder clover cobalt comet copper coral cotton crane "
          "crystal dagger daisy delta denim dune eagle ember falcon fennel fern fiddle flint fossil garnet ginger glacier "
          "granite harbor hazel heron hollow honey indigo iris ivory jasper juniper kettle kiwi lagoon lantern lemon lilac "
          "linen lotus lynx magnet maple marble meadow melon mint mosaic nectar nickel nutmeg oasis olive onyx orchid otter "
          "pebble pepper pine plum poppy prism quartz quill raven reef ribbon robin saffron sage sapphire satin shadow "
          "silver sparrow spruce summit tango thistle thunder topaz tulip tundra velvet violet walnut willow zephyr").split()


def _num(rng: random.Random) -> str:
    return str(rng.randrange(1_000_000, 9_999_999))


def _key(rng: random.Random) -> str:
    return f"{rng.choice(_WORDS)}-{rng.choice(_WORDS)}"


def _insert(hay: str, items: list[str], rng: random.Random) -> str:
    """Put each sentence at a random sentence boundary of the haystack."""
    cuts = sorted(rng.uniform(0.02, 0.98) for _ in items)
    out, last = [], 0
    for frac, item in zip(cuts, items):
        c = hay.rfind(". ", last, int(len(hay) * frac))
        c = (c + 2) if c >= last else int(len(hay) * frac)
        out += [hay[last:c], item + " "]
        last = c
    out.append(hay[last:])
    return "".join(out)


def make_sample(task: str, length: int, chars_per_tok: float, text: str, rng: random.Random) -> dict:
    n_chars = int(length * chars_per_tok)
    start = rng.randrange(0, max(1, len(text) - n_chars - 1))
    hay = text[start:start + n_chars]
    if task == "niah_multikey":
        keys = list(dict.fromkeys(_key(rng) for _ in range(12)))[:4]
        vals = [_num(rng) for _ in keys]
        doc = _insert(hay, [f"One of the special magic numbers for {k} is: {v}." for k, v in zip(keys, vals)], rng)
        i = rng.randrange(len(keys))
        return {"prompt": f"{doc}\n\nWhat is the special magic number for {keys[i]} mentioned in the text above? Reply with the number only.",
                "answers": [vals[i]], "max_tokens": 24}
    if task == "niah_multivalue":
        k, vals = _key(rng), [_num(rng) for _ in range(4)]
        doc = _insert(hay, [f"One of the special magic numbers for {k} is: {v}." for v in vals], rng)
        return {"prompt": f"{doc}\n\nWhat are all the special magic numbers for {k} mentioned in the text above? List every one, numbers only.",
                "answers": vals, "max_tokens": 64}
    if task == "vt":
        names = list(dict.fromkeys("".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) for _ in range(20)))
        chain, decoy = names[:5], names[5:10]
        v, dv = _num(rng), _num(rng)
        stmts = [f"VAR {chain[0]} = {v}."] + [f"VAR {chain[i]} = VAR {chain[i - 1]}." for i in range(1, 5)]
        dstm = [f"VAR {decoy[0]} = {dv}."] + [f"VAR {decoy[i]} = VAR {decoy[i - 1]}." for i in range(1, 5)]
        # Interleave the two chains at random, each kept in order (a variable is defined before it's used).
        a, b, items = list(stmts), list(dstm), []
        while a or b:
            items.append((a if a and (not b or rng.random() < 0.5) else b).pop(0))
        doc = _insert(hay, items, rng)
        return {"prompt": f"{doc}\n\nFind all variables that are assigned the value {v} in the text above, directly or through other variables. "
                          f"List their names, separated by spaces.", "answers": chain, "max_tokens": 64}
    # cwe: a numbered word list, 10 frequent words (10x as often) among many rare ones
    words = rng.sample(_WORDS, len(_WORDS))
    common, rare = words[:10], words[10:]
    n_words = length // 6                       # a numbered line ("123. word") is about 5-6 tokens
    r = max(1, n_words // (100 + len(rare)))    # each rare word r times, each common word 10r times
    lst = common * (10 * r) + rare * r
    rng.shuffle(lst)
    body = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(lst))
    return {"prompt": f"Below is a numbered list of words. In it, some words appear much more often than others.\n{body}\n\n"
                      "Question: What are the 10 most common words in the list above? Answer with the 10 words only, separated by commas.",
            "answers": common, "max_tokens": 64}


def score(task: str, reply: str, answers: list[str]) -> float:
    r = reply.lower()
    if task in ("niah_multikey",):
        return 1.0 if answers[0] in re.sub(r"[^\d]", " ", r).split() or answers[0] in r else 0.0
    if task == "vt":
        toks = set(re.findall(r"[a-z]{5}", r))
        return sum(a.lower() in toks for a in answers) / len(answers)
    return sum(a.lower() in r for a in answers) / len(answers)


def _ci(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return 1.96 * math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


async def run(client: httpx.AsyncClient, text: str, chars_per_tok: float, ctx: int, samples: int, should_stop, progress) -> dict:
    from . import generate
    lengths = [L for L in LENGTHS if L + 1200 <= ctx]
    rng = random.Random(SEED)
    cells = []
    total = len(lengths) * len(TASKS) * samples
    done = 0
    for L in lengths:
        for task in TASKS:
            scores = []
            for _ in range(samples):
                if should_stop():
                    raise asyncio.CancelledError
                s = make_sample(task, L, chars_per_tok, text, rng)
                info = await generate.complete(client, [{"role": "user", "content": s["prompt"]}], think=False, sampling={"temperature": 0.0},
                                               seed=1, max_tokens=s["max_tokens"], should_stop=should_stop)
                scores.append(score(task, info.get("content") or "", s["answers"]))
                done += 1
                progress(done, total)
            cells.append({"task": task, "length": L, "score": round(sum(scores) / len(scores), 4), "ci95": round(_ci(scores), 4), "n": len(scores)})
    by_len = {str(L): round(sum(c["score"] for c in cells if c["length"] == L) / len(TASKS), 4) for L in lengths}
    by_task = {t: round(sum(c["score"] for c in cells if c["task"] == t) / len(lengths), 4) for t in TASKS} if lengths else {}
    eff = None
    for L in lengths:
        if by_len[str(L)] >= THRESHOLD:
            eff = L
        else:
            break
    allx = [c["score"] for c in cells]
    return {"lengths": lengths, "tasks": list(TASKS), "samples": samples, "cells": cells, "by_length": by_len, "by_task": by_task,
            "score": round(sum(allx) / len(allx), 4) if allx else None, "effective_length": eff, "threshold": THRESHOLD,
            "method": f"RULER-style synthetic tasks ({', '.join(TASKS)}), {samples} samples per task and length, seed {SEED}, greedy"}
