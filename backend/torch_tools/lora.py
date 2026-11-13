"""LoRA fine-tuning of a Hugging Face checkpoint, merged and saved for GGUF conversion (run inside the ML venv).

Data: a Hugging Face dataset (instruction/output, prompt/response, question/answer or chat `messages` columns are
recognised) or a local .jsonl with the same fields. The chat template of the model is applied; loss is computed on
the answer tokens only. Progress (step, loss, learning rate) is appended to <out>/progress.jsonl as it trains.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def to_messages(row):
    if row.get("messages"):
        return row["messages"]
    for q, a in (("instruction", "output"), ("prompt", "response"), ("question", "answer"), ("input", "output")):
        if row.get(q) and row.get(a):
            user = row[q] + (f"\n\n{row['input']}" if q == "instruction" and row.get("input") else "")
            return [{"role": "user", "content": user}, {"role": "assistant", "content": row[a]}]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="HF dataset id (optionally id:config) or a .jsonl path")
    ap.add_argument("--split", default="train")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prog = open(out / "progress.jsonl", "a", buffering=1)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.data.endswith(".jsonl"):
        ds = load_dataset("json", data_files=a.data, split="train")
    else:
        name, _, cfgname = a.data.partition(":")
        ds = load_dataset(name, cfgname or None, split=a.split)
    ds = ds.shuffle(seed=a.seed).select(range(min(a.rows, len(ds))))

    examples = []
    for row in ds:
        msgs = to_messages(row)
        if not msgs or msgs[-1]["role"] != "assistant":
            continue
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = prompt + msgs[-1]["content"] + (tok.eos_token or "")
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        f_ids = tok(full, add_special_tokens=False).input_ids[: a.max_len]
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
        if any(x != -100 for x in labels):
            examples.append((f_ids, labels[: len(f_ids)]))
    if not examples:
        sys.exit("No usable rows: the dataset needs instruction/output, prompt/response, question/answer or messages columns")
    print(f"{len(examples)} training examples", file=sys.stderr, flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if bf16 else torch.float32).to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in a.targets.split(",") if t in names] or "all-linear"
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.steps // 10)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    prog.write(json.dumps({"event": "start", "examples": len(examples), "trainable": trainable, "total": total, "targets": targets, "bf16": bf16}) + "\n")

    model.train()
    t0 = time.time()
    for step in range(a.steps):
        batch = random.sample(examples, min(a.batch, len(examples)))
        L = max(len(x[0]) for x in batch)
        ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in batch], device=dev)
        lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in batch], device=dev)
        att = (torch.arange(L, device=dev)[None, :] < torch.tensor([len(x[0]) for x in batch], device=dev)[:, None]).long()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=bf16):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        prog.write(json.dumps({"step": step + 1, "loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0], "t": round(time.time() - t0, 1)}) + "\n")

    model = model.merge_and_unload()
    model.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    prog.write(json.dumps({"event": "merged", "seconds": round(time.time() - t0, 1)}) + "\n")


if __name__ == "__main__":
    main()
