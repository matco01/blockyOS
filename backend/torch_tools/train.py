"""Full-parameter training of a causal language model (run inside the ML venv): fine-tuning or pretraining.

  finetune   start from the checkpoint's weights and update all of them (not adapters, unlike lora.py)
  pretrain   start from random weights with the checkpoint's architecture and tokenizer (optionally fewer layers)

Data: a Hugging Face dataset (optionally id:config) or a local .txt / .jsonl file. Rows with chat fields
(instruction/output, prompt/response, question/answer, messages) train on the answer tokens only, through the model's
chat template. Rows with a `text` field (and .txt files) are concatenated and cut into max_len blocks.

A fixed share of the examples (--eval_frac, chosen with --seed before training) is held out. Held-out loss and
perplexity are measured before the first step and every --eval_every steps, so the result shows generalization, not
just the training loss.

One GPU: plain PyTorch. Several GPUs or machines: launch with torchrun; the model is sharded with FSDP (FULL_SHARD,
one unit per decoder layer, bf16 mixed precision where supported), and each rank draws its own batches. Without GPUs
(FSDP needs an accelerator) several processes use DistributedDataParallel over gloo instead.

Progress goes to <out>/progress.jsonl (rank 0): a start event, one line per step (loss, lr, tokens, time), eval lines,
and an end event. The trained model is saved as a Hugging Face checkpoint in <out>/model.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path


def to_messages(row):
    if row.get("messages"):
        return row["messages"]
    for q, a in (("instruction", "output"), ("prompt", "response"), ("question", "answer"), ("input", "output")):
        if row.get(q) and row.get(a):
            user = row[q] + (f"\n\n{row['input']}" if q == "instruction" and row.get("input") else "")
            return [{"role": "user", "content": user}, {"role": "assistant", "content": row[a]}]
    return None


def load_rows(data, split, rows, seed):
    if data.endswith(".txt"):
        return [{"text": Path(data).read_text(encoding="utf-8")}]
    from datasets import load_dataset
    if data.endswith(".jsonl"):
        ds = load_dataset("json", data_files=data, split="train")
    else:
        name, _, cfgname = data.partition(":")
        ds = load_dataset(name, cfgname or None, split=split)
    ds = ds.shuffle(seed=seed).select(range(min(rows, len(ds))))
    return list(ds)


def build_examples(rows, tok, max_len):
    """-> list of (input_ids, labels) and the data kind ('chat' or 'text')."""
    chat = [m for m in (to_messages(r) for r in rows) if m and m[-1]["role"] == "assistant"]
    examples = []
    if chat and getattr(tok, "chat_template", None):
        for msgs in chat:
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            full = prompt + msgs[-1]["content"] + (tok.eos_token or "")
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            f_ids = tok(full, add_special_tokens=False).input_ids[:max_len]
            labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
            if any(x != -100 for x in labels):
                examples.append((f_ids, labels[:len(f_ids)]))
        return examples, "chat"
    texts = [r["text"] for r in rows if isinstance(r.get("text"), str) and r["text"].strip()]
    if not texts:
        sys.exit("No usable rows: the data needs a text field, or chat fields (instruction/output, prompt/response, question/answer, messages)")
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
    stream = []
    for t in texts:
        stream += tok(t, add_special_tokens=False).input_ids + eos
    for i in range(0, len(stream) - max_len + 1, max_len):  # packed blocks; the tail shorter than max_len is dropped
        block = stream[i:i + max_len]
        examples.append((block, list(block)))
    return examples, "text"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("finetune", "pretrain"), default="finetune")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=4, help="sequences per GPU per micro-step")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation micro-steps per optimizer step")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--eval_frac", type=float, default=0.02)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--layers", type=int, default=0, help="pretrain only: use this many decoder layers instead of the checkpoint's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import torch.distributed as dist
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local)
    dev = torch.device("cuda", local) if cuda else torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    main_rank = rank == 0
    out = Path(a.out)
    if main_rank:
        out.mkdir(parents=True, exist_ok=True)
    prog = open(out / "progress.jsonl", "a", buffering=1) if main_rank else None

    def emit(d):
        if prog:
            prog.write(json.dumps(d) + "\n")

    random.seed(a.seed); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    examples, kind = build_examples(load_rows(a.data, a.split, a.rows, a.seed), tok, a.max_len)
    rng = random.Random(a.seed)
    rng.shuffle(examples)
    n_eval = max(1, int(len(examples) * a.eval_frac)) if len(examples) > 1 else 0
    eval_set, train_set = examples[:n_eval], examples[n_eval:]
    if not train_set:
        sys.exit("Too little data: nothing left to train on after holding out the evaluation share")

    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    if a.mode == "pretrain":
        cfg = AutoConfig.from_pretrained(a.model)
        if a.layers:
            for key in ("num_hidden_layers", "n_layer", "num_layers"):
                if hasattr(cfg, key):
                    setattr(cfg, key, a.layers)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32 if world > 1 or not cuda else dtype)
    if not a.no_grad_ckpt:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    hf_model = model
    params = sum(p.numel() for p in model.parameters())

    use_fsdp = world > 1 and cuda
    use_ddp = world > 1 and not cuda
    if use_fsdp:
        import functools
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        layer_names = set(getattr(model, "_no_split_modules", None) or [])
        layer_cls = {type(m) for m in model.modules() if type(m).__name__ in layer_names}
        policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls) if layer_cls else None
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16) if bf16 else None
        model = FSDP(model, auto_wrap_policy=policy, mixed_precision=mp, sharding_strategy=ShardingStrategy.FULL_SHARD,
                     device_id=dev, use_orig_params=True)
    elif use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model.to(dev))
    else:
        model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    warm = max(1, int(a.steps * a.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))

    def collate(batch):
        L = max(len(x[0]) for x in batch)
        ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in batch], device=dev)
        lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in batch], device=dev)
        att = (torch.arange(L, device=dev)[None, :] < torch.tensor([len(x[0]) for x in batch], device=dev)[:, None]).long()
        return ids, lab, att

    def reduce_sum(*vals):
        t = torch.tensor(vals, dtype=torch.float64, device=dev)
        if world > 1:
            dist.all_reduce(t)
        return t.tolist()

    @torch.no_grad()
    def evaluate():
        """Token-weighted mean loss over (up to eval_batches) held-out batches, split across ranks."""
        if not eval_set:
            return None
        model.eval()
        total = count = 0.0
        mine = eval_set[rank::world][: a.eval_batches * a.batch]
        for i in range(0, len(mine), a.batch):
            ids, lab, att = collate(mine[i:i + a.batch])
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
                logits = model(input_ids=ids, attention_mask=att).logits.float()
            shift_logits, shift_lab = logits[:, :-1], lab[:, 1:]
            loss = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_lab.reshape(-1),
                                                     ignore_index=-100, reduction="sum")
            total += loss.item(); count += (shift_lab != -100).sum().item()
        total, count = reduce_sum(total, count)
        model.train()
        return total / count if count else None

    emit({"event": "start", "mode": a.mode, "data_kind": kind, "params": params, "world_size": world, "bf16": bf16,
          "weights_dtype": "float32 (FSDP, bf16 compute)" if use_fsdp else str(next(hf_model.parameters()).dtype).replace("torch.", ""),
          "parallelism": "fsdp" if use_fsdp else "ddp" if use_ddp else "single",
          "fsdp": use_fsdp, "train_examples": len(train_set), "eval_examples": len(eval_set),
          "sequences_per_step": a.batch * a.accum * world, "max_len": a.max_len, "torch": torch.__version__,
          "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []})
    ev = evaluate()
    if ev is not None:
        emit({"event": "eval", "step": 0, "eval_loss": round(ev, 5), "eval_ppl": round(math.exp(min(ev, 50)), 4)})

    model.train()
    data_rng = random.Random(a.seed * 1000 + rank)
    t0, tokens_seen = time.time(), 0
    for step in range(a.steps):
        loss_sum = tok_sum = 0.0
        for _ in range(a.accum):
            ids, lab, att = collate(data_rng.sample(train_set, min(a.batch, len(train_set))))
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / a.accum).backward()
            n = (lab[:, 1:] != -100).sum().item()
            loss_sum += loss.item() * n; tok_sum += n
        if use_fsdp:
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        loss_sum, tok_sum, att_tokens = reduce_sum(loss_sum, tok_sum, float(att.sum().item()) * a.accum)
        tokens_seen += att_tokens
        emit({"step": step + 1, "loss": round(loss_sum / max(1, tok_sum), 5), "lr": sched.get_last_lr()[0],
              "t": round(time.time() - t0, 2), "tokens": int(tokens_seen)})
        if a.eval_every and (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
            ev = evaluate()
            if ev is not None:
                emit({"event": "eval", "step": step + 1, "eval_loss": round(ev, 5), "eval_ppl": round(math.exp(min(ev, 50)), 4)})
    seconds = time.time() - t0
    ev = evaluate()
    if ev is not None:
        emit({"event": "eval", "step": a.steps, "eval_loss": round(ev, 5), "eval_ppl": round(math.exp(min(ev, 50)), 4)})

    # Save a regular Hugging Face checkpoint (FSDP: gather the full weights on rank 0, on the CPU).
    if use_fsdp:
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = model.state_dict()
    else:
        state = None
    if main_rank:
        save_dtype = torch.bfloat16 if bf16 else torch.float16
        if state is not None:
            state = {k: v.to(save_dtype) for k, v in state.items()}
            hf_model.save_pretrained(out / "model", state_dict=state, safe_serialization=True)
        else:
            hf_model.to(save_dtype).save_pretrained(out / "model", safe_serialization=True)
        tok.save_pretrained(out / "model")
        emit({"event": "end", "seconds": round(seconds, 1), "tokens": int(tokens_seen),
              "tokens_per_second": round(tokens_seen / seconds, 1) if seconds > 0 else None})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
