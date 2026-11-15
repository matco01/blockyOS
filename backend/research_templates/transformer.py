"""Conventional local Transformers checkpoint scored with the shared byte-normalized contract.

Requires torch and transformers in the selected environment; checkpoint must already exist locally.
"""
import math
from pathlib import Path
import statistics
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from blocky_research import Run, fingerprint

run = Run()
cfg = run.config
checkpoint = cfg.get("checkpoint")
if not checkpoint or not Path(checkpoint).is_dir():
    raise ValueError("Set config.checkpoint to an existing local Transformers checkpoint directory")
import json
manifest = run.directory / "checkpoint-hashes.json"
manifest.write_text(json.dumps({str(p.relative_to(checkpoint)): fingerprint(p) for p in Path(checkpoint).rglob("*") if p.is_file()}, indent=2))
run.artifact(manifest, "Checkpoint files hashed before evaluation")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
if start_id is None:
    raise ValueError("This scorer requires a model-native beginning-of-document token")
evaluation = Path(cfg.get("eval", "eval.txt"))
docs = evaluation.read_bytes().splitlines(keepends=True)
if not docs or not sum(map(len, docs)):
    raise ValueError("Evaluation data is empty")
reps = int(cfg.get("reps", 3))
if not 1 <= reps <= 100:
    raise ValueError("Choose 1–100 repetitions")
rates = []
for repeat in range(reps):
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    losses = []
    with torch.inference_mode():
        for doc in docs:
            text = doc.decode("utf-8", errors="strict")
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > int(cfg.get("max_document_tokens", 2048)):
                raise ValueError("A document exceeds max_document_tokens; prepare shorter documents rather than silently truncate")
            tokens = torch.tensor([[start_id] + ids], device=device)
            logits = model(input_ids=tokens).logits[:, :-1].float()
            nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction="sum")
            losses.append(float(nll) / math.log(2))
    if device == "cuda":
        torch.cuda.synchronize()
    rates.append(sum(map(len, docs)) / (time.perf_counter() - started))
    run.progress(f"Scored repetition {repeat + 1}")
run.metric("bits_per_byte", sum(losses) / sum(map(len, docs)), unit="bit/byte", protocol="text-byte-nll/1",
           task="text-likelihood", dataset_sha256=fingerprint(evaluation), better="lower",
           scope="all UTF-8 bytes; newline-delimited documents; model-native start state",
           samples=[b / len(d) for b, d in zip(losses, docs)], note="Likelihood of the tokenizer's canonical encoding, byte-normalized; not a sum over alternate tokenizations. Different pretraining budgets are not controlled.")
run.metric("evaluation_bytes_per_second", statistics.median(rates), unit="byte/s", protocol="text-score-walltime/1",
           task="text-likelihood", dataset_sha256=fingerprint(evaluation), better="higher", samples=rates,
           scope="model loaded; complete evaluation including input encoding; no warm-up")
run.progress(f"Checkpoint: {checkpoint}; device: {device}. Record checkpoint hashes as artifacts for reproducibility.")
