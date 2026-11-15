"""Small byte-bigram baseline. Included text is a smoke test, not a research benchmark."""
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
import time

from blocky_research import Run, fingerprint

run = Run()
cfg = run.config
train = Path(cfg.get("train", "train.txt"))
evaluation = Path(cfg.get("eval", "eval.txt"))
counts = defaultdict(Counter)
for line in train.read_bytes().splitlines(keepends=True):
    previous = 256
    for byte in line:
        counts[previous][byte] += 1
        previous = byte
docs = evaluation.read_bytes().splitlines(keepends=True)
for doc in docs:
    doc.decode("utf-8", errors="strict")
if not docs or not sum(map(len, docs)):
    raise ValueError("Evaluation data is empty")
totals = {key: sum(values.values()) for key, values in counts.items()}
alpha = float(cfg.get("smoothing", 1))
if alpha <= 0:
    raise ValueError("Smoothing must be positive")
reps = int(cfg.get("reps", 3))
if not 1 <= reps <= 100:
    raise ValueError("Choose 1–100 repetitions")
rates = []
for repeat in range(reps):
    started = time.perf_counter()
    losses = []
    for doc in docs:
        previous, bits = 256, 0
        for byte in doc:
            bits -= math.log2((counts[previous][byte] + alpha) / (totals.get(previous, 0) + 256 * alpha))
            previous = byte
        losses.append(bits)
    elapsed = time.perf_counter() - started
    rates.append(sum(map(len, docs)) / elapsed)
    run.progress(f"Scored repetition {repeat + 1}")
contract = dict(task="text-likelihood", dataset_sha256=fingerprint(evaluation),
                scope="all UTF-8 bytes; newline-delimited documents; model-native start state")
run.metric("bits_per_byte", sum(losses) / sum(map(len, docs)), unit="bit/byte", protocol="text-byte-nll/1",
           better="lower", samples=[b / len(d) for b, d in zip(losses, docs)],
           note="Document-level samples; correlated text is not independent evidence. Byte-bigram baseline, not BLT.", **contract)
run.metric("evaluation_bytes_per_second", statistics.median(rates), unit="byte/s", protocol="text-score-walltime/1",
           scope="model loaded; complete evaluation including input encoding; no warm-up", task="text-likelihood",
           dataset_sha256=fingerprint(evaluation), samples=rates, better="higher")
model = run.directory / "byte-model.json"
import json
model.write_text(json.dumps({str(k): dict(v) for k, v in counts.items()}))
run.artifact(model, "Trained byte-bigram counts")
run.progress(f"Training data SHA-256: {fingerprint(train)}")
