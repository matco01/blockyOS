# Measurement protocol

This document specifies how blockyOS measures every number it reports, for the protocol version

```
blocky-profile/3 (MMLU-Pro option logprobs · llama-bench p512 n128 d0,4096 r5 · PPL/KLD WikiText-2 c512 x32 · serving/energy: actual usage, full request interval)
```

The string is the `PROTOCOL` constant in `backend/app/profile.py` and is recorded in every run's provenance. Results
produced under different protocol versions are never compared: the UI marks them "not comparable", and a baseline from
an older protocol cannot anchor new runs. Any change to how a metric is measured changes this string
([CONTRIBUTING.md](../CONTRIBUTING.md)).

Section numbers below are referenced from issues and papers; keep them stable.

## 1. Unit of measurement: a run

An **experiment** fixes one model family and a set of tests. A **run** profiles one complete **setup**: weights file,
inference engine (stock llama.cpp, a built llama.cpp commit, vLLM, SGLang or TensorRT-LLM), context size, KV-cache type,
GPU layers, flash attention, threads, device limits, GPU count and split, and other machines (RPC workers, pipeline
stages). The setup is normalized (defaults filled in, values validated) and stored with the run; settings left at their
defaults are not stored, so a setup's identity doesn't change when new settings are added.

The **baseline** is the experiment's first completed run. Every other run is compared with the baseline and with its
**parent** (the run it was branched from). "Change one thing at a time" is the intended use; the run's label names
every setting that differs from its parent.

Runs execute one at a time, holding the lab lock. The device's chat model is stopped for the duration, and other
measurements (benchmarks, quality tests, training) wait.

## 2. Phases of a run

In this order, against a dedicated server started for the run on a loopback port:

| # | Phase | Section |
|---|---|---|
| 0 | Provenance, weight summary, hardware snapshot; optional cool-down and GPU clock lock | 3, 11 |
| 1 | Load: start the server with the exact setup, wait for `/health`, record load time and memory layout | 4 |
| 2 | Quality: MMLU-Pro | 5 |
| 2b | Long context: needle in a haystack; optional RULER-style suite | 6 |
| 2c | Serving speed (engine-neutral) | 7.2 |
| 2d | Energy: idle, then sustained generation | 8 |
| 2e | Optional lm-evaluation-harness tasks | 5.4 |
| 3 | Speed: llama-bench (llama.cpp engines) | 7.1 |
| 3b | Optional GPU kernel profile (Nsight Systems) | 9.3 |
| 4 | Fidelity: perplexity and KL divergence against the baseline | 9 |
| 5 | Telemetry summary and hardware analysis | 10 |

vLLM, SGLang and TensorRT-LLM runs end after phase 2e: llama-bench, kernel profiles and llama-perplexity are llama.cpp
tools. Their speed comes from phase 2c only, and they have no perplexity or KL divergence.

## 3. Provenance

Recorded per run (`results.provenance`), and compared between any two runs before statistics are reported:

- protocol string; blockyOS commit; engine identity (llama.cpp version and commit, or framework version)
- model file name, SHA-256 (GGUF) or SHA-256 over the safetensors files (checkpoints), size, and the weight recipe if
  Blocky built the file (quantization type, importance matrix, overrides)
- SHA-256 of the MMLU-Pro dataset file, the MMLU seed, the SHA-256 of the WikiText-2 test text and of the
  perplexity/KLD text actually used
- GPU name, driver, VRAM, PCIe link; CUDA runtime and toolkit versions; CPU, RAM, OS, kernel, whether it ran in WSL
- test settings: MMLU-Pro question count, perplexity chunk count, needle and RULER settings, lm-eval tasks and limit
- noise controls: cool-down, requested and applied GPU clock lock (or the error if locking failed), GPU temperature at
  the start
- other machines taking part (RPC workers, pipeline stages)
- the exact command line of every phase (`results.commands`)

Raw evaluator output (llama-bench JSON lines, llama-perplexity output, the run log) is retained on the device and can
be inspected per metric ("evidence") and exported in the reproducibility bundle.

**Comparability rules.** Statistics between two runs are reported only when protocol, dataset hashes, MMLU seed and
question count, perplexity chunk count and text, GPU clock lock, llama.cpp commit (unless the engine is the variable
being changed), GPU driver and CUDA versions match. Otherwise the comparison lists the mismatched conditions and shows no
significance. A changed model file under the same name (different SHA-256) is reported explicitly. Different hardware is
reported as information, not as a mismatch: comparing devices is a valid use.

## 4. Load and memory

The server is started with the setup's flags and polled at `/health` (300 s timeout; 1500 s for checkpoint frameworks,
which compile kernels on first load). If the process exits first, the run fails with the last lines of the server log.
For llama.cpp, the load log is parsed for every `<device> <kind> buffer size = <MiB>` line (model weights, KV cache,
compute buffers per device) and the number of offloaded layers. VRAM in use is read from the telemetry 1.5 s after load.

**RAM limit simulation:** the model's host-side buffers plus 700 MB for a minimal operating system must fit the limit,
otherwise the run fails ("doesn't fit"). **VRAM limit simulation** uses llama.cpp's `--fit-target` to leave the
difference to the real VRAM unused.

## 5. Quality

### 5.1 MMLU-Pro (default 560 questions)

- **Questions.** From the MMLU-Pro test set, grouped by its 14 subjects (sorted by name). With the fixed seed
  `20240611`, each subject's questions are shuffled once. One question each from the first five subjects becomes a
  few-shot example and is removed from that subject's pool. The test set takes the first 40 questions per subject
  (560), or 100 (1,400) or 250 (3,500) per subject when an experiment chooses a larger size. Larger sizes are
  nested: their first 560 questions are the default 560, so paired tests between sizes use the common questions.
- **Prompt.** "The following are multiple choice questions (with answers).", the five examples with their answers, then
  the question with options lettered A–J and "Answer:".
- **Scoring.** One completion token at temperature 0; the server's top next-token log-probabilities (llama.cpp:
  `n_probs = 50`; OpenAI-compatible frameworks: `logprobs = 20`). Tokens that are a single option letter (after
  stripping whitespace) are kept; the predicted answer is the letter with the highest log-probability. If no option
  letter is among the returned tokens, the run fails rather than guessing.
- **Reported.** Accuracy *k/n* with a 95% Wilson score interval (Wilson, 1927); per-subject counts; the per-question
  right/wrong string (for paired tests); the mean probability of the correct letter, renormalized over the option
  letters present in the shortlist. That last number is **not** the model's full probability of the answer (letters
  outside the shortlist have no measured probability), and the UI says so.

### 5.2 Paired comparison of two runs

Both runs answered the same questions, so the comparison is paired:

- **McNemar exact test** (McNemar, 1947): with *b* questions right only in the baseline and *c* right only in the
  other run, two-sided *p* = min(1, 2 · P(X ≤ min(b, c))) for X ~ Binomial(b + c, ½); *p* = 1 when b + c = 0.
- **Difference with a 95% interval**: the accuracy difference with Newcombe's hybrid score interval for paired
  proportions (Newcombe, 1998, method 10), which keeps nominal coverage when few answers flip. A simulation in the test
  suite checks coverage of 93–97.5% at 560 questions.
- **Several runs against one baseline**: Holm-adjusted *p*-values (Holm, 1979) are shown next to the raw ones.
- **Equivalence (TOST).** Two one-sided tests at α = 0.05 (Schuirmann, 1987), via the 90% Newcombe interval: two setups
  are reported equivalent within ±*m* accuracy points when that interval lies inside (−*m*, +*m*). The margin *m* must be
  chosen in advance (in a preregistration or the experiment's test settings). Without one, only the smallest margin the
  data would support is shown, explicitly labeled as not preselected. A non-significant McNemar test is never presented
  as evidence of equivalence.

### 5.3 Power

The planner turns the discordance rate measured in the experiment (share of questions on which two setups disagree;
12% assumed before there is data) into the questions needed to detect a chosen difference at the chosen power and a
Bonferroni-adjusted α for the number of planned comparisons, and into the smallest detectable difference at the
current size.

### 5.4 lm-evaluation-harness (optional)

`lm_eval run --model local-chat-completions` against the run's own server, chat template applied, greedy
(`temperature=0`), seed `0,0,0,0`, the experiment's task list and per-task limit. Only generative tasks are meaningful:
the servers don't return prompt log-probabilities. The harness's own metrics and standard errors are stored as reported.

## 6. Long context

### 6.1 Needle in a haystack

Haystacks are slices of the WikiText-2 test text sized in tokens with the model's own tokenizer (characters per token
measured on the first 40,000 characters via `/tokenize`). Lengths: 2,048, 4,096, 8,192, … up to the largest that fits
the run's context with 600 tokens to spare; depths 5%, 25%, 50%, 75%, 95% (inserted at the nearest sentence boundary).
The needle is "The secret code for the *city* vault is *7-digit code*." with city and code from a fixed seed (`1234`),
so every run sees identical prompts. The question asks for the number only; the answer counts as correct when the code
appears among the reply's digits. Greedy, thinking off, 24 tokens maximum. Reported per length × depth and overall.

### 6.2 RULER-style suite (optional)

Four synthetic tasks modeled on RULER (Hsieh et al., 2024): multi-key retrieval with distractors, multi-value retrieval,
variable tracking with a decoy chain, and common-word extraction. Lengths from 4,096 to the run's context; 4, 8 or 20
samples per task and length from seed `7777`. Scores are per-sample recall in [0, 1], averaged with a 95% interval; the
effective context length is the longest length whose mean score is at least 85.6%, RULER's threshold.

## 7. Speed

### 7.1 llama-bench (llama.cpp engines)

```
llama-bench -m <model> -p 512 -n 128 -d 0,4096 -r 5 -ctk <kv> -ctv <kv> -fa <fa> -o jsonl <placement flags>
```

Prefill (512-token prompt) and decode (128 tokens) at depth 0 and at 4,096 tokens of context, five repetitions each.
Per test, the reported value is the **median** of the per-repetition rates (`samples_ts`), with a robust spread
1.4826 × MAD. Samples that are non-finite or not positive are dropped and counted, and so are samples whose duration
(`samples_ns`) is not positive: a clock that steps backwards gives a negative duration, which llama-bench prints as a
huge unsigned number and a tiny positive rate. Positive outliers are kept, because
a summary cannot tell a timer error from a real slowdown. The five repetitions come from one process at one time, so
their spread is **descriptive**, not a standard error: it doesn't include session-to-session drift (section 12).

### 7.2 Serving speed (every engine)

The same client-side measurement for every engine, through the server's OpenAI-compatible `/v1/chat/completions`
with streaming: a prompt of about 512 tokens (a different WikiText-2 slice each time, so no prompt cache helps), asking
for exactly 128 output tokens (`ignore_eos`), temperature 0. One warm-up request, then five measured ones. Token
counts come from the server's reported usage only; a request without usage counts contributes no rate. Reported medians
and robust spreads of:

- output throughput: output tokens / full request time (**includes prefill**);
- prompt rate: prompt tokens / time to the first content chunk (a latency proxy, not isolated prefill);
- time to first token.

## 8. Energy

With the model loaded and idle for 8 s, then during a sustained generation of max(256, 15 s × measured serving
throughput) tokens (capped by the context and at 8,192), telemetry is sampled every 0.5 s. The first 2 s of idle are
discarded (boost clocks settling). Power sources, in order of preference: a wall-power meter configured in Settings
(the whole device), GPU and CPU package sensors together, the GPU sensor alone, the CPU package (RAPL) alone. For each
available source: mean and SD of idle and load power, **energy per output token** = mean load power / (output tokens /
full request time, including prefill), the same above idle power, and tokens per Wh. With a configured electricity
price, cost per million output tokens. Without any power source, the run says energy wasn't measured.

## 9. Fidelity: perplexity and KL divergence

### 9.1 Text and sample

`llama-perplexity -c 512 --chunks <N>` on the WikiText-2 test split (raw), or on a `.txt` file the experiment chose
(at least 50 KB; its SHA-256 is recorded). *N* is 32 by default, or 64, 128 or 256 chunks of 512 tokens.

### 9.2 Baseline and KL divergence

The baseline run saves its logits (`--kl-divergence-base`) together with the baseline's run id, the protocol, the chunk
count and the text's SHA-256. Every other run with the **same tokenizer** is scored against them
(`--kl-divergence`): mean KL divergence with llama.cpp's reported uncertainty, the 99th percentile of per-token KL
divergence, the share of tokens whose top prediction matches the baseline's, and perplexity. A run whose experiment
changed the chunk count or text after the baseline, or whose saved logits belong to another run or protocol, fails
with an explanation. Runs with a **different tokenizer** get their own perplexity only, marked not comparable: per-token
KL divergence is undefined across tokenizers.

**Known limitation.** llama-perplexity reports cumulative means and a token-level uncertainty; tokens within a chunk are
correlated, so that uncertainty understates the sampling error of the text. Blocky therefore shows KL divergence as a
measured value and its spread, but doesn't attach significance to differences in KL divergence between two runs. To
decide between close setups, raise the chunk count and use repeated validation (section 12) or compare against the
instrument's own floor (a baseline rerun, section 13).

### 9.3 GPU kernel profile (optional)

A short llama-bench prefill and decode under Nsight Systems; kernel time is grouped by operation class (matrix
multiplication, attention, normalization, …). Needs CUDA kernel tracing (regular Linux, not WSL).

## 10. Hardware analysis (estimates)

Per run, from stored data, against the hardware snapshot taken during that run. All results here are labeled
**estimated**:

- **Weight bytes per token**: weight bytes from the GGUF tensor table, excluding the embedding table (looked up, not
  read in full); for mixture-of-experts models, expert weights scaled by experts used / experts total. Checkpoints:
  safetensors size (16-bit).
- **Decode roofline**: time to read the GPU-resident weights at the GPU's spec-sheet memory bandwidth plus the
  CPU-resident weights at the measured RAM bandwidth; reported as a share of the measured time per token. With several
  GPUs in tensor parallelism (llama.cpp row split, vLLM/SGLang/TensorRT-LLM with more than one GPU) their bandwidths add;
  with layer (pipeline) split they don't, because one sequence uses one GPU at a time. When part of the model runs on
  other machines, no single-machine limit applies and none is reported.
- **Prefill compute**: 2 × parameters FLOPs per token over the spec-sheet peak, and against a roofline limited by
  bandwidth × arithmetic intensity.
- **Clocks and thermals**: average SM clock as a share of the maximum, peak temperature, utilization.
- **Bottleneck**: a stated *possible* limit with its evidence, never a measured time breakdown.
- **Device projection** (CPU-only presets): decode speed at the target device's spec-sheet RAM bandwidth, assuming the
  same share of bandwidth as achieved here.
- **Training utilization** (full training jobs): 6 × parameters FLOPs per trained token over the spec-sheet peak of the
  GPUs used; excludes attention and gradient-checkpointing recomputation.

## 11. Noise controls (optional, per experiment)

- **Cool-down**: wait 5–600 s before each run starts measuring.
- **GPU clock lock**: lock the GPU's graphics clock to a fixed frequency during the run (through a root-owned helper
  that accepts only `lock <MHz>` and `reset`), then reset it. Choose a clock the GPU can sustain under load (below its
  maximum boost) so thermal throttling doesn't vary between runs.

Both are recorded per run; runs with a different clock lock are not compared.

## 12. Repeated validation of a speed or energy difference

A batch re-measures two completed setups *A* and *B* in 3–12 pairs, in counterbalanced order (A B, B A, A B, …;
method `paired-median-sign/2`), for one preselected metric and a preselected minimum improvement. Per pair, the percent
change of B against A; reported: the median change, the exact two-sided sign test, and a distribution-free 95% interval
for the median from binomial order statistics (unbounded with fewer than six pairs). B (or A) is favored only when
every planned pair finished with valid measurements and matching conditions, *p* < 0.05, and the whole interval
exceeds the minimum improvement. Anything else is "insufficient evidence", never "equivalent". Counterbalancing
cancels a steady drift, not every order effect.

## 13. Instrument checks

Properties every correct measurement pipeline has, checked from an experiment's own runs (`backend/app/instrument.py`):

| Check | Pass | Warn | Fail |
|---|---|---|---|
| Baseline vs itself | rerun KLD < 0.001 and same top token ≥ 99.5% | KLD < 0.01 | otherwise |
| Same answers | rerun MMLU-Pro answers ≥ 99% identical, McNemar not significant | < 99% identical | paired test significant |
| Speed repeatability | rerun decode within 3% of the baseline | within 8% | otherwise |
| Quantization order | Spearman ρ(bits per weight, KLD) ≤ −0.8 over ≥ 3 variants | ρ > −0.8 | a clearly larger file (> 15% more bits) with clearly higher KLD (> 1.5×) |
| Bandwidth limit | estimated decode traffic ≤ 105% of the nominal bandwidth | above it | — |
| Cited values | each recorded reference value within its tolerance | — | outside it |

Reference values are the user's, with a source; Blocky ships none ([VALIDATION.md](VALIDATION.md)).

## 14. Preregistration

An experiment can record plans before running: hypothesis, primary metric, claim (difference, equivalence or
non-inferiority), direction, α, minimum effect, equivalence margin, analysis plan. Plans are append-only, stored with
their time and the SHA-256 of their content, and carry the protocol version. Runs queued after a plan are listed as
confirmatory for it; earlier runs are exploratory. A preregistered equivalence margin takes precedence over one set
later in the test settings.

## 15. Full training (Internals → Full training)

Not part of a profile run, but its outputs feed experiments. A fixed share of the examples (default 2%, chosen with the
seed before training) is held out; held-out loss (token-weighted) and perplexity are measured before the first step,
every *k* steps and at the end. Text data is packed into fixed-length blocks; chat data trains on answer tokens only.
Optimizer AdamW (β = 0.9, 0.95; weight decay 0.1), linear warm-up over 5% of steps, cosine decay to 10%, gradient
clipping at 1.0. Several GPUs use FSDP (full sharding per decoder layer, bf16 mixed precision where supported); without
GPUs, several processes use DDP. The resulting checkpoint carries a recipe with the exact command, environment and
evaluation results.

## References

- Hsieh, C.-P. et al. (2024). RULER: What's the real context size of your long-context language models? arXiv:2404.06654.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
- Newcombe, R. G. (1998). Improved confidence intervals for the difference between binomial proportions based on paired data. *Statistics in Medicine*, 17(22), 2635–2650.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
- Wang, Y. et al. (2024). MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. arXiv:2406.01574.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212.
- Merity, S. et al. (2016). Pointer sentinel mixture models (WikiText). arXiv:1609.07843.
- Williams, S., Waterman, A., Patterson, D. (2009). Roofline: an insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4), 65–76.
