# Evidence and repeated validation

In Experiments → Report, every result has an evidence button:

- **Measured:** evaluator or sensor output from the recorded hardware.
- **Estimated:** calculated from measurements and assumptions, including hardware rooflines and costs.
- **Simulated:** measured on this hardware with target-device constraints. This is not a measurement on another device.
- **Unavailable:** no valid value was recorded.

An estimated result can also use a simulated setup; its evidence explanation identifies both. Open the button for
the measurement source, protocol, supporting data, commands, and retained run log. New runs retain speed/fidelity
evaluator output locally. Earlier runs and imported results may have summaries without the original output files.
The display explicitly reports absent evidence. Use “Show metrics unavailable in every run” to inspect unsupported
or unrecorded metrics as well.

## Repeat and validate

Choose completed local runs for reference A and candidate B, one metric, a minimum meaningful percentage improvement,
and 3–12 pairs. The action queues full new profiles in counterbalanced order: A B, B A, A B, … (method
`paired-median-sign/2`). Alternating which setup runs first cancels a steady drift such as warm-up or rising temperature.
Batches made before this change used a fixed A B order (`paired-median-sign/1`) and say so in their report. Six pairs is the minimum that can produce
a finite distribution-free two-sided 95% interval. The default six pairs queues twelve full profiles; runtime depends
on the model, hardware and enabled tests.

The batch saves both configurations, the measurement protocol, test settings, and all planned run IDs. Its queued
runs keep the saved test settings even if the experiment's optional tests are subsequently edited. A backend protocol
change invalidates queued validation runs instead of silently changing their measurement semantics. Queued batches
survive restarts; interrupted runs remain errors and prevent a positive conclusion.

The report shows each pair, paired percentage improvements, between-pair SD, median improvement, an exact sign-test
p-value, and a distribution-free interval for the median improvement. For lower-is-better energy/token, a reduction
is a positive improvement. The interval uses binomial order statistics; with too few pairs it is unbounded.

A setup is favored for the selected metric only when every planned pair has valid measurements, required provenance
is present, conditions match, the two-sided sign test has p < 0.05, and the entire interval exceeds the preselected
minimum improvement. Missing/cancelled/deleted/failed runs, changed weights or engines within an arm, changed hardware,
test inputs or power sources prevent a positive conclusion. New batch measurements do not reuse historical source
scores as repetitions.

The analysis assumes independent paired measurements under stable conditions. Counterbalancing cancels a linear drift
but not every order effect (for example a cache warmed by whichever setup ran first). Each batch tests one preselected metric; trying many batches or metrics and selecting a favorable one
does not preserve a study-wide 5% false-positive rate. The result does not establish quality preservation or overall
model superiority. Inspect quality separately; an inconclusive result does not prove equivalence.

The implementation adds storage and analysis, without changing how an individual profile is measured, so it uses the
existing profile protocol. Validation analysis itself records `paired-median-sign/1` in each saved plan.

## Accuracy differences

MMLU-Pro comparisons report the paired McNemar p-value and the accuracy difference with a 95% interval for paired
proportions (Newcombe's hybrid score method), in percentage points. The interval says how large the change could
plausibly be; a non-significant result with a wide interval is not evidence that two setups are equivalent. A backend
test checks the interval's coverage by simulation (about 95% at 560 questions).

## Perplexity and KL divergence sample size

By default every run is scored on 32 chunks of 512 tokens of WikiText-2 (test split). Add tests → "Perplexity / KL
divergence sample" raises this to 64, 128 or 256 chunks, which narrows the error bars and makes the 99th-percentile KLD
more stable. Set it before the baseline: the baseline's logits are the reference, so every run of an experiment must use
the same text. A run whose experiment changed the setting afterwards stops with an explanation, and runs scored on
different samples are marked as not comparable.

## Instrument checks

Experiments → Repeat and validate → Instrument checks tests properties that hold for any correct measurement
pipeline, using runs the experiment already has:

| Check | Pass when | Runs it needs |
|---|---|---|
| Baseline vs itself | KL divergence of a rerun < 0.001 and same top token ≥ 99.5% | “Repeat this setup” on the baseline, ideally twice at different times |
| Same answers | Greedy MMLU-Pro answers of reruns agree ≥ 99% and the paired test finds no change | The same reruns |
| Speed repeatability | Decode speed of reruns within 3% of the baseline (warning up to 8%) | The same reruns |
| Quantization order | Across ≥ 3 weight variants of the baseline model, fewer bits per weight never give a clearly lower KLD | Branches that change only the weights file (e.g. Q8_0, Q4_K_M, Q2_K) |
| Bandwidth limit | Estimated decode traffic ≤ 105% of the nominal memory bandwidth | Any run on a GPU with a known bandwidth |
| Cited values | Each recorded reference value is matched within its tolerance | A model file whose SHA-256 has a reference entry |

A failed check means: stop and investigate before trusting comparisons in this experiment. Passing checks is evidence
that the pipeline works on this machine, not proof that every number is right. The same report is available to AI
assistants through the MCP tool `get_instrument_check`.

### Recording reference values

Blocky ships no reference numbers: values only count when you took them from a source you can cite and the model file
is byte-identical. Record them in Tests → Instrument checks → **Record a published value** (it fills in the file's
SHA-256 from the run), with `POST /api/reference-values`, or directly in `/opt/blocky/data/reference-values.json`:

```json
[
  {"model_sha256": "<SHA-256 of the GGUF, from the run's provenance>", "metric": "ppl", "value": 6.23, "tolerance": 0.05,
   "source": "Author, title, table/figure, URL"}
]
```

Metrics: `ppl`, `kld`, `same_top`, `mmlu_acc` (fraction), `tg`, `pp`. Choose the tolerance from the source's reported
uncertainty and any differences in setup (context length, chunk count, dataset version); speed references only make sense
for the same hardware.

### A validation study on new hardware

1. Create an experiment with a high-precision GGUF (F16/BF16 or Q8_0) as the baseline. Optionally raise the perplexity
   sample to 128 chunks first.
2. Queue “Repeat this setup” twice, an hour apart.
3. Branch three or more weight variants (e.g. Q6_K, Q4_K_M, Q3_K_M, Q2_K) from the baseline, changing nothing else.
4. Open Instrument checks. Export the experiment (JSON) together with the report for your records.

## Studies on real hardware

### Current: protocol v4, Qwen3.5-0.8B and Qwen3.5-9B, RTX 2080 Super (2026-11-27)

Data: 0.8B [CSV](validation/2026-11-27-rtx2080super-qwen3.5-0.8b-v4.csv) ·
[instrument checks](validation/2026-11-27-rtx2080super-qwen3.5-0.8b-v4-instrument-checks.json); 9B
[CSV](validation/2026-11-27-rtx2080super-qwen3.5-9b-v4.csv) ·
[instrument checks](validation/2026-11-27-rtx2080super-qwen3.5-9b-v4-instrument-checks.json). Experiments 13 and 14.

**Setup.** RTX 2080 Super 8 GB (driver 610.88), the GPU that also drives the desktop, under WSL2 on a Ryzen 5 5600G;
llama.cpp 0.9.0-dev (`956591f`); protocol `blocky-profile/4`; blockyOS `114347f` with no local changes in every run.
GPU memory margin 2048 MiB (display reserve on); the tools reused the server's GPU layer count in every completed run.
Perplexity/KLD on 128 chunks of 512 tokens; 60 s cool-down before each run; GPU clocks not locked.

- **0.8B:** baseline Qwen3.5-0.8B F16 (SHA-256 `838810d3…`). Q6_K, Q4_K_M, Q3_K_M, Q2_K built from that F16 with the
  weights workshop (no importance matrix); Q8_0 is a downloaded file. Every layer on the GPU in every run.
- **9B:** baseline Qwen3.5-9B Q8_0 (SHA-256 `80962657…`, downloaded). Q6_K and Q2_K requantized from that Q8_0 (so they
  lose slightly more than files made from full precision); Q4_K_M and Q3_K_M are downloaded files. The model does not fit
  in the GPU memory left free: the baseline ran 18 of 33 layers on the GPU, Q6_K 24, Q4_K_M 32.

**Instrument checks: all 5 that can run pass in both studies, 0 warnings, 0 failures.**

| Check | 0.8B | 9B |
|---|---|---|
| Baseline vs itself | pass: KLD −0.000016, same top token 100% (2 reruns) | pass: KLD ≈ 0, same top token 99.997% (1 rerun) |
| Same answers | pass: 560/560 identical | pass: 560/560 identical |
| Speed repeatability | pass: 1.6% | pass: 1.2% |
| Quantization order | pass: Spearman ρ = −1.00 (5 files) | pass: Spearman ρ = −1.00 (3 files) |
| Bandwidth limit | pass: at most 32% | pass: at most 67% |
| Cited values | not run: no usable published reference | not run |

**0.8B results.** Decode = llama-bench at depth 0, median of 5 repetitions ± 1.4826 × MAD. Perplexity ± its standard
error. MMLU-Pro differences paired against the baseline (McNemar, Holm-corrected within the experiment).

| Weights | Decode tok/s | Perplexity | Mean KLD | KLD p99 | Same top token | MMLU-Pro | Δ vs F16 (p, Holm) | Needle |
|---|---|---|---|---|---|---|---|---|
| F16 (baseline / repeats) | 151.99 ± 1.26 / 149.60 ± 2.51 / 150.60 ± 1.50 | 18.393 ± 0.314 | 0 | 0 | 100% | 21.61% | — | 10/10 |
| Q8_0 | 223.99 ± 7.47 | 18.425 ± 0.315 | 0.00148 | 0.0071 | 97.86% | 21.79% | +0.18 pts (1.0) | 10/10 |
| Q6_K | 241.06 ± 11.54 | 18.540 ± 0.317 | 0.00624 | 0.031 | 95.52% | 21.61% | 0.00 pts (1.0) | 10/10 |
| Q4_K_M | 256.50 ± 7.34 | 18.919 ± 0.322 | 0.0604 | 0.318 | 87.32% | 18.75% | **−2.86 pts (0.035)** | 10/10 |
| Q2_K | 265.04 ± 10.02 | 105.93 ± 2.15 | 1.71 | 6.85 | 44.00% | 11.96% | **−9.64 pts (< 0.001)** | 0/10 |

Q3_K_M has no result (see incidents). Only Q4_K_M and Q2_K differ significantly from F16 on MMLU-Pro.

**9B results.** Same columns. Speed differences here mostly reflect how many layers fit on the GPU with the margin
(18, 24, 32 of 33), not the formats themselves: on a GPU with more free memory the order and the gaps would differ.

| Weights | GPU layers | Decode tok/s | Perplexity | Mean KLD | KLD p99 | Same top token | MMLU-Pro | Δ vs Q8_0 (p, Holm) | Needle |
|---|---|---|---|---|---|---|---|---|---|
| Q8_0 (baseline / repeat) | 18/33 | 7.49 ± 0.50 / 7.40 ± 0.42 | 7.783 ± 0.110 | 0 / ≈0 | 0 / 0.00004 | 100% / 99.997% | 59.64% | — | 10/10 |
| Q6_K | 24/33 | 15.42 ± 0.31 | 7.834 ± 0.111 | 0.00612 | 0.034 | 96.81% | 60.89% | +1.25 pts (0.69) | 10/10 |
| Q4_K_M | 32/33 | 50.32 ± 1.33 | 7.887 ± 0.112 | 0.0162 | 0.125 | 94.46% | 58.93% | −0.71 pts (1.0) | 10/10 |

No MMLU-Pro difference is significant for the 9B. Q3_K_M, Q2_K and the second repeat have no result (see incidents),
so the 9B quantization-order check rests on three files and the repeatability checks on one rerun.

**Repeatability across sessions.** The 0.8B quality numbers are identical to the protocol-v3 study two days earlier on
the same files: perplexity (e.g. 18.393088 for F16), KLD, same-top-token rate and MMLU-Pro accuracy for every variant
completed in both. Speed is not
compared across the two studies: the protocol changed how runs are placed on the GPU.

**Incidents.**
- **The GPU hung twice under sustained 9B load** (2026-11-25 and 2026-11-27). Both times the GPU temperature had risen
  to 85–88 °C over about 45 minutes of back-to-back runs. Then the GPU stopped responding: WSL lost it (`dxg` ioctl
  errors, NVML unavailable) and the computer had to be reset. Windows logged no driver recovery. During the 9B Q4_K_M
  run the GPU reached 88 °C and only cooled to 75 °C during the idle phase. The 0.8B runs stayed below 64 °C. This
  points to this card's cooling rather than to the measurements, but it means the 9B study is incomplete. Runs 71
  (Q3_K_M) and 72 (Q2_K) failed; run 72 also started after the GPU was gone, and its provenance records no display
  (margin 1024), because the display check could not reach the GPU.
- **0.8B Q3_K_M (run 64)** failed 45 s into the serving test: llama-server shut down cleanly ("cleaning up before exit")
  with no out-of-memory error and the GPU at a normal temperature. The cause has not been found. A retry was queued and
  then cancelled with the rest of the queue after the second hang.

**Still open:** published reference values (cited-values check); the missing 9B runs; a second device (cross-device
reproduction with the bundles); the cause of run 64.

### Earlier: protocol v3, Qwen3.5-0.8B, RTX 2080 Super (2026-11-25), superseded

Kept for the record. It found the two defects fixed since (negative llama-bench durations; separate GPU layouts for speed
and analysis). Use the protocol-v4 study above.

Data: [CSV of every run](validation/2026-11-25-rtx2080super-qwen3.5-0.8b.csv) ·
[instrument-check report](validation/2026-11-25-rtx2080super-qwen3.5-0.8b-instrument-checks.json).

**Setup.** RTX 2080 Super 8 GB (driver 610.88) under WSL2 on a Ryzen 5 5600G; llama.cpp 0.9.0-dev; protocol
`blocky-profile/3`. Baseline: Qwen3.5-0.8B F16 (SHA-256 `838810d3…`). Variants built from that F16 with the weights
workshop (no importance matrix): Q6_K, Q4_K_M, Q3_K_M, Q2_K; Q8_0 is a downloaded file. Perplexity/KLD on 128 chunks of
512 tokens; 60 s cool-down before each run; GPU clocks not locked. Queue order: baseline, repeat, Q8_0 … Q2_K, repeat
(22 minutes in total).

**Instrument checks: 4 pass, 1 warning, 0 fail, 1 not run.**

| Check | Result |
|---|---|
| Baseline vs itself | pass: largest KLD of 2 reruns −0.000016, same top token 100% |
| Same answers | pass: 560/560 MMLU-Pro answers identical in both reruns |
| Speed repeatability | **warning**: 7.7% (see below) |
| Quantization order | pass: Spearman ρ(bits per weight, KLD) = −1.00 over 6 files |
| Bandwidth limit | pass: at most 33% of the nominal 496 GB/s |
| Cited values | not run: no usable published reference (see below) |

**Results.** Speed = llama-bench decode at depth 0, median of 5 repetitions ± 1.4826 × MAD. Perplexity ± its standard
error over chunks. MMLU-Pro differences are paired against the baseline (McNemar, Holm-corrected over the experiment).

| Weights | Decode tok/s | Perplexity | Mean KLD | KLD p99 | Same top token | MMLU-Pro | Δ vs F16 (p, Holm) | Needle |
|---|---|---|---|---|---|---|---|---|
| F16 (repeat 1 / 2) | 146.05 ± 2.09 / 145.88 ± 1.28 | 18.393 ± 0.314 | 0 | 0 | 100% | 21.61% | — | 10/10 |
| Q8_0 | 223.94 ± 11.22 | 18.425 ± 0.315 | 0.00148 | 0.0071 | 97.86% | 21.79% | +0.18 pts (1.0) | 10/10 |
| Q6_K | 233.12 ± 3.86 | 18.540 ± 0.317 | 0.00624 | 0.031 | 95.52% | 21.61% | 0.00 pts (1.0) | 10/10 |
| Q4_K_M | 242.75 ± 10.94 | 18.919 ± 0.322 | 0.0604 | 0.318 | 87.32% | 18.75% | **−2.86 pts (0.042)** | 10/10 |
| Q3_K_M | 250.86 ± 4.92 | 22.172 ± 0.387 | 0.208 | 1.19 | 77.27% | 20.18% | −1.43 pts (1.0) | 10/10 |
| Q2_K | 280.80 ± 3.46 | 105.93 ± 2.15 | 1.71 | 6.85 | 44.00% | 11.96% | **−9.64 pts (< 0.001)** | 0/10 |

Only Q4_K_M and Q2_K differ significantly from F16 on MMLU-Pro. Q3_K_M scoring higher than Q4_K_M is within noise (its
difference from F16 is not significant), so it shows nothing about the two formats. These are results for one 0.8B
model; they do not generalize to other models.

**Speed repeatability.** The two reruns agree within 0.1% (146.05 vs 145.88 tok/s). The warning comes from the first
run of the session, which started on a cold GPU (33 °C) and decoded at about 159 tok/s. Single-run speed differences
under about 8% are therefore not trustworthy on this setup; use counterbalanced repeated validation, and give the
baseline the same warm-up as the runs it is compared with.

**Defect found and fixed.** In the baseline's decode test one repetition had a negative duration (the clock stepped
backwards under WSL). llama-bench printed it as 18,446,744,072,840,475,146 ns and 6.9 × 10⁻⁹ tok/s, which the parser
accepted as a valid positive rate, so the stored median is 158.13 instead of 159.28 tok/s. Such samples are dropped
since blockyOS `7e8df8c` (see the protocol, section 7.1). The stored result of this run was not rewritten.

**Provenance caveat.** blockyOS code changed during the study: runs 44–47 record commit `7fc585f` with uncommitted
changes, runs 48–51 record `abf8c35`. The difference was the installer and the changelog only, no measurement code,
and the repeats before and after it are identical (KLD 0, same answers). Studies should not change the checkout while
they run.

**Cited values: no usable reference found.** The protocol only accepts a published value for a byte-identical model
file measured on the same text, context and chunk count. Candidates checked: the llama.cpp perplexity scoreboard
(LLaMA 3 8B; reference files not distributed) and eaddario/Qwen3.5-9B-GGUF (per-file PPL/KLD with SHA-256s, but
custom `--target-bpw` tensor mixes whose logs omit context size and chunk count, and a 17.9 GB F16 reference). This
check stays open.

## What the tests cover

Backend tests cover HTTP queueing, order, frozen settings, cancellation, incomplete or noisy samples, changed conditions,
and raw evidence retrieval. Frontend tests cover evidence labels and missing samples. Browser verification used an
isolated database with a disabled inference worker to exercise evidence, queueing and cancellation. It did not launch
a full GPU validation batch.
