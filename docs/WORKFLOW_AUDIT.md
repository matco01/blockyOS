# Workflow and measurement audit — 2026-11-15

The main application paths have real implementations: model-server responses, evaluator subprocess output, SQLite
records, and hardware sensors supply the results. This audit did not find a general mechanism that generates random
benchmark scores. It did find missing-data fallbacks, unsupported configuration labels, and stronger conclusions than
the measurements justified. Those are enough to mislead a user even when the underlying inference is real.

## What was exercised

- The original backend suite passed all 100 tests in an isolated WSL environment with the pinned development dependencies.
- Added regression tests for measurement integrity and HTTP workflows: experiment creation, branching, invalid sweep
  rejection, cancellation, export/import, settings, memory, the MCP catalog, and denial of unauthenticated remote requests.
  These tests use temporary databases and explicitly synthetic fixtures; they are not hardware performance evidence.
- A headless browser opened all 14 main pages against the running application. No JavaScript exceptions or failed API
  responses were observed during that smoke test. This checks initial rendering and reads, not every button or long job.
- The live MCP endpoint accepted initialization, tool listing, status, and experiment reads. Six experiments contained
  27 runs marked complete. Reading them confirms persistence and retrieval, not independent reproduction of their scores.
- A short request through the running `/v1/chat/completions` proxy returned an answer and actual token usage. It did not
  create a stored conversation or experiment.
- Final checks: **122 backend tests passed**, **17 frontend tests passed**, and the TypeScript/production build passed.
  The build still reports a large JS bundle. The rebuilt experiment, guide, settings, and internals pages also loaded
  without browser errors.

## Corrections

| Problem | Resulting behavior |
|---|---|
| Streaming chunks substituted for token counts | Serving/energy require server-reported usage; missing instrumentation leaves rates unavailable. |
| First/last chunk treated as a per-token decode interval | Protocol 3 reports output throughput over the whole request, including prefill; labels and energy intervals match. This is distinct from llama-bench decode speed. |
| Positive repetitions discarded solely for being far from the median | Positive finite observations are retained; median and MAD remain descriptive. Invalid samples cannot reappear through an aggregate fallback. |
| Descriptive SD/MAD treated as standard error, or percentage thresholds called significant | No statistical claim is made without a supported test. MMLU uses paired McNemar; explicit lm-eval standard errors permit an approximate normal test. Neutral does not mean equivalent. |
| Different protocols still produced comparisons | Incompatible comparisons are suppressed; new runs against an older protocol require a new experiment/baseline. |
| No applicable retrieval trials became zero accuracy | Needle and RULER results remain unavailable when no length fits. |
| Missing MMLU instrumentation became an incorrect answer | No returned option-letter scores causes a measurement error. The shortlist-based method is described as such, not as a full official benchmark reproduction. |
| Missing layer KLD became zero drift | Layer sensitivity fails explicitly instead of ranking unmeasured layers. |
| Lost or stale reference logits could imply zero divergence | Baseline logits must exist and have metadata identifying the actual baseline run and protocol; otherwise comparison fails. |
| Remote jobs did not copy optional protocol settings | The protocol is applied on the remote device before its runs are queued. |
| Imported hardware analysis was replaced with local analysis | Imported analysis is preserved; new local runs store a weight summary so analysis survives removal of local model files. |
| KL values relative to different imported baselines compared directly | Such deltas are marked not comparable. |
| Low roofline utilization asserted “Software overhead” | Bottleneck explanations are hypotheses; missing bandwidth remains unknown, and utilization above the estimate is visible. |
| CPU-only power readings were ignored as a primary source | CPU package energy is available with an explicit partial-device label. |
| vLLM accepted settings its command never applied | Unsupported KV/layer/attention/thread/RAM settings are rejected. Its recorded KV setting is engine default; the single-run UI resets/disables incompatible controls. GPU memory budget is no longer rounded up to a 20% minimum. |
| Guide contained fixed machine-specific study results | It now describes a workflow and evidence to collect without preset numerical conclusions. |
| Packaged service listened on all interfaces by default | Updated service defaults to localhost. Direct network access requires an explicit listener setting plus the remote-access token. |

## Data sources and interpretation

| Output | Source and limits |
|---|---|
| Chat text and usage | Local model-server response; optional Claude agent sends its messages/tool results externally only when selected/configured. |
| llama-bench speed | Subprocess samples. Spread describes variability; this audit does not validate every supported engine's output format. |
| MMLU-Pro | Fixed sampled questions and real option-letter scores. Top-k truncation, prompt format, and limited sample sizes mean these scores should not be treated as published leaderboard equivalents. |
| Perplexity/KLD | llama-perplexity output against stored reference logits. Zero baseline KLD is a self-reference, not evidence of an independently measured perfect model. |
| Energy/cost | Sampled power and actual usage; sensor totals cover only named components. Cost is a calculation using user-entered prices. The sampling model does not establish a confidence interval for energy/token. |
| Roofline and device projection | Calculations based on weight-traffic assumptions and nominal/cached bandwidth, not measurements on the projected device or a measured kernel time breakdown. |
| Quantization simulation | Quantize/dequantize operations on real weights followed by model evaluation. “Fake quantization” is the technique's name; it does not measure packed-kernel speed or deployed memory use. |
| Fresh questions/RULER-style tasks | Generated test inputs with known answers, followed by model inference. Synthetic inputs are intentional; they are not synthetic model scores or an official RULER reproduction. |

## Remaining validation and limitations

- Full new profiles, engine builds, quantization exports, layer scans, LoRA training, Nsight traces, and jobs on another
  physical device were not rerun during this audit. The remote test verifies orchestration against a controlled HTTP
  fixture, not two real machines. Their complete runtime behavior remains to be validated.
- Hardware estimates still simplify model traffic and use nominal GPU lookup data and a cached RAM microbenchmark.
  Treat these as exploratory estimates, particularly for hardware variants, MoE, and mixed CPU/GPU placement.
- Cheap-device presets adjust placement, affinity, and capacity checks; they do not recreate another device's CPU,
  memory controller, or full operating-system memory constraints.
- HTTP workflow coverage is now present, but browser automation has only exercised page loading/read paths. Long-job
  cancellation, recovery after process failure, downloads, and every interactive mutation need dedicated end-to-end coverage.
- Legacy results remain as recorded. The audit did not rewrite the live database or claim their measurements were
  reproduced. Older protocols may contain the fallbacks described above.
- The current large frontend bundle remains a performance issue to address separately.

## Applying the changes

Measurement semantics changed, so `PROTOCOL` is now `blocky-profile/3`. Start a fresh experiment and baseline for new
measurements. The running backend was not restarted by the audit; source changes take effect after its normal restart.
The frontend build regenerates the files served by the application. Service binding changes additionally require
installing the updated systemd unit and reloading systemd; see [setup](SETUP.md#remote-access).
