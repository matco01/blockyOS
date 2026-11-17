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
is byte-identical. Put them in `/opt/blocky/data/reference-values.json`:

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

## What the tests cover

Backend tests cover HTTP queueing, order, frozen settings, cancellation, incomplete or noisy samples, changed conditions,
and raw evidence retrieval. Frontend tests cover evidence labels and missing samples. Browser verification used an
isolated database with a disabled inference worker to exercise evidence, queueing and cancellation. It did not launch
a full GPU validation batch.
