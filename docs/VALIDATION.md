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
and 3–12 pairs. The action queues full new profiles in A, B, A, B order. Six pairs is the minimum that can produce
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

The analysis assumes independent paired measurements under stable conditions. Fixed A/B ordering does not eliminate
order effects. Each batch tests one preselected metric; trying many batches or metrics and selecting a favorable one
does not preserve a study-wide 5% false-positive rate. The result does not establish quality preservation or overall
model superiority. Inspect quality separately; an inconclusive result does not prove equivalence.

The implementation adds storage and analysis, without changing how an individual profile is measured, so it uses the
existing profile protocol. Validation analysis itself records `paired-median-sign/1` in each saved plan.

Backend tests cover HTTP queueing, order, frozen settings, cancellation, incomplete or noisy samples, changed conditions,
and raw evidence retrieval. Frontend tests cover evidence labels and missing samples. Browser verification used an
isolated database with a disabled inference worker to exercise evidence, queueing and cancellation. It did not launch
a full GPU validation batch.
