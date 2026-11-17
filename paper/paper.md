---
title: 'blockyOS: a local laboratory for measuring language-model inference on the hardware at hand'
tags:
  - Python
  - large language models
  - quantization
  - inference
  - benchmarking
  - reproducibility
authors:
  - name: FIRST LAST            # TODO: author name(s) as they should appear
    orcid: 0000-0000-0000-0000  # TODO
    affiliation: 1
affiliations:
  - name: TODO affiliation
    index: 1
date: 17 November 2026
bibliography: paper.bib
---

# Summary

Running large language models locally forces many choices: how strongly to quantize the weights, which inference
engine and which version of it, how much context to allocate and at what precision, how to split a model across GPUs
or machines. Each choice trades quality, speed, memory and energy, and the trade-offs depend on the specific model and
hardware. blockyOS is a self-hosted laboratory that measures them. A researcher defines an experiment around one model,
changes one setting at a time, and every resulting configuration is profiled with the same versioned protocol: paired
accuracy on a fixed MMLU-Pro sample [@mmlupro], Kullback–Leibler divergence of the output distribution against the
baseline's logits on held-out text, long-context retrieval, speed from llama-bench and from an engine-neutral serving
client, energy per token, and a roofline-style analysis [@roofline] of how close the run came to the hardware's limits.
It drives llama.cpp [@llamacpp] builds from any commit, and vLLM [@vllm], SGLang [@sglang] and TensorRT-LLM for
original checkpoints, on one GPU, several GPUs, or several machines. A PyTorch side covers weight-quantization
simulation (GPTQ- and AWQ-style references [@gptq; @awq]), activation inspection, and LoRA, full fine-tuning and
pretraining whose results flow back into experiments.

# Statement of need

Numbers about local inference are usually produced ad hoc: a single benchmark invocation, accuracy on a handful of
prompts, speed without a spread, on a machine whose configuration is not recorded. Such results cannot be compared
across setups, let alone across papers. Existing tools address parts of the problem — llama-bench measures speed,
lm-evaluation-harness [@lmeval] measures task accuracy, llama-perplexity measures fidelity — but leave experimental
design, provenance and statistics to the user.

blockyOS targets researchers and practitioners who need defensible comparisons of inference configurations on
specific hardware, from consumer GPUs and CPU-only devices to multi-GPU and multi-machine setups. Its design choices
address common failure modes of such comparisons:

- **Provenance and comparability.** Every run records the protocol version, engine commit, SHA-256 of the model and
  datasets, driver and CUDA versions, test settings and the exact commands. Comparisons between runs whose conditions
  differ are refused rather than shown.
- **Paired statistics.** Accuracy differences use McNemar's exact test [@mcnemar] on the shared questions with a
  paired Newcombe interval [@newcombe]; many comparisons against one baseline are Holm-corrected [@holm]; equivalence
  claims require a preselected margin and use two one-sided tests [@tost]. Speed differences are confirmed by
  counterbalanced repeated pairs with a sign test and a distribution-free interval for the median.
- **Instrument checks.** Properties every correct pipeline must have — a baseline rerun yields near-zero KL divergence
  and identical greedy answers, fewer bits per weight do not yield lower divergence, decode traffic does not exceed the
  memory bandwidth — are evaluated on the user's own runs, so measurement faults surface before conclusions are drawn.
- **Preregistration and reproducibility.** Hypotheses and analysis plans can be recorded, append-only, before data
  exist; each experiment exports as a bundle with provenance, raw evaluator output and a script that re-runs and
  compares every configuration on another device.

The measurement protocol is specified in the repository (`docs/PROTOCOL.md`). The software includes a test suite that
runs complete experiments through the pipeline against a simulated engine, and an MCP server so AI assistants can plan
and queue experiments under the same rules.

# Acknowledgements

TODO.

# References
