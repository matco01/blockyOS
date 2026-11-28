# Hardware lab

The hardware lab runs a real model on a simulated chip and measures what the chip's arithmetic does to the model. Level 1
(this document) covers the chip's **arithmetic**: analog in-memory computing with device physics, and digital number
formats. Speed, energy and circuits are the later levels ([roadmap](#roadmap-the-rest-of-the-lab)).

In the app: All tools → **Hardware lab**. API: `/api/hwlab/*`. Simulator: `backend/torch_tools/hwsim.py` (runs in the ML
environment as a PyTorch job: the chat model pauses, GPU safety applies). Chip presets and checks: `backend/app/hwlab.py`.

## What is simulated

Every linear layer inside the decoder layers (attention projections, MLP, the projections of linear-attention layers) runs
on the chip. Attention scores (Q·Kᵀ, softmax·V), norms, embeddings, the output head and the non-matrix parts of
linear-attention layers (convolutions, gates) stay exact digital: they multiply tensors that change with every token,
which a programmed array can't hold. Each run reports how many matrices and weights were on the chip.

### Analog in-memory computing

**Mapping.** A weight matrix is split into tiles of `tile_rows` inputs (default 512; 0 = one tile per matrix). Within a
tile, each column's weights are scaled so the largest magnitude is 1 (`weight_scaling: column`; `tile` uses one scale per
tile), and each weight is stored on a device pair: `G+ = max(w, 0)·g_max`, `G− = max(−w, 0)·g_max` (`g_max` default
25 µS). With `levels ≥ 2`, targets are first rounded to that many conductance levels between 0 and `g_max`.

**Computing.** Each input vector is scaled per tile so its largest entry is 1, then quantized by the DAC to a symmetric
grid of ±(2^(b−1) − 1) steps (`dac_bits`, 0 = ideal). The tile computes `y = Σ (G+ − G−)/g_max · x` (Ohm's and Kirchhoff's
laws), optionally plus fresh Gaussian noise `out_noise` per output, then the ADC clips `y` to ±range and quantizes it to
±(2^(b−1) − 1) steps (`adc_bits`). The range is in full-scale units (one full-scale weight times one full-scale input),
either fixed (`adc_range: number`) or `auto`: per tile, the `adc_percentile` (default 99.99) of the ideal tile outputs on
calibration text (WikiText-2 *train*, never the evaluation text). Outputs are scaled back and the tiles are added
digitally; the bias is added digitally.

**Device physics** (each optional; conductances in µS, per device):

| Effect | Model |
|---|---|
| Programming noise | `G = max(0, G_target + σ_prog(G_target)·ε)`. PCM: `σ_prog = (0.26348 + 1.9650·r − 1.1731·r²)·g_max/25`, `r = G_target/g_max`. Constant: `σ_prog = sigma·g_max`. |
| Stuck devices | A fraction `stuck.zero` at 0 and `stuck.max` at `g_max`, from the start; they never drift or fluctuate. |
| Drift | `G(t) = G_prog·((t + t₀)/t₀)^(−ν)`, `t₀ = 20 s`, `t = t_seconds`. PCM: `ν = |μ + s·ε|` with `μ = clip(−0.0155·ln r + 0.0244, 0.049, 0.1)`, `s = clip(−0.0125·ln r − 0.0059, 0.008, 0.045)`. Constant: `ν = |nu + nu_sd·ε|`. |
| Read noise | Accumulated 1/f noise at time `t`: `G = max(0, G + |G|·σ_read·ε)`. PCM: `σ_read = q·√ln((t + t₀ + t_read)/(2·t_read))`, `q = min(0.2, 0.0088 / max(10⁻³, (G_prog/g_max)^0.65))`, `t_read = 250 ns`. Constant: `σ_read = sigma`. |
| Drift compensation | Per tile, outputs × `mean|w|` right after programming / `mean|w|` at time `t` (one global factor, measured on the chip). |

`ε` is a standard normal sample. The PCM models port IBM AIHWKit's `PCMLikeNoiseModel` (MIT licence) equation by
equation; AIHWKit cites Nandakumar et al., ICECS 2019 for the device measurements. As in AIHWKit, read noise applies
even at `t = 0` (the reference read is 20 s after programming). Tests check every constant and distribution
(`backend/tests/test_hwsim.py`).

### Digital number formats

| Format | Definition |
|---|---|
| `int8`, `int4` | Symmetric grid ±(2^(b−1) − 1), one scale per output (weights) or per token (inputs); weights optionally in groups of 32–256. |
| `fp8_e4m3` | Scaled so the largest value is 448, then rounded to E4M3 (PyTorch's conversion, round to nearest even). |
| `mxfp4` | OCP MX v1.0: blocks of 32 values share the scale `2^(⌊log₂ max|v|⌋ − 2)`; values are FP4 E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6), saturating at 6. Ties round toward zero. |
| `fp16` | Unchanged (the check). |

Products are accumulated exactly (fp32).

## What is measured

The same held-out text (WikiText-2 *test*, chunks of 512 tokens) runs through the exact model and through the model on
the chip, layer by layer (so models larger than the GPU fit):

- **KL divergence** per token between the two next-token distributions (mean ± standard error per chip, and its 99th
  percentile), **same top token** (%), and **perplexity** of both.
- **Chips.** Each trial is a separately manufactured chip: new programming noise, drift coefficients, read noise and stuck
  devices (seeded, reproducible). Summaries give the mean, standard deviation, 95% interval (t distribution) and range
  over chips. Chips without anything random run once.
- **Per matrix.** Output error `‖y_chip − y_exact‖ / ‖y_exact‖` on the same input (each matrix's own error, given the
  inputs the chip model actually sees), and weight error `‖W_chip − W‖ / ‖W‖` of the effective weights. Reported per
  matrix type and per layer, pooled over chips.
- **Checks.** The presets *Exact* and *Ideal analog chip* must give a KL divergence of about 0: they confirm the
  pipeline. Run them first on a new model.

Sweeps run the same model and chip with one setting varied (2–12 values, one run each) and plot the KL divergence with its
95% interval. Every run and sweep exports CSV; a run's CSV includes the full chip description and seed.

## Presets

| Preset | Chip | Source |
|---|---|---|
| Exact (check) | Digital FP16 | — |
| Ideal analog chip (check) | Tiles, device pairs, scaling; no imperfections, ideal converters | — |
| Digital INT8 | INT8 weights per output, INT8 inputs per token | — |
| Digital INT4 weights, INT8 inputs | INT4 in groups of 128, INT8 inputs | — |
| Digital FP8 (E4M3) | FP8 weights and inputs | Micikevicius et al., FP8 Formats for Deep Learning (2022) |
| Digital MXFP4 | MXFP4 weights and inputs | OCP Microscaling Formats (MX) Specification v1.0 (2023) |
| Analog PCM, 1 hour / 1 year | PCM programming noise, drift and read noise; 8-bit DAC and ADC (calibrated range), 512-row tiles, drift compensation | IBM AIHWKit `PCMLikeNoiseModel`; Nandakumar et al., ICECS 2019 |
| Analog RRAM-like, 16-level cells | 16 levels, 2% programming noise, 0.5% stuck at 0, 0.2% at `g_max` | **Illustrative values**, not fitted to a device |

The converter and tile settings of the PCM presets are a common design point, not a specific chip. Change any setting
under *Chip settings*; the run is then labelled "(changed)".

## Not modeled

IR drop (wire resistance) and sneak paths; temperature (the PCM constants are for the conditions they were measured at);
converter noise and nonlinearity beyond quantization (except the optional output noise); how inputs are encoded in time
(pulse width, bit-serial); precision of the digital parts (exact); speed, energy and area (level 2). Results are
estimates under these assumptions, not measurements of a chip.

## Reproducing a run

```
python backend/torch_tools/hwsim.py --model <checkpoint folder> --text wikitext2-test.txt --calib wikitext2-train-calib.txt \
  --chip chip.json --trials 5 --seed 0 --chunks 16 --out result.json [--device cpu]
```

`chip.json` is the `chip_json` column of the run's CSV. The same seed, chip and software give the same chips.
`--device cpu` works for any model PyTorch runs on the CPU, but models with linear attention (Qwen3.5) need an
environment without flash-linear-attention: transformers uses its kernels whenever they're installed, and they only run on
NVIDIA GPUs.

## Roadmap: the rest of the lab

1. **Arithmetic** (this): does the model survive the chip's math?
2. **Architecture**: memory capacity and bandwidth, compute, on-chip SRAM and energy per byte and per operation, turned
   into tokens per second, energy per token and chip area for any model on the device. Calibrated first against what
   Blocky measures on real GPUs, and open to detailed tools (Timeloop and Accelergy, CiMLoop for compute-in-memory).
3. **Circuits**: SPICE (ngspice) for memory cells and converters, Verilog with Yosys and OpenROAD on open process kits for
   digital blocks; their measured numbers (energy per operation, noise, area) become the settings of levels 1 and 2.
4. **Silicon**: small test chips on shared shuttle runs, measured against the simulation.

## References

- IBM AIHWKit, `aihwkit.inference.noise.pcm.PCMLikeNoiseModel` and `GlobalDriftCompensation`, https://github.com/IBM/aihwkit (MIT licence).
- S. R. Nandakumar et al., phase-change memory models for deep learning training and inference, ICECS 2019 (the source AIHWKit cites).
- J. Büchel et al., Analog Foundation Models, NeurIPS 2025 (arXiv:2505.09663): language models adapted to the same kind of analog noise.
- Open Compute Project, OCP Microscaling Formats (MX) Specification v1.0, 2023.
- P. Micikevicius et al., FP8 Formats for Deep Learning, 2022 (arXiv:2209.05433).
- T. Andrulis, J. Emer, V. Sze, CiMLoop: A Flexible, Accurate, and Fast Compute-In-Memory Modeling Tool, ISPASS 2024 (arXiv:2405.07259).
