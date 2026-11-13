import type { ExpResults, ExpRun } from "./api";

/* ---------- Metric registry: one place defines every number, its noise and which direction is better ---------- */

export type Ctx = { run: ExpRun; base: ExpRun };
export type Metric = {
  key: string; group: string; label: string; term?: string; unit: string; better: "higher" | "lower" | null;
  get: (r: ExpResults) => number | null | undefined;
  sd?: (r: ExpResults) => number | null | undefined;        // ± noise of one measurement (for significance)
  sig?: (c: Ctx) => boolean | null;                          // custom significance test
  fmt?: (v: number) => string; note?: string;
  minRel?: number;                                           // smallest relative change treated as real (no noise model)
};
export const n1 = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
export const pct = (v: number) => v.toFixed(1);
export const gb = (mb?: number | null) => (mb != null ? mb / 1024 : null);

export const METRICS: Metric[] = [
  { key: "tg", group: "Speed", label: "Decode (writing)", term: "tg", unit: "tok/s", better: "higher", get: (r) => r.speed?.tg, sd: (r) => r.speed?.tg_sd },
  { key: "tg_long", group: "Speed", label: "Decode at 4k context", unit: "tok/s", better: "higher", get: (r) => r.speed?.tg_d4096, sd: (r) => r.speed?.tg_d4096_sd },
  { key: "pp", group: "Speed", label: "Prefill (reading)", term: "pp", unit: "tok/s", better: "higher", get: (r) => r.speed?.pp, sd: (r) => r.speed?.pp_sd, fmt: (v) => v.toFixed(0) },
  { key: "pp_long", group: "Speed", label: "Prefill at 4k context", unit: "tok/s", better: "higher", get: (r) => r.speed?.pp_d4096, sd: (r) => r.speed?.pp_d4096_sd, fmt: (v) => v.toFixed(0) },
  { key: "ttft", group: "Speed", label: "Latency, 512-token prompt", unit: "ms", better: "lower", get: (r) => (r.speed?.pp ? (512 / r.speed.pp) * 1000 : null), fmt: (v) => v.toFixed(0) },
  { key: "mmlu", group: "Quality", label: "MMLU-Pro accuracy", unit: "%", better: "higher", get: (r) => (r.mmlu ? r.mmlu.acc * 100 : null), fmt: pct,
    sig: (c) => {
      // Paired test against whichever run we're comparing with: the baseline, or this run's parent in the tree.
      const s = c.run.stats;
      const p = c.base.id === s?.baseline_id ? s?.mmlu_p : c.base.id === c.run.parent_id ? s?.vs_parent?.mmlu_p : null;
      return p != null ? p < 0.05 : null;
    }, note: "paired McNemar test, p < 0.05" },
  { key: "mmlu_p", group: "Quality", label: "MMLU-Pro P(correct)", unit: "%", better: "higher", get: (r) => (r.mmlu ? r.mmlu.p_correct_mean * 100 : null), fmt: pct },
  { key: "needle", group: "Quality", label: "Long-context retrieval", unit: "%", better: "higher", get: (r) => (r.needle ? r.needle.acc * 100 : null), fmt: (v) => v.toFixed(0),
    note: "Needle in a haystack: a code hidden at 5 depths in 2k…ctx tokens of text", minRel: 0.1 },
  { key: "kld", group: "Quality", label: "KL divergence vs baseline", unit: "", better: "lower", get: (r) => r.kld, fmt: (v) => v.toFixed(4) },
  { key: "kld99", group: "Quality", label: "KLD, 99th percentile", unit: "", better: "lower", get: (r) => r.kld_p99, fmt: (v) => v.toFixed(3) },
  { key: "same_top", group: "Quality", label: "Same top token as baseline", unit: "%", better: "higher", get: (r) => r.same_top, sd: (r) => r.same_top_sd, fmt: pct },
  { key: "ppl", group: "Quality", label: "Perplexity (WikiText-2)", unit: "", better: "lower", get: (r) => r.ppl, sd: (r) => r.ppl_sd, fmt: (v) => v.toFixed(3) },
  { key: "vram_model", group: "Memory", label: "VRAM, model buffers", term: "vram", unit: "GB", better: "lower", get: (r) => gb(r.hw?.headroom?.vram_model_mb), fmt: (v) => v.toFixed(2) },
  { key: "vram_free", group: "Memory", label: "VRAM free", unit: "GB", better: "higher", get: (r) => gb(r.hw?.headroom?.vram_free_mb), fmt: (v) => v.toFixed(2) },
  { key: "kv", group: "Memory", label: "Context memory per 1k tokens", term: "kv", unit: "MB", better: "lower", get: (r) => r.hw?.headroom?.kv_mb_per_1k_tokens },
  { key: "w_cpu", group: "Memory", label: "Weights in system RAM", unit: "GB", better: "lower", get: (r) => r.hw?.weights_on_cpu_gb, fmt: (v) => v.toFixed(2) },
  { key: "load", group: "Memory", label: "Load time", unit: "s", better: "lower", get: (r) => r.load_seconds, minRel: 0.4, note: "Depends on the disk cache: only large changes count" },
  { key: "eff_bw", group: "Hardware", label: "Effective weight bandwidth", unit: "GB/s", better: "higher", get: (r) => r.hw?.decode?.effective_gbps, fmt: (v) => v.toFixed(0) },
  { key: "roof", group: "Hardware", label: "Decode, % of memory roof", unit: "%", better: "higher", get: (r) => (r.hw?.decode?.roof_util != null ? r.hw.decode.roof_util * 100 : null), fmt: (v) => v.toFixed(0) },
  { key: "tflops", group: "Hardware", label: "Prefill compute", unit: "TFLOPS", better: "higher", get: (r) => r.hw?.prefill?.achieved_tflops },
  { key: "croof", group: "Hardware", label: "Prefill, % of compute roof", unit: "%", better: "higher", get: (r) => (r.hw?.prefill?.roof_util != null ? r.hw.prefill.roof_util * 100 : null), fmt: (v) => v.toFixed(0) },
  { key: "clock", group: "Hardware", label: "SM clock, % of max", unit: "%", better: "higher", get: (r) => (r.hw?.clocks.ratio != null ? r.hw.clocks.ratio * 100 : null), fmt: (v) => v.toFixed(0) },
  { key: "temp", group: "Hardware", label: "GPU temperature, peak", unit: "°C", better: "lower", get: (r) => r.hw?.clocks.temp_max_c, fmt: (v) => v.toFixed(0) },
  { key: "gpu_util", group: "Hardware", label: "GPU busy (decode)", unit: "%", better: null, get: (r) => r.hw?.clocks.gpu_util_avg, fmt: (v) => v.toFixed(0) },
  { key: "cpu_util", group: "Hardware", label: "CPU busy (decode)", unit: "%", better: null, get: (r) => r.hw?.clocks.cpu_util_avg, fmt: (v) => v.toFixed(0) },
  { key: "power", group: "Energy", label: "GPU power (decode)", unit: "W", better: "lower", get: (r) => r.hw?.clocks.power_avg_w, fmt: (v) => v.toFixed(0) },
  { key: "tpj", group: "Energy", label: "Tokens per joule (GPU)", term: "tpj", unit: "tok/J", better: "higher",
    get: (r) => (r.speed?.tg && r.hw?.clocks.power_avg_w ? r.speed.tg / r.hw.clocks.power_avg_w : null), fmt: (v) => v.toFixed(2) },
];

const TOKEN_LEVEL = new Set(["ppl", "kld", "kld99", "same_top"]);

/** Difference vs baseline, with a noise check: inside 2σ of the combined measurement noise = not significant. */
export function delta(m: Metric, c: Ctx, session = 0) {
  const v = m.get(c.run.results), b = m.get(c.base.results);
  if (v == null || b == null || b === 0) return null;
  // Token-level metrics are meaningless between models that split text into different tokens.
  // (token_note marks a run whose tokenizer differs from the experiment's baseline.)
  const crossTok = !!c.run.results.token_note !== !!c.base.results.token_note
    || (!!c.run.results.token_note && c.run.config.model.split("-")[0] !== c.base.config.model.split("-")[0]);
  if (TOKEN_LEVEL.has(m.key) && crossTok) return { text: "not comparable", tone: "same", title: "Different tokenizer" };
  const d = (v - b) / Math.abs(b);
  let sig: boolean | null = m.sig ? m.sig(c) : null;
  if (sig == null && m.sd) {
    const s1 = m.sd(c.run.results), s2 = m.sd(c.base.results);
    if (s1 != null && s2 != null) sig = Math.abs(v - b) > 2 * Math.sqrt(s1 * s1 + s2 * s2);
  }
  if (sig == null) sig = Math.abs(d) >= (m.minRel ?? 0.02);
  // Session-to-session drift (measured by repeating the baseline) must also be exceeded.
  if (sig && !m.sig && session > 0 && Math.abs(v - b) <= 2 * session) sig = false;
  const text = `${d > 0 ? "+" : ""}${(d * 100).toFixed(Math.abs(d) < 0.1 ? 1 : 0)}%`;
  if (!sig) return { text, tone: "same", title: "Within measurement noise" };
  if (!m.better) return { text, tone: "same", title: "" };
  const good = m.better === "higher" ? d > 0 : d < 0;
  return { text, tone: good ? "good" : "bad", title: "Significant" };
}


export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];
export const colorOf = (i: number) => SERIES[i] ?? "var(--text-2)";
/** The static metrics plus one row per lm-evaluation-harness task/metric found in these runs. */
export function allMetrics(runs: ExpRun[]): Metric[] {
  const seen = new Map<string, Metric>();
  for (const r of runs) {
    for (const [task, t] of Object.entries(r.results.lm_eval?.tasks ?? {})) {
      for (const k of Object.keys(t.metrics)) {
        if (/^sample_len|samples$/.test(k)) continue;  // a count, not a score
        const key = `lm:${task}:${k}`;
        if (seen.has(key)) continue;
        const pctLike = /acc|match|exact|strict|loose|f1|pass/i.test(k);
        seen.set(key, {
          key, group: "lm-eval", label: `${task} · ${k.replace(",none", "").replace(",", " · ")}`, unit: pctLike ? "%" : "", better: "higher",
          get: (x) => { const v = x.lm_eval?.tasks?.[task]?.metrics?.[k]; return v == null ? null : pctLike ? v * 100 : v; },
          sd: (x) => { const e = x.lm_eval?.tasks?.[task]?.stderr?.[k]; return e == null ? null : pctLike ? e * 100 : e; },
          fmt: (v) => v.toFixed(1), note: "lm-evaluation-harness; ± is its standard error",
        });
      }
    }
  }
  return [...METRICS, ...seen.values()];
}

/** The hardware a run ran on, from its own snapshot: "RTX 2080 SUPER · box-2". */
export const hwLabel = (r: ExpRun) => {
  const gpu = (r.results.hardware?.gpu ?? r.results.provenance?.gpu?.name ?? r.results.imported_from?.gpu ?? "unknown GPU").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "");
  const dev = r.results.hardware?.device ?? r.results.provenance?.host?.device ?? r.results.imported_from?.device;
  return dev ? `${gpu} · ${dev}` : gpu;
};

export const shortModel = (r: ExpRun) => r.config.model.replace(/\.gguf$/, "");
/** A run's display name. The baseline always says what it is: "Baseline · Qwen3.5-9B-Q8_0". */
export const runTitle = (r: ExpRun, isBase = r.parent_id == null) =>
  isBase ? (r.label.startsWith("Baseline ·") ? r.label : `Baseline · ${shortModel(r)}`) : r.label;
export const nameOf = (r: ExpRun, i: number) => runTitle(r, i === 0);
