import type { ExpResults, ExpRun } from "./api";
import { adjust, Correction, zTestP } from "./stats";

/* ---------- Metric registry: one place defines every number, its noise and which direction is better ---------- */

export type Ctx = { run: ExpRun; base: ExpRun };
export type Metric = {
  researchName?: string;
  key: string; group: string; label: string; term?: string; unit: string; better: "higher" | "lower" | null;
  get: (r: ExpResults) => number | null | undefined;
  sd?: (r: ExpResults) => number | null | undefined;        // descriptive spread shown next to the value
  se?: (r: ExpResults) => number | null | undefined;        // standard error, only when the source explicitly reports one
  p?: (c: Ctx) => number | null | undefined;                 // custom test: p-value for this pair (e.g. paired McNemar)
  fmt?: (v: number) => string; note?: string;
  minRel?: number;                                           // legacy display metadata; never a significance test
};
const esrc = (r: ExpResults) => (r.energy?.primary ? r.energy.sources?.[r.energy.primary] : undefined);
export const n1 = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
export const pct = (v: number) => v.toFixed(1);
export const gb = (mb?: number | null) => (mb != null ? mb / 1024 : null);

export const METRICS: Metric[] = [
  { key: "tg", group: "Speed", label: "Decode (writing)", term: "tg", unit: "tok/s", better: "higher", get: (r) => r.speed?.tg, sd: (r) => r.speed?.tg_sd },
  { key: "tg_long", group: "Speed", label: "Decode at 4k context", unit: "tok/s", better: "higher", get: (r) => r.speed?.tg_d4096, sd: (r) => r.speed?.tg_d4096_sd },
  { key: "pp", group: "Speed", label: "Prefill (reading)", term: "pp", unit: "tok/s", better: "higher", get: (r) => r.speed?.pp, sd: (r) => r.speed?.pp_sd, fmt: (v) => v.toFixed(0) },
  { key: "pp_long", group: "Speed", label: "Prefill at 4k context", unit: "tok/s", better: "higher", get: (r) => r.speed?.pp_d4096, sd: (r) => r.speed?.pp_d4096_sd, fmt: (v) => v.toFixed(0) },
  { key: "srv_tg", group: "Speed", label: "Serving: output throughput", unit: "tok/s", better: "higher", get: (r) => r.serving?.tg, sd: (r) => r.serving?.tg_sd,
    note: "Protocol 3: server-reported output tokens / full request time, including prefill. Older protocols measured a different interval; check provenance." },
  { key: "srv_pp", group: "Speed", label: "Serving: prompt / first-chunk latency", unit: "tok/s", better: "higher", get: (r) => r.serving?.pp, sd: (r) => r.serving?.pp_sd, fmt: (v) => v.toFixed(0), note: "A client-side latency proxy, not isolated prefill speed" },
  { key: "srv_ttft", group: "Speed", label: "Serving: time to first token", unit: "ms", better: "lower", get: (r) => r.serving?.ttft_ms, sd: (r) => r.serving?.ttft_sd, fmt: (v) => v.toFixed(0) },
  { key: "ttft", group: "Speed", label: "Latency, 512-token prompt", unit: "ms", better: "lower", get: (r) => (r.speed?.pp ? (512 / r.speed.pp) * 1000 : null), fmt: (v) => v.toFixed(0) },
  { key: "mmlu", group: "Quality", label: "MMLU-Pro accuracy", unit: "%", better: "higher", get: (r) => (r.mmlu ? r.mmlu.acc * 100 : null), fmt: pct,
    p: (c) => {
      // Paired test against whichever run we're comparing with: the baseline, or this run's parent in the tree.
      const s = c.run.stats;
      return c.base.id === s?.baseline_id ? s?.mmlu_p : c.base.id === c.run.parent_id ? s?.vs_parent?.mmlu_p : null;
    }, note: "paired McNemar test" },
  { key: "mmlu_p", group: "Quality", label: "MMLU-Pro returned-option score", unit: "%", better: "higher", get: (r) => (r.mmlu ? r.mmlu.p_correct_mean * 100 : null), fmt: pct,
    note: "Renormalized over option letters in the server's top-k shortlist; not the model's full probability of the correct answer" },
  { key: "needle", group: "Quality", label: "Long-context retrieval", unit: "%", better: "higher", get: (r) => (r.needle?.cells.length && r.needle.acc != null ? r.needle.acc * 100 : null), fmt: (v) => v.toFixed(0),
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
  { key: "e_load", group: "Energy", label: "Power while generating", unit: "W", better: "lower", get: (r) => esrc(r)?.load_w.avg, sd: (r) => esrc(r)?.load_w.sd, fmt: (v) => v.toFixed(0),
    note: "Primary power source: wall meter if set, else GPU + CPU sensors, else GPU sensor" },
  { key: "power", group: "Energy", label: "GPU power during speed tests", unit: "W", better: "lower", get: (r) => r.hw?.clocks.power_avg_w, fmt: (v) => v.toFixed(0) },
  { key: "e_idle", group: "Energy", label: "Power at idle (model loaded)", unit: "W", better: "lower", get: (r) => esrc(r)?.idle_w?.avg, sd: (r) => esrc(r)?.idle_w?.sd, fmt: (v) => v.toFixed(0) },
  { key: "jpt", group: "Energy", label: "Energy per token", unit: "J", better: "lower", get: (r) => esrc(r)?.j_per_token, fmt: (v) => v.toFixed(3), minRel: 0.05,
    note: "Protocol 3: sampled energy over the full request, including prefill, divided by server-reported output tokens. Sensor totals cover only the named components." },
  { key: "jpt_net", group: "Energy", label: "Energy per token above idle", unit: "J", better: "lower", get: (r) => esrc(r)?.net_j_per_token, fmt: (v) => v.toFixed(3), minRel: 0.05 },
  { key: "tpwh", group: "Energy", label: "Tokens per watt-hour", term: "tpj", unit: "tok/Wh", better: "higher", get: (r) => esrc(r)?.tokens_per_wh, fmt: (v) => v.toFixed(0), minRel: 0.05 },
  { key: "cost_mtok", group: "Energy", label: "Electricity per 1M tokens", unit: "", better: "lower", get: (r) => r.cost?.electricity_per_mtok, fmt: (v) => v.toFixed(3), minRel: 0.05,
    note: "Energy per token × the electricity price set in Settings when the run was made" },
  { key: "tps_price", group: "Energy", label: "Decode speed per 1000 of device price", unit: "tok/s", better: "higher", get: (r) => r.cost?.tps_per_1k_price, fmt: (v) => v.toFixed(1),
    note: "Decode tok/s ÷ device price × 1000, with the price set in Settings when the run was made" },
];

const TOKEN_LEVEL = new Set(["ppl", "kld", "kld99", "same_top"]);
const RELATIVE_TO_BASELINE = new Set(["kld", "kld99", "same_top"]);

/** Refuse comparisons whose measurement protocol or reference distribution differs. */
function incomparable(m: Metric, c: Ctx): string | null {
  const a = c.base.results, b = c.run.results;
  if (m.researchName) {
    const x = a.research_metrics?.[m.researchName], y = b.research_metrics?.[m.researchName];
    if (!x || !y) return "The evaluator did not report this shared metric";
    for (const key of ["unit", "protocol", "task", "dataset_sha256", "scope", "kind", "better"] as const) {
      if (x[key] !== y[key] || key !== "better" && !x[key]) return `Different or missing ${key}`;
    }
    if (x.unit?.toLowerCase().includes("token") && (!x.tokenizer_sha256 || x.tokenizer_sha256 !== y.tokenizer_sha256)) return "Different or unknown tokenizers";
    return null;
  }
  if (a.provenance?.protocol !== b.provenance?.protocol) return "Different or missing measurement protocols";
  const stats = c.base.id === c.run.stats?.baseline_id ? c.run.stats : c.base.id === c.run.parent_id ? c.run.stats?.vs_parent : null;
  if (stats?.env_mismatch?.length) return `Different measurement conditions: ${stats.env_mismatch.join(", ")}`;
  if (TOKEN_LEVEL.has(m.key) && (a.token_note || b.token_note)) return "Tokenizer compatibility is not established for this pair";
  if (RELATIVE_TO_BASELINE.has(m.key) && (a.imported_from?.imported_at !== b.imported_from?.imported_at)) return "Different KL reference baselines";
  return null;
}

/** p-value of the difference between two runs on one metric, or null when the metric has no noise model. */
export function pValue(m: Metric, c: Ctx): number | null {
  if (incomparable(m, c)) return null;
  if (m.p) return m.p(c) ?? null;
  const v = m.get(c.run.results), b = m.get(c.base.results);
  if (v == null || b == null || !m.se) return null;
  const s1 = m.se(c.run.results), s2 = m.se(c.base.results);
  return s1 != null && s2 != null && Number.isFinite(s1) && Number.isFinite(s2) && s1 > 0 && s2 > 0 ? zTestP(v - b, s1, s2) : null;
}

/** Adjusted p-values for many comparisons on one metric at once (run id -> adjusted p). */
export function adjustedP(m: Metric, ctxs: Ctx[], method: Correction): Map<number, number> {
  const withP = ctxs.map((c) => [c.run.id, pValue(m, c)] as const).filter((x): x is readonly [number, number] => x[1] != null);
  const adj = adjust(withP.map((x) => x[1]), method);
  return new Map(withP.map((x, i) => [x[0], adj[i]]));
}

export const fmtP = (p: number) => (p < 0.001 ? "< 0.001" : p.toFixed(3));

/** Color only comparisons supported by a statistical test. Descriptive differences remain neutral. */
export function delta(m: Metric, c: Ctx, session = 0, pAdj?: number | null) {
  const v = m.get(c.run.results), b = m.get(c.base.results);
  if (v == null || b == null || !Number.isFinite(v) || !Number.isFinite(b)) return null;
  const reason = incomparable(m, c);
  if (reason) return { text: "not comparable", tone: "same", title: reason };
  if (b === 0) return null;
  const d = (v - b) / Math.abs(b);
  const p = pValue(m, c);
  const pa = pAdj ?? p;
  const text = `${d > 0 ? "+" : ""}${(d * 100).toFixed(Math.abs(d) < 0.1 ? 1 : 0)}%`;
  if (pa == null) return { text, tone: "same", title: "Observed difference; statistical significance has not been established. Reported spread is not a standard error." };
  let sig = pa < 0.05;
  // Session-to-session drift (measured by repeating the baseline) must also be exceeded.
  if (sig && !m.p && session > 0 && Math.abs(v - b) <= 2 * session) sig = false;
  const ptxt = `p = ${fmtP(pa)}${pAdj != null && pAdj !== p ? " after correcting for multiple comparisons" : ""}`;
  if (!sig) return { text, tone: "same", title: `No statistically established difference (${ptxt}); this does not prove equivalence` };
  if (!m.better) return { text, tone: "same", title: ptxt };
  const good = m.better === "higher" ? d > 0 : d < 0;
  return { text, tone: good ? "good" : "bad", title: `Significant (${ptxt})` };
}


export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];
export const colorOf = (i: number) => SERIES[i] ?? "var(--text-2)";
/** The static metrics plus one row per lm-evaluation-harness task/metric found in these runs. */
export function allMetrics(runs: ExpRun[]): Metric[] {
  const seen = new Map<string, Metric>();
  for (const r of runs) {
    for (const [name, event] of Object.entries(r.results.research_metrics ?? {})) {
      const key = `research:${name}`;
      if (seen.has(key)) continue;
      seen.set(key, { key, researchName: name, group: "Shared evaluations", label: name.replaceAll("_", " "), unit: event.unit ?? "",
        better: event.better === "higher" || event.better === "lower" ? event.better : null,
        get: (x) => x.research_metrics?.[name]?.value,
        sd: (x) => x.research_metrics?.[name]?.spread?.sd,
        note: "Evaluator-reported metric. Compare only matching task, dataset, protocol and scope; expand evidence to inspect the contract." });
    }
  }
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
          se: (x) => { const e = x.lm_eval?.tasks?.[task]?.stderr?.[k]; return e == null ? null : pctLike ? e * 100 : e; },
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
