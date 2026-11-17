import { describe, expect, it } from "vitest";
import type { ExpRun } from "./api";
import { allMetrics, delta, equivText, hwLabel, METRICS, pValue, runTitle } from "./expmetrics";

const run = (id: number, results: Record<string, unknown>, extra: Partial<ExpRun> = {}): ExpRun => ({
  id, experiment_id: 1, created_at: 0, status: "done", label: `run ${id}`, phase: null, error: null, parent_id: id === 1 ? null : 1,
  config: { model: "Qwen3.5-9B-Q4_K_M.gguf", ctx: 8192, kv: "q8_0", ngl: "auto", vram_limit_mb: null, fa: "auto", threads: null },
  results: results as ExpRun["results"], ...extra,
});
const m = (key: string) => METRICS.find((x) => x.key === key)!;

describe("delta: only differences bigger than the noise are colored", () => {
  it("a speed change inside 2σ is grey", () => {
    const d = delta(m("tg"), { base: run(1, { speed: { tg: 60, tg_sd: 2 } }), run: run(2, { speed: { tg: 63, tg_sd: 2 } }) });
    expect(d?.tone).toBe("same");
  });
  it("a speed change with descriptive spread does not claim statistical significance", () => {
    const d = delta(m("tg"), { base: run(1, { speed: { tg: 60, tg_sd: 0.5 } }), run: run(2, { speed: { tg: 66, tg_sd: 0.5 } }) });
    expect(d).toMatchObject({ tone: "same", text: "+10%" });
    expect(d?.title).toContain("not been established");
  });
  it("metrics without uncertainty remain descriptive even for large changes", () => {
    const d = delta(m("kld"), { base: run(1, { kld: 0.01 }), run: run(2, { kld: 0.05 }) });
    expect(d?.tone).toBe("same");
  });
  it("session drift from a repeated setup raises the bar", () => {
    const ctx = { base: run(1, { speed: { tg: 60, tg_sd: 0.2 } }), run: run(2, { speed: { tg: 62, tg_sd: 0.2 } }) };
    expect(delta(m("tg"), ctx)?.tone).toBe("same");
    expect(delta(m("tg"), ctx, 1.5)?.tone).toBe("same");   // a repeat drifted 1.5 tok/s: 2 isn't beyond 2 × 1.5
  });
  it("MMLU-Pro uses the paired test against the baseline, and against the parent", () => {
    const base = run(1, { mmlu: { acc: 0.6 } });
    const sig = run(2, { mmlu: { acc: 0.57 } }, { stats: { mmlu_p: 0.01, baseline_id: 1 } });
    const notSig = run(3, { mmlu: { acc: 0.57 } }, { stats: { mmlu_p: 0.4, baseline_id: 1 } });
    expect(delta(m("mmlu"), { base, run: sig })?.tone).toBe("bad");
    expect(delta(m("mmlu"), { base, run: notSig })?.tone).toBe("same");
    const parent = run(5, { mmlu: { acc: 0.6 } });
    const child = run(6, { mmlu: { acc: 0.59 } }, { parent_id: 5, stats: { baseline_id: 1, vs_parent: { mmlu_p: 0.9 } } });
    expect(delta(m("mmlu"), { base: parent, run: child })?.tone).toBe("same");
  });
  it("equivalence is only claimed against a margin chosen in advance", () => {
    expect(equivText({ margin_pts: 1, equivalent: true, min_margin_pts: 0.8 })).toBe("equivalent within ±1 points (TOST, α = 0.05)");
    expect(equivText({ margin_pts: 1, equivalent: false, min_margin_pts: 2.4 })).toContain("not shown");
    expect(equivText({ min_margin_pts: 2.4 })).toContain("no margin was chosen in advance");
  });
  it("MMLU-Pro shows the paired interval of the difference next to the p-value", () => {
    const base = run(1, { mmlu: { acc: 0.6 } });
    const r = run(2, { mmlu: { acc: 0.57 } }, { stats: { mmlu_p: 0.01, baseline_id: 1, mmlu_diff: { diff: -0.03, ci95: [-0.052, -0.008], n: 560, method: "Newcombe" } } });
    expect(delta(m("mmlu"), { base, run: r })?.title).toContain("difference -3.0 points, 95% CI -5.2 to -0.8");
  });
  it("token-level metrics are not comparable across tokenizers", () => {
    const d = delta(m("ppl"), { base: run(1, { ppl: 6.7 }), run: run(2, { ppl: 13.9, token_note: "different tokenizer" }, { config: { ...run(2, {}).config, model: "Qwen3-4B-Q4_K_M.gguf" } }) });
    expect(d?.text).toBe("not comparable");
  });
  it("missing paired tests never fall back to a percentage threshold", () => {
    expect(delta(m("mmlu"), { base: run(1, { mmlu: { acc: 0.9 } }), run: run(2, { mmlu: { acc: 0.1 } }) })?.tone).toBe("same");
  });
  it("different protocols block both comparison and p-values", () => {
    const base = run(1, { mmlu: { acc: 0.9 }, provenance: { protocol: "v2" } });
    const other = run(2, { mmlu: { acc: 0.1 }, provenance: { protocol: "v3" } }, { stats: { baseline_id: 1, mmlu_p: 0.001 } });
    expect(delta(m("mmlu"), { base, run: other })?.text).toBe("not comparable");
    expect(pValue(m("mmlu"), { base, run: other })).toBeNull();
  });
  it("KL scores relative to different imported baselines are not comparable", () => {
    const base = run(1, { kld: 0.01 });
    const other = run(2, { kld: 0.02, imported_from: { imported_at: 123 } });
    expect(delta(m("kld"), { base, run: other })?.text).toBe("not comparable");
  });
  it("no retrieval trials does not mean zero accuracy", () => {
    expect(m("needle").get(run(1, { needle: { cells: [], acc: 0 } }).results)).toBeNull();
  });
});

describe("names", () => {
  it("the baseline always says what it is", () => {
    expect(runTitle(run(1, {}, { label: "Baseline" }))).toBe("Baseline · Qwen3.5-9B-Q4_K_M");
    expect(runTitle(run(1, {}, { label: "Baseline · X · KV f16" }))).toBe("Baseline · X · KV f16");
    expect(runTitle(run(2, {}, { label: "KV q4_0" }))).toBe("KV q4_0");
  });
  it("hardware label comes from the run's own snapshot", () => {
    expect(hwLabel(run(2, { hardware: { gpu: "NVIDIA GeForce RTX 4090", device: "lab-2" } }))).toBe("RTX 4090 · lab-2");
    expect(hwLabel(run(2, {}))).toBe("unknown GPU");
  });
});

describe("lm-eval metrics", () => {
  it("become rows, without sample counts", () => {
    const r = run(2, { lm_eval: { tasks: { gsm8k: { metrics: { "exact_match,strict-match": 0.36, sample_len: 50 }, stderr: { "exact_match,strict-match": 0.07 } } } } });
    const rows = allMetrics([r]).filter((x) => x.group === "lm-eval");
    expect(rows.map((x) => x.key)).toEqual(["lm:gsm8k:exact_match,strict-match"]);
    expect(rows[0].get(r.results)).toBeCloseTo(36);
    expect(rows[0].sd!(r.results)).toBeCloseTo(7);
  });
});

describe("registered evaluator metrics", () => {
  const result = (value: number, overrides = {}) => ({ research_metrics: { bits_per_byte: { value, unit: "bit/byte", protocol: "eval/1", task: "text", dataset_sha256: "hash", scope: "all bytes", kind: "measured", better: "lower", spread: { n: 3, sd: .1 }, ...overrides } } });
  it("shows shared values and spread without inventing native measurements", () => {
    const r = run(1, result(2));
    const metric = allMetrics([r]).find((m) => m.researchName === "bits_per_byte")!;
    expect(metric.get(r.results)).toBe(2);
    expect(metric.sd!(r.results)).toBe(.1);
    expect(METRICS.find((m) => m.key === "tg")!.get(r.results)).toBeUndefined();
  });
  it("blocks differences across changed datasets and units", () => {
    const base = run(1, result(2));
    const other = run(2, result(1, { dataset_sha256: "different" }));
    const metric = allMetrics([base, other]).find((m) => m.researchName === "bits_per_byte")!;
    expect(delta(metric, { base, run: other })?.text).toBe("not comparable");
    expect(delta(metric, { base, run: run(2, result(1)) })?.tone).toBe("same");
  });
});
