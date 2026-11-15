import type { ExpRun } from "./api";
import type { Metric } from "./expmetrics";

export function evidence(m: Metric, run: ExpRun) {
  const r = run.results;
  const value = m.get(r);
  const valid = value != null && Number.isFinite(value);
  if (m.researchName) {
    const event = r.research_metrics?.[m.researchName];
    return { kind: !valid ? "Unavailable" : event?.kind === "estimated" ? "Estimated" : event?.kind === "simulated" ? "Simulated" : "Measured",
      source: event?.protocol ?? "Registered evaluator", raw: { metric: event, provenance: r.research_provenance },
      reason: valid ? "Reported by the evaluator; inspect its dataset, scope, samples and source evidence." : "This evaluator did not supply this metric.", simulated: event?.kind === "simulated" };
  }
  const simulated = !!run.config.preset || !!run.config.vram_limit_mb || !!run.config.ram_limit_mb || run.config.device === "cpu" && !!run.config.cores;
  const estimated = ["ttft", "srv_pp", "mmlu_p", "w_cpu", "kv", "vram_free", "eff_bw", "roof", "tflops", "croof", "clock", "cost_mtok", "tps_price"].includes(m.key)
    || ["kld", "same_top"].includes(m.key) && !!(r as unknown as Record<string, unknown>).kld_reference;
  const kind = !valid ? "Unavailable" : estimated ? "Estimated" : simulated ? "Simulated" : "Measured";
  const x = r as unknown as Record<string, unknown>;
  const speed = r.speed as Record<string, unknown> | undefined;
  const serving = r.serving as unknown as Record<string, unknown> | undefined;
  const speedKey = ({ tg_long: "tg_d4096", pp_long: "pp_d4096" } as Record<string, string>)[m.key] ?? m.key;
  let source: string;
  let raw: unknown;
  if (["tg", "tg_long", "pp", "pp_long", "ttft"].includes(m.key)) {
    source = "llama-bench repetitions";
    raw = speed?.[`${speedKey === "ttft" ? "pp" : speedKey}_samples`];
  } else if (m.key.startsWith("srv_")) {
    source = r.serving?.method ?? "Model server streaming response";
    raw = (serving?.samples as Record<string, unknown> | undefined)?.[m.key === "srv_ttft" ? "ttft_ms" : m.key.slice(4)];
  } else if (m.key.startsWith("mmlu")) {
    source = r.mmlu?.protocol ?? "MMLU-Pro evaluation";
    raw = r.mmlu;
  } else if (m.key === "needle") {
    source = "Needle retrieval trials"; raw = r.needle;
  } else if (["ppl", "kld", "kld99", "same_top"].includes(m.key)) {
    source = "llama-perplexity evaluator output";
    raw = Object.fromEntries(Object.entries(x).filter(([k]) => /^(ppl|kld|same_top)/.test(k)));
  } else if (m.group === "Energy") {
    source = r.energy?.primary_label ?? "Power sensors and server token usage"; raw = { energy: r.energy, cost: r.cost };
  } else if (m.group === "lm-eval") {
    source = "lm-evaluation-harness"; raw = r.lm_eval;
  } else {
    source = estimated ? "Calculated from stored measurements and model/hardware assumptions" : "Model load log or sampled telemetry";
    raw = { memory: r.memory, telemetry: r.telemetry, analysis: r.hw };
  }
  const reason = valid ? (estimated ? "Calculated value; inspect the inputs and assumptions below." : simulated ? "Measured on this hardware with target-device constraints; it is not a measurement on the projected device." : "Obtained from an evaluator or sensor on the recorded device.")
    : run.error ?? r.energy?.error ?? r.energy?.note ?? r.fidelity_note ?? r.token_note ?? "This metric was not recorded, is unsupported, or has no valid samples.";
  return { kind, source, raw, reason, simulated };
}
