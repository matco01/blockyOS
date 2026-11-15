import { useState } from "react";
import { api, Experiment, ExpRun } from "./api";
import { allMetrics, runTitle } from "./expmetrics";

const conclusions: Record<string, string> = { pending: "Waiting for all repeats", insufficient: "Insufficient evidence to choose a setup", candidate_better: "Candidate improvement supported for the selected metric", reference_better: "Reference advantage supported for the selected metric" };
const fmt = (v: number | null) => v == null ? "—" : v.toFixed(2);
export default function ValidationPanel({ exp, runs, onChanged }: { exp: Experiment; runs: ExpRun[]; onChanged: () => void }) {
  const choices = runs.filter((r) => r.status === "done" && !r.results.imported_from);
  const shared = allMetrics(choices).filter((m) => m.researchName && m.better && choices.some((r) => r.config.engine === "research" && m.get(r.results) != null));
  const [reference, setReference] = useState(choices[0]?.id ?? 0);
  const [candidate, setCandidate] = useState(choices[1]?.id ?? 0);
  const [pairs, setPairs] = useState(6), [metric, setMetric] = useState(choices[0]?.config.engine === "research" ? shared[0]?.key ?? "tg" : "tg"), [minimum, setMinimum] = useState(3);
  const [error, setError] = useState(""), [busy, setBusy] = useState(false);
  const queue = async () => {
    setBusy(true); setError("");
    try { await api.experimentValidate(exp.id, { reference_id: reference, candidate_id: candidate, pairs, metric, min_effect_pct: minimum }); onChanged(); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const select = (label: string, value: number, set: (v: number) => void) => <label>{label}<select className="field select" aria-label={label} value={value} onChange={(e) => set(Number(e.target.value))}><option value={0}>Choose a run</option>{choices.map((r) => <option key={r.id} value={r.id}>#{r.id} {runTitle(r)}</option>)}</select></label>;
  return <div className="group pad xv-panel">
    <b>Repeat and validate</b><p className="row-sub">Queue fresh measurements in A → B → A → B order. Choose the metric and minimum meaningful improvement before running. Six pairs is the minimum for a finite distribution-free 95% interval. Test settings are saved with the batch.</p>
    <div className="xv-controls">{select("Reference (A)", reference, setReference)}{select("Candidate (B)", candidate, setCandidate)}
      <label>Metric<select className="field select" value={metric} onChange={(e) => setMetric(e.target.value)}>{shared.map((m) => <option key={m.key} value={m.key}>{m.label} (shared evaluator)</option>)}<option value="tg">Decode speed</option><option value="pp">Prefill speed</option><option value="srv_tg">Serving output throughput</option><option value="jpt">Energy per output token</option></select></label>
      <label>Pairs<select className="field select" value={pairs} onChange={(e) => setPairs(Number(e.target.value))}>{[3, 6, 8, 10, 12].map((n) => <option key={n}>{n}</option>)}</select></label>
      <label>Minimum improvement (%)<input className="field" type="number" min={0} max={100} value={minimum} onChange={(e) => setMinimum(Number(e.target.value))} /></label>
      <button className="btn" disabled={busy || !reference || !candidate || reference === candidate} onClick={queue}>Queue {pairs * 2} validation runs</button>
    </div><p className="row-sub">Each item reruns its evaluator; native profiles can take several minutes or longer. Conclusions apply to this metric only; inspect quality separately.</p>
    {error && <p className="error" role="alert">{error}</p>}
    {(exp.validations ?? []).map((batch) => { const s = batch.summary, st = s.stats; return <details key={batch.id} className="xv-batch" open>
      <summary>Validation #{batch.id} · {s.metric_label} · {conclusions[s.conclusion]}</summary>
      <p>Reference: #{batch.plan.sources[0].id} · Candidate: #{batch.plan.sources[1].id}. Completed pairs: {st.n}/{batch.plan.pairs}. Minimum improvement: {batch.plan.min_effect_pct}%.</p>
      <p>Median paired improvement: {fmt(st.median_pct)}%. Between-pair SD: {fmt(st.sd_pct)} percentage points. 95% interval: {st.ci95_pct ? `${fmt(st.ci95_pct[0])}% to ${fmt(st.ci95_pct[1])}%` : "Unbounded / unavailable"}. Exact sign-test p: {st.p == null ? "—" : st.p.toPrecision(3)}.</p>
      <div className="table-wrap"><table className="data"><thead><tr><th>Pair</th><th>A ({s.unit})</th><th>B ({s.unit})</th><th>Improvement</th><th>State A / B</th></tr></thead><tbody>{s.pairs.map((p) => <tr key={p.pair}><td>{p.pair}</td><td>{fmt(p.reference)}</td><td>{fmt(p.candidate)}</td><td>{fmt(p.improvement_pct)}%</td><td>{p.states.join(" / ")}</td></tr>)}</tbody></table></div>
      {s.problems.map((problem) => <p className="error" key={problem}>{problem}</p>)}<p className="row-sub">{s.note}</p>
      {s.conclusion === "pending" && <button className="btn small ghost" onClick={async () => { try { await api.validationCancel(exp.id, batch.id); onChanged(); } catch (e) { setError((e as Error).message); } }}>Cancel this validation</button>}
    </details>; })}
  </div>;
}
