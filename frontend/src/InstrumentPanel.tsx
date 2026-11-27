import { useEffect, useState } from "react";
import { api, ExpRun, InstrumentReport } from "./api";

const LABEL: Record<string, { text: string; tone: string }> = {
  pass: { text: "Pass", tone: "good" }, warn: { text: "Check", tone: "meh" }, fail: { text: "Fail", tone: "bad" }, not_run: { text: "Not run yet", tone: "meh" },
};

const METRICS: { id: string; label: string }[] = [
  { id: "ppl", label: "Perplexity" }, { id: "kld", label: "Mean KL divergence" }, { id: "same_top", label: "Same top token (%)" },
  { id: "mmlu_acc", label: "MMLU-Pro accuracy (0–1)" }, { id: "tg", label: "Decode speed (tok/s, same hardware only)" }, { id: "pp", label: "Prefill speed (tok/s, same hardware only)" },
];

/** Records a value published for the exact model file a run used, so the cited-values check can compare with it. */
function RecordReference({ runs, onSaved }: { runs: ExpRun[]; onSaved: () => void }) {
  const files = runs.filter((r, i, all) => r.results.provenance?.model?.sha256 && all.findIndex((x) => x.results.provenance?.model?.sha256 === r.results.provenance?.model?.sha256) === i);
  const [run, setRun] = useState(files[0]?.id ?? 0), [metric, setMetric] = useState("ppl");
  const [value, setValue] = useState(""), [tol, setTol] = useState(""), [source, setSource] = useState(""), [err, setErr] = useState("");
  const sha = files.find((r) => r.id === run)?.results.provenance?.model?.sha256 ?? "";
  if (!files.length) return null;
  const save = () => {
    setErr("");
    api.referenceAdd({ model_sha256: sha, metric, value: Number(value), tolerance: Number(tol), source })
      .then(() => { setValue(""); setTol(""); setSource(""); onSaved(); }).catch((e) => setErr((e as Error).message));
  };
  return (
    <details className="ref-form">
      <summary>Record a published value</summary>
      <p className="row-sub">A value counts only for the byte-identical model file (same SHA-256) and comparable settings (text, context, chunk count). Set the tolerance from the source's own uncertainty.</p>
      <div className="ws-grid">
        <label><span>Model file</span><select className="field select" value={run} onChange={(e) => setRun(Number(e.target.value))}>
          {files.map((r) => <option key={r.id} value={r.id}>{r.config.model} · {r.label}</option>)}</select>
          <em className="mono">{sha.slice(0, 16)}…</em></label>
        <label><span>Metric</span><select className="field select" value={metric} onChange={(e) => setMetric(e.target.value)}>
          {METRICS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}</select></label>
        <label><span>Published value</span><input className="field" type="number" step="any" value={value} onChange={(e) => setValue(e.target.value)} /></label>
        <label><span>Tolerance (±)</span><input className="field" type="number" step="any" min={0} value={tol} onChange={(e) => setTol(e.target.value)} /></label>
        <label className="ref-source"><span>Source</span><input className="field" value={source} placeholder="Author, title, table or figure, link" onChange={(e) => setSource(e.target.value)} /></label>
      </div>
      <div className="xc-btns"><button className="btn small" disabled={!value || !tol || source.trim().length < 15} onClick={save}>Record</button></div>
      {err && <div className="error">{err}</div>}
    </details>
  );
}

/** Is the measurement pipeline behaving like a correct instrument on this machine? Computed from runs the experiment already has. */
export default function InstrumentPanel({ expId, runsKey, runs = [] }: { expId: number; runsKey: string; runs?: ExpRun[] }) {
  const [report, setReport] = useState<InstrumentReport | null>(null);
  const [error, setError] = useState("");
  const [rev, setRev] = useState(0);
  useEffect(() => {
    api.instrumentCheck(expId).then(setReport, (e) => setError((e as Error).message));
  }, [expId, runsKey, rev]);
  if (error) return <div className="group pad"><div className="error">{error}</div></div>;
  if (!report || !report.checks.length) return null;
  return (
    <details className="group pad xadv">
      <summary><b>Instrument checks</b> <span className="row-sub">does the measurement pipeline behave correctly on this machine?
        {report.counts && <> {report.counts.pass} pass · {report.counts.warn} to check · {report.counts.fail} fail · {report.counts.not_run} not run</>}</span></summary>
      <div className="row-sub">{report.note}</div>
      <ul className="xinst">
        {report.checks.map((c) => (
          <li key={c.id}>
            <span className={`verdict tone-${LABEL[c.status].tone}`}>{LABEL[c.status].text}</span> <b>{c.title}</b>
            <div className="row-sub">{c.detail}{c.hint ? ` ${c.hint}` : ""}{c.todo && c.id !== "reference_values" ? <> <em>To run it: {c.todo}</em></> : null}
              {c.runs?.length ? <> (runs {c.runs.map((r) => `#${r}`).join(", ")})</> : null}</div>
            {c.matches?.length ? <ul>{c.matches.map((m, i) => (
              <li key={i} className="row-sub">#{m.run} {m.metric}: measured {m.measured.toFixed(4)}, reference {m.reference} ± {m.tolerance} ({m.source}) — {m.ok ? "within tolerance" : "outside tolerance"}</li>
            ))}</ul> : null}
            {c.id === "reference_values" && <RecordReference runs={runs} onSaved={() => setRev((v) => v + 1)} />}
          </li>
        ))}
      </ul>
    </details>
  );
}
