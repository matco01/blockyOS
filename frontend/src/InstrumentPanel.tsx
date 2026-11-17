import { useEffect, useState } from "react";
import { api, InstrumentReport } from "./api";

const LABEL: Record<string, { text: string; tone: string }> = {
  pass: { text: "Pass", tone: "good" }, warn: { text: "Check", tone: "meh" }, fail: { text: "Fail", tone: "bad" }, not_run: { text: "Not run yet", tone: "meh" },
};

/** Is the measurement pipeline behaving like a correct instrument on this machine? Computed from runs the experiment already has. */
export default function InstrumentPanel({ expId, runsKey }: { expId: number; runsKey: string }) {
  const [report, setReport] = useState<InstrumentReport | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api.instrumentCheck(expId).then(setReport, (e) => setError((e as Error).message));
  }, [expId, runsKey]);
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
            <div className="row-sub">{c.detail}{c.hint ? ` ${c.hint}` : ""}{c.todo ? <> <em>To run it: {c.todo}</em></> : null}
              {c.runs?.length ? <> (runs {c.runs.map((r) => `#${r}`).join(", ")})</> : null}</div>
            {c.matches?.length ? <ul>{c.matches.map((m, i) => (
              <li key={i} className="row-sub">#{m.run} {m.metric}: measured {m.measured.toFixed(4)}, reference {m.reference} ± {m.tolerance} ({m.source}) — {m.ok ? "within tolerance" : "outside tolerance"}</li>
            ))}</ul> : null}
          </li>
        ))}
      </ul>
    </details>
  );
}
