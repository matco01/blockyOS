import { useEffect, useRef, useState } from "react";
import { api, ExpRun } from "./api";
import { evidence } from "./evidence";
import { Metric, n1, runTitle } from "./expmetrics";

export default function MetricEvidence({ metric, run, onClose }: { metric: Metric; run: ExpRun; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [outputs, setOutputs] = useState<Record<string, { text: string; truncated: boolean }>>({});
  const [outputNote, setOutputNote] = useState("Loading retained evaluator output…");
  useEffect(() => { dialog.current?.showModal(); }, []);
  useEffect(() => { let active = true; api.experimentEvidence(run.id).then((d) => { if (active) { setOutputs(d.outputs); setOutputNote(d.note); } }).catch(() => { if (active) setOutputNote("Evaluator output is unavailable from this backend."); }); return () => { active = false; }; }, [run.id]);
  const e = evidence(metric, run), v = metric.get(run.results);
  return <dialog ref={dialog} className="xevidence" onCancel={onClose} onClick={(ev) => { if (ev.target === ev.currentTarget) onClose(); }}>
    <div className="xe-head"><h2>{metric.label}</h2><button className="btn small ghost" onClick={onClose}>Close evidence</button></div>
    <p>{runTitle(run)} · <b>{v == null ? "Unavailable" : `${(metric.fmt ?? n1)(v)} ${metric.unit}`}</b> <span className="xe-badge">{e.kind}</span></p>
    <p>{e.reason}</p><p><b>Source:</b> {e.source}</p>{metric.note && <p>{metric.note}</p>}
    <p><b>Protocol:</b> {(metric.researchName ? run.results.research_metrics?.[metric.researchName]?.protocol : run.results.provenance?.protocol) ?? "Not recorded"}</p>
    {run.results.research_job_id && <p><a href={`/api/research/jobs/${run.results.research_job_id}/artifact?name=stdout.log`}>Download full evaluator log</a> · <a href={`/api/research/jobs/${run.results.research_job_id}/artifact?name=environment.txt`}>Download environment</a></p>}
    {run.results.research_job_id && (run.results.research_artifacts ?? []).map((a, i) => <p key={i}><a href={`/api/research/jobs/${run.results.research_job_id}/artifact?name=${encodeURIComponent(a.path ?? "")}`}>Download {a.path}</a></p>)}
    <details open><summary>Samples and supporting data</summary><pre>{e.raw == null ? "Raw samples were not retained for this result." : JSON.stringify(e.raw, null, 2)}</pre></details>
    <details><summary>Commands</summary><pre>{Object.entries(run.results.commands ?? {}).map(([k, v]) => `# ${k}\n${v}`).join("\n\n") || "No commands retained for this run."}</pre></details>
    <details><summary>Run log</summary><pre>{run.log || "No run log retained."}</pre><p>Retained log tail; evaluator summaries may not include every original output line.</p></details>
    <details><summary>Retained evaluator output</summary><p>{outputNote}</p>{Object.entries(outputs).map(([phase, data]) => <section key={phase}><b>{phase}{data.truncated ? " (truncated at 1 MB)" : ""}</b><pre>{data.text}</pre></section>)}</details>
  </dialog>;
}
