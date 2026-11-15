import { useEffect, useState } from "react";
import { api, ResearchComparison, ResearchJob, ResearchProject } from "./api";

export default function ResearchPage() {
  const [projects, setProjects] = useState<ResearchProject[]>([]), [pid, setPid] = useState(0);
  const [name, setName] = useState("New research project"), [template, setTemplate] = useState("byte");
  const [files, setFiles] = useState<string[]>([]), [file, setFile] = useState("main.py"), [code, setCode] = useState("");
  const [python, setPython] = useState(""), [argv, setArgv] = useState('["{python}", "main.py"]'), [config, setConfig] = useState("{}");
  const [gpu, setGpu] = useState(false), [timeout, setTimeoutValue] = useState(3600), [parent, setParent] = useState<number | null>(null);
  const [jobs, setJobs] = useState<ResearchJob[]>([]), [selected, setSelected] = useState<number | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]), [comparison, setComparison] = useState<ResearchComparison | null>(null);
  const [profileRuns, setProfileRuns] = useState<{ id: number; label: string; experiment: string }[]>([]);
  const [error, setError] = useState(""), [message, setMessage] = useState(""), [saving, setSaving] = useState(false);
  const [methodName, setMethodName] = useState("");
  const refreshProjects = () => api.researchProjects().then((d) => { setProjects(d.projects); setPython((v) => v || d.python); setPid((v) => v || d.projects[0]?.id || 0); });
  useEffect(() => { refreshProjects().catch((e) => setError(String(e))); }, []);
  useEffect(() => { api.researchProfileRuns().then(setProfileRuns).catch((e) => setError(String(e))); }, []);
  useEffect(() => {
    if (!pid) return;
    let active = true;
    setJobs([]); setParent(null); setSelected(null); setMessage("");
    api.researchFiles(pid).then((f) => { if (active) setFiles(f); }).catch((e) => setError(String(e)));
    const poll = () => api.researchJobs(pid).then((d) => { if (active) setJobs(d); }).catch((e) => { if (active) setError(String(e)); });
    poll(); const timer = window.setInterval(poll, 2000);
    return () => { active = false; window.clearInterval(timer); };
  }, [pid]);
  useEffect(() => {
    if (!pid || !file) return;
    let active = true;
    api.researchRead(pid, file).then((d) => { if (active) setCode(d.content); }).catch((e) => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, [pid, file]);
  const act = async (fn: () => Promise<unknown>) => { setError(""); try { await fn(); } catch (e) { setError((e as Error).message); } };
  const save = async () => { await api.researchWrite(pid, file, code); setMessage("Source saved. Queued jobs keep their existing snapshot."); };
  const run = async () => {
    setSaving(true);
    try {
      await save();
      const result = await api.researchQueue(pid, { python, argv: JSON.parse(argv), config: JSON.parse(config), gpu, timeout_s: timeout, parent_id: parent });
      setSelected(result.id); setJobs(await api.researchJobs(pid)); setMessage(`Queued job #${result.id} with a source snapshot.`);
    } finally { setSaving(false); }
  };
  const shown = jobs.find((j) => j.id === selected) ?? jobs[0];
  return <div className="page"><div className="page-inner wide">
    <h1 className="page-title">Research projects</h1>
    <p className="page-sub">Run your own architectures, trainers and evaluators alongside the existing lab. Jobs share the hardware queue and report evidence through the Python SDK.</p>
    {error && <p className="error" role="alert">{error}</p>}{message && <p role="status">{message}</p>}
    <div className="group pad rp-controls">
      <label>Project<select className="field select" value={pid} onChange={(e) => setPid(Number(e.target.value))}><option value={0}>Choose a project</option>{projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select></label>
      <label>New project name<input className="field" value={name} onChange={(e) => setName(e.target.value)} /></label>
      <label>Starting code<select className="field select" value={template} onChange={(e) => setTemplate(e.target.value)}><option value="byte">Byte-bigram baseline (CPU smoke test)</option><option value="transformer">Conventional Transformers evaluator</option><option value="blank">Custom architecture / trainer</option></select></label>
      <button className="btn" onClick={() => act(async () => { const p = await api.researchCreate(name, template); await refreshProjects(); setFile("main.py"); setPid(p.id); })}>Create project</button>
    </div>
    {!!pid && <>
      <p className="row-sub">Workspace: <code>{projects.find((p) => p.id === pid)?.path}</code>. Edit here or use your local editor. Keep large datasets and checkpoints outside source; pass their paths in config. Included text is a tiny smoke test, not a benchmark.</p>
      <div className="group pad">
        <div className="rp-controls"><label>File<select className="field select" value={file} onChange={(e) => setFile(e.target.value)}>{files.map((f) => <option key={f}>{f}</option>)}</select></label><button className="btn small" onClick={() => act(save)}>Save file</button></div>
        <textarea aria-label="Research source code" className="field rp-code" spellCheck={false} value={code} onChange={(e) => setCode(e.target.value)} />
        <div className="rp-controls">
          <label>Python interpreter<input className="field" value={python} onChange={(e) => setPython(e.target.value)} /></label>
          <label>Command arguments (JSON array)<input className="field" value={argv} onChange={(e) => setArgv(e.target.value)} /></label>
          <label>Timeout (seconds)<input className="field" type="number" min={1} value={timeout} onChange={(e) => setTimeoutValue(Number(e.target.value))} /></label>
          <label><input type="checkbox" checked={gpu} onChange={(e) => setGpu(e.target.checked)} /> Reserve GPU and pause chat</label>
        </div>
        <label>Job configuration (JSON)<textarea aria-label="Research job configuration" className="field rp-config" value={config} onChange={(e) => setConfig(e.target.value)} /></label>
        <p className="row-sub">Dependencies come from your chosen environment; no packages are installed automatically. Code runs with the Blocky account’s permissions. Common model libraries use offline mode; this is not a network or filesystem sandbox.</p>
        <button className="btn" disabled={saving} onClick={() => act(run)}>Save and queue research job{parent ? ` (branch of #${parent})` : ""}</button>{parent && <button className="btn ghost" onClick={() => setParent(null)}>Clear parent</button>}
      </div>
      <div className="group-label">Jobs and evidence</div><div className="group pad">
        {jobs.length === 0 && <p>No jobs yet.</p>}
        {jobs.map((j) => <div className="rp-controls" key={j.id}><label><input type="checkbox" aria-label={`Compare research job ${j.id}`} checked={compareIds.includes(String(j.id))} onChange={(e) => setCompareIds((ids) => e.target.checked ? [...ids, String(j.id)] : ids.filter((id) => id !== String(j.id)))} /> Compare</label><button className="btn small ghost" onClick={() => setSelected(j.id)}>#{j.id} · {j.status}{j.parent_id ? ` · from #${j.parent_id}` : ""}</button><button className="btn small ghost" onClick={() => { setParent(j.id); setArgv(JSON.stringify(j.spec.argv)); setConfig(JSON.stringify(j.spec.config, null, 2)); setPython(j.spec.python); setGpu(j.spec.gpu); }}>Branch settings</button>{["queued", "running"].includes(j.status) && <button className="btn small ghost" onClick={() => act(async () => { await api.researchCancel(j.id); setJobs(await api.researchJobs(pid)); })}>Cancel #{j.id}</button>}</div>)}
        {shown && <><h3>Job #{shown.id} · {shown.status}</h3>{shown.error && <p className="error">{shown.error}</p>}
          {shown.status === "done" && <div className="rp-controls"><label>Registered method name<input className="field" value={methodName} onChange={(e) => setMethodName(e.target.value)} placeholder="My architecture or evaluator" /></label><button className="btn" onClick={() => act(async () => { const m = await api.researchRegister(shown.id, methodName || projects.find((p) => p.id === pid)?.name || ""); setMessage(`Registered ${m.name}. Open Experiments and choose it as a baseline or evaluation branch.`); })}>Register for Experiments</button><a href="#/experiments">Open Experiments</a></div>}
          <p className="row-sub">{shown.directory}</p>
          {shown.events.filter((e) => e.type === "metric").map((m, i) => <details key={i}><summary>{m.name}: {m.value} {m.unit} · {m.kind} · {m.spread?.sd != null ? `sample SD ${m.spread.sd.toPrecision(3)} (n=${m.spread.n})` : "uncertainty not reported"} · reported by project evaluator</summary><pre className="rp-output">{JSON.stringify(m, null, 2)}</pre></details>)}
          {shown.events.filter((e) => e.type === "artifact").map((e, i) => <p key={i}><a href={`/api/research/jobs/${shown.id}/artifact?name=${encodeURIComponent(e.path ?? "")}`}>Download {e.path}</a></p>)}
          <p>{shown.events.filter((e) => e.type === "progress").slice(-1)[0]?.message}</p>
          <details open><summary>Live output (last 32 KB)</summary><pre className="rp-output">{shown.log || "No output yet."}</pre></details>
          <details><summary>Source hashes, command and environment</summary><pre className="rp-output">{JSON.stringify({ spec: shown.spec, provenance: shown.provenance }, null, 2)}</pre><a href={`/api/research/jobs/${shown.id}/artifact?name=environment.txt`}>Download package versions</a></details>
        </>}
      </div>
    </>}
    <div className="group pad"><h3>Compare methods across projects</h3><p className="row-sub">Select jobs in any project. Matching dataset fingerprints, task, unit, protocol and measurement scope permit side-by-side interpretation. Use bytes or common task scores for different tokenizers.</p>
      <label>Add an existing experiment run<select className="field select" value="" onChange={(e) => { if (e.target.value) setCompareIds((ids) => Array.from(new Set([...ids, e.target.value]))); }}><option value="">Choose a completed profile</option>{profileRuns.map((r) => <option key={r.id} value={`profile:${r.id}`}>#{r.id} {r.experiment}: {r.label}</option>)}</select></label><p className="row-sub">Profile scores retain their original protocols. Missing shared metrics stay unavailable; token rates are not converted to byte rates.</p><p>Selected jobs: {compareIds.join(", ") || "none"}</p><button className="btn" disabled={compareIds.length < 2 || compareIds.length > 8} onClick={() => act(async () => setComparison(await api.researchCompare(compareIds)))}>Compare selected jobs</button><button className="btn ghost" onClick={() => { setCompareIds([]); setComparison(null); }}>Clear selection</button>
      {comparison && <><div className="table-wrap"><table className="data"><thead><tr><th>Metric</th>{comparison.runs.map((r) => <th key={r.id}>Job #{r.id}</th>)}<th>Compatibility</th></tr></thead><tbody>{comparison.metrics.map((m) => <tr key={m.name}><td>{m.name}</td>{m.values.map((v, i) => <td key={i}>{v ? `${v.value} ${v.unit}` : "Unavailable"}</td>)}<td>{m.compatible ? "Matching declared contract" : m.reasons.join("; ")}</td></tr>)}</tbody></table></div><p>{comparison.note}</p></>}
    </div>
  </div></div>;
}
