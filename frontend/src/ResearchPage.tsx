import { useEffect, useRef, useState } from "react";
import { api, ResearchComparison, ResearchJob, ResearchProject } from "./api";
import { Tabs, useMode } from "./ui";
import ConfigFields from "./ConfigFields";
import { configObject } from "./workflow";
import { FilePick, ProgressBar, uploadCode } from "./Upload";

/** The script a run starts, read from the command (`["{python}", "train.py", ...]`). */
const entryOf = (argv: string) => { try { const a = JSON.parse(argv); return Array.isArray(a) && typeof a[1] === "string" ? a[1] : ""; } catch { return ""; } };

/** Your own code: programs, evaluators, architectures. Inside a project (`codeIds`), it shows only that project's code
 *  and hands every new workspace to `onCreate` so the project can claim it. */
export default function ResearchPage({ codeIds, onCreate }: { codeIds?: number[]; onCreate?: (id: number) => Promise<unknown> } = {}) {
  const mode = useMode();
  const [created, setCreated] = useState<number[]>([]);
  const pageRef = useRef<HTMLDivElement>(null);
  const [stage, setStage] = useState<"prepare" | "code" | "run" | "results" | "compare">("prepare");
  const [loading, setLoading] = useState(true), [jobsLoading, setJobsLoading] = useState(false), [sourceLoading, setSourceLoading] = useState(false);
  const [projectsReady, setProjectsReady] = useState(false);
  const [creating, setCreating] = useState(false);
  useEffect(() => { pageRef.current?.scrollTo({ top: 0 }); }, [stage]);
  const [environments, setEnvironments] = useState<{ label: string; python: string }[]>([]);
  const [savedCode, setSavedCode] = useState("");
  const [sourceReady, setSourceReady] = useState(false);
  const [jobsReady, setJobsReady] = useState(false);
  const [sourceRevision, setSourceRevision] = useState(0);
  const [allProjects, setProjects] = useState<ResearchProject[]>([]), [pid, setPid] = useState(0);
  const projects = codeIds ? allProjects.filter((p) => codeIds.includes(p.id) || created.includes(p.id)) : allProjects;
  useEffect(() => { if (codeIds && pid && !projects.some((p) => p.id === pid)) setPid(projects[0]?.id ?? 0); }, [codeIds, pid, projects]);
  const [name, setName] = useState(codeIds ? "My code" : "New research project"), [template, setTemplate] = useState(codeIds ? "import" : "byte");
  const [importFiles, setImportFiles] = useState<File[]>([]);
  const [progress, setProgress] = useState<{ value: number; label: string } | null>(null);
  const [files, setFiles] = useState<string[]>([]), [file, setFile] = useState("main.py"), [code, setCode] = useState("");
  const [python, setPython] = useState(""), [argv, setArgv] = useState('["{python}", "main.py"]'), [config, setConfig] = useState("{}");
  const [gpu, setGpu] = useState(false), [timeout, setTimeoutValue] = useState(3600), [parent, setParent] = useState<number | null>(null);
  const [jobs, setJobs] = useState<ResearchJob[]>([]), [selected, setSelected] = useState<number | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]), [comparison, setComparison] = useState<ResearchComparison | null>(null);
  const [profileRuns, setProfileRuns] = useState<{ id: number; label: string; experiment: string }[]>([]);
  const [error, setError] = useState(""), [message, setMessage] = useState(""), [saving, setSaving] = useState(false);
  const [methodName, setMethodName] = useState("");
  const refreshProjects = () => api.researchProjects().then((d) => { setProjects(d.projects); setProjectsReady(true); setEnvironments(d.environments ?? [{ label: "Standard Python", python: d.python }]); setPython((v) => v || d.python); setPid((v) => v || (codeIds ? d.projects.find((p) => codeIds.includes(p.id))?.id : d.projects[0]?.id) || 0); }).finally(() => setLoading(false));
  useEffect(() => { refreshProjects().catch((e) => setError(String(e))); }, []);
  useEffect(() => { api.researchProfileRuns().then(setProfileRuns).catch((e) => setError(String(e))); }, []);
  useEffect(() => {
    if (!pid) return;
    let active = true;
    setJobs([]); setParent(null); setSelected(null); setMessage(""); setJobsLoading(true); setJobsReady(false);
    api.researchFiles(pid).then((f) => { if (active) setFiles(f); }).catch((e) => setError(String(e)));
    const poll = () => api.researchJobs(pid).then((d) => { if (active) { setJobs(d); setJobsLoading(false); setJobsReady(true); } }).catch((e) => { if (active) { setError(String(e)); setJobsLoading(false); } });
    poll(); const timer = window.setInterval(poll, 2000);
    return () => { active = false; window.clearInterval(timer); };
  }, [pid]);
  useEffect(() => {
    if (!pid || !file) return;
    let active = true; setSourceLoading(true); setSourceReady(false);
    api.researchRead(pid, file).then((d) => { if (active) {
      let draft: string | null = null;
      try { draft = sessionStorage.getItem(`blocky.source.${pid}.${file}`); } catch { /* storage may be disabled */ }
      setCode(draft ?? d.content); setSavedCode(d.content); setSourceReady(true);
      if (draft !== null && draft !== d.content) setMessage("Restored your unsaved draft. Review it before saving or running.");
    } }).catch((e) => { if (active) setError(String(e)); }).finally(() => { if (active) setSourceLoading(false); });
    return () => { active = false; };
  }, [pid, file, sourceRevision]);
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => { if (code !== savedCode) { event.preventDefault(); event.returnValue = ""; } };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [code, savedCode]);
  const act = async (fn: () => Promise<unknown>) => { setError(""); setSaving(true); try { await fn(); } catch (e) { setError((e as Error).message); } finally { setSaving(false); } };
  const save = async () => { await api.researchWrite(pid, file, code); setSavedCode(code); try { sessionStorage.removeItem(`blocky.source.${pid}.${file}`); } catch { /* storage may be disabled */ } setMessage("Source saved. Queued jobs keep their existing snapshot."); };
  const run = async () => {
    setSaving(true);
    try {
      const parsed = configObject(config);
      await save();
      const result = await api.researchQueue(pid, { python, argv: JSON.parse(argv), config: parsed, gpu, timeout_s: timeout, parent_id: parent });
      setSelected(result.id); setJobs(await api.researchJobs(pid)); setStage("results"); setMessage(`Run #${result.id} queued. Follow its progress below.`);
    } finally { setSaving(false); }
  };
  const shown = jobs.find((j) => j.id === selected) ?? jobs[0];
  return <div className={codeIds ? "embedded" : "page"} ref={pageRef}><div className={codeIds ? "" : "page-inner wide"}>
    {!codeIds && <>
      <h1 className="page-title">Develop a method</h1>
      <p className="page-sub">Build your implementation, run an evaluation, then use the tested result in Experiments.</p>
    </>}
    {error && <p className="error" role="alert">{error} <button className="btn small ghost" onClick={() => { setError(""); setLoading(true); refreshProjects().catch((e) => setError(String(e))); }}>Retry loading projects</button></p>}{message && <p role="status">{message}</p>}
    {loading ? <p role="status">Loading projects…</p> : <div className="flow-toolbar"><label>Workspace<select className="field select" value={pid} onChange={(e) => { if (code === savedCode || confirm("Discard unsaved source edits?")) { try { sessionStorage.removeItem(`blocky.source.${pid}.${file}`); } catch { /* storage may be disabled */ } setPid(Number(e.target.value)); setConfig("{}"); setParent(null); setMethodName(""); setFile("main.py"); setStage("prepare"); } }}><option value={0}>Choose a project</option>{projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select></label><button className="btn" onClick={() => setCreating(!creating)}>New project</button></div>}
    {!loading && projectsReady && (creating || projects.length === 0) && <div className="group pad flow-create"><h2>Where is your code?</h2><p>Upload the code you already have, or start from an example.</p><div className="rp-controls">
      <label>Name<input className="field" value={name} onChange={(e) => setName(e.target.value)} /></label>
      <label>Starting code<select className="field select" value={template} onChange={(e) => setTemplate(e.target.value)}><option value="import">My own code (upload a folder or .zip)</option><option value="byte">Example: byte-bigram baseline (CPU smoke test)</option><option value="transformer">Example: Transformers evaluator</option><option value="blank">Empty custom architecture / trainer</option></select></label>
    </div>
      {template === "import" && <div className="rp-import">
        <div className="flow-toolbar">
          <FilePick label="Choose a folder" folder ghost={importFiles.length > 0} onFiles={setImportFiles} />
          <FilePick label="Choose a .zip" accept=".zip" ghost onFiles={setImportFiles} />
          {importFiles.length > 0 && <span className="row-sub">{importFiles.length === 1 ? importFiles[0].name : `${importFiles.length} files selected`}</span>}
        </div>
        <p className="row-sub">Your files are copied onto this device. Environments and caches (<code>.git</code>, <code>.venv</code>, <code>__pycache__</code>, <code>node_modules</code>) are left out. A .zip from GitHub (“Download ZIP”) works as it is.</p>
      </div>}
      {progress && <ProgressBar value={progress.value} label={progress.label} />}
      <button className="btn" disabled={saving || !name.trim() || (template === "import" && importFiles.length === 0)} onClick={() => act(async () => {
        const p = await api.researchCreate(name, template); setCreated((c) => [...c, p.id]); await onCreate?.(p.id);
        let entry = "main.py";
        if (template === "import") {
          const entries = await uploadCode(p.id, importFiles, (value, label) => setProgress({ value, label }));
          entry = entries[0] ?? ""; setImportFiles([]); setProgress(null);
          setArgv(JSON.stringify(["{python}", entry || "main.py"]));
          setMessage(entries.length ? `Code uploaded. It starts with ${entry}; change that under Run if needed.` : "Code uploaded. No Python file was found; set the command under Run → Advanced.");
        }
        await refreshProjects(); setFile(entry); setPid(p.id); setCreating(false); setStage(template === "import" ? "run" : "prepare");
        setConfig(JSON.stringify(template === "byte" ? { smoothing: 1, reps: 3 } : template === "transformer" ? { checkpoint: "", eval: "eval.txt", reps: 3 } : {}, null, 2));
      })}>{template === "import" ? "Upload and continue" : "Create"}</button>
      {template !== "import" && <p className="row-sub">The byte example runs on CPU without downloads and uses tiny smoke-test data. It is not BLT. The Transformers starter requires a local checkpoint and an ML environment.</p>}</div>}
    {!!pid && <>
      <Tabs value={stage} onChange={setStage} tabs={[{ id: "prepare", label: "1 · Prepare" }, { id: "code", label: "2 · Implementation" }, { id: "run", label: "3 · Run" }, { id: "results", label: "4 · Results", badge: jobs.length }, { id: "compare", label: "Compare methods", badge: compareIds.length }]} />
      {stage === "prepare" && <div className="group pad flow-next"><h2>{projects.find((p) => p.id === pid)?.name}</h2><ol><li>Prepare your implementation and held-out evaluation data.</li><li>Choose an environment and run settings.</li><li>Inspect the result and continue in Experiments.</li></ol><div className="flow-toolbar"><button className="btn" onClick={() => setStage("run")}>Configure a run</button><button className="btn ghost" onClick={() => setStage("code")}>Edit implementation and data</button>{jobs.length > 0 && <button className="btn ghost" onClick={() => setStage("results")}>Review previous runs</button>}</div><details open={mode === "pro" || undefined}><summary>Workspace and dependency setup</summary><p><code>{projects.find((p) => p.id === pid)?.path}</code></p><p>Edit here or in your local editor. Keep large datasets and checkpoints outside source; pass their paths as settings. Included text is a tiny smoke test, not a benchmark.</p></details></div>}
      <div className="group pad" hidden={stage !== "code" && stage !== "run"}>
        <section hidden={stage !== "code"}><div className="rp-controls"><label>File<select className="field select" value={file} onChange={(e) => { if (code === savedCode || confirm("Discard unsaved source edits?")) { try { sessionStorage.removeItem(`blocky.source.${pid}.${file}`); } catch { /* storage may be disabled */ } setFile(e.target.value); } }}>{files.map((f) => <option key={f}>{f}</option>)}</select></label><button className="btn small" disabled={sourceLoading || !sourceReady || saving} onClick={() => act(save)}>Save file</button><button className="btn ghost small" onClick={() => setStage("run")}>Continue to run settings</button>
          <FilePick label="Upload a folder" folder ghost disabled={saving} onFiles={(f) => act(async () => { await uploadCode(pid, f, (value, label) => setProgress({ value, label })); setProgress(null); setFiles(await api.researchFiles(pid)); setSourceRevision((v) => v + 1); setMessage("Files uploaded. Files with the same name were replaced."); })} />
          <FilePick label="Upload a .zip" accept=".zip" ghost disabled={saving} onFiles={(f) => act(async () => { await uploadCode(pid, f, (value, label) => setProgress({ value, label })); setProgress(null); setFiles(await api.researchFiles(pid)); setSourceRevision((v) => v + 1); setMessage("Archive unpacked. Files with the same name were replaced."); })} /></div>
        {progress && <ProgressBar value={progress.value} label={progress.label} />}
        {sourceLoading ? <p role="status">Loading source…</p> : !sourceReady ? <p>Source unavailable. <button className="btn small ghost" onClick={() => setSourceRevision((v) => v + 1)}>Retry loading source</button></p> : <textarea aria-label="Research source code" className="field rp-code" spellCheck={false} value={code} onChange={(e) => { setCode(e.target.value); try { sessionStorage.setItem(`blocky.source.${pid}.${file}`, e.target.value); } catch { /* storage may be disabled */ } }} />}<p className="row-sub">{!sourceReady ? "" : code === savedCode ? "Source saved." : "Unsaved edits (a draft is kept in this browser tab). Starting a run saves this file first."}</p></section>
        <section hidden={stage !== "run"}><h2>{parent ? `Branch from run #${parent}` : "Configure a run"}</h2><ConfigFields value={config} onChange={setConfig} />
        <div className="flow-fields"><label>Start file<select className="field select" value={entryOf(argv)} onChange={(e) => { try { const a = JSON.parse(argv); setArgv(JSON.stringify(Array.isArray(a) && a.length >= 2 ? [a[0], e.target.value, ...a.slice(2)] : ["{python}", e.target.value])); } catch { setArgv(JSON.stringify(["{python}", e.target.value])); } }}>
          {!files.includes(entryOf(argv)) && <option value={entryOf(argv)}>{entryOf(argv) || "Choose the file that starts your program"}</option>}
          {files.filter((f) => f.endsWith(".py")).map((f) => <option key={f} value={f}>{f}</option>)}</select></label>
          <label>Environment<select className="field select" value={environments.some((e) => e.python === python) ? python : "custom"} onChange={(e) => { if (e.target.value === "custom") setPython(""); else setPython(e.target.value); }}>{environments.map((e) => <option key={e.python} value={e.python}>{e.label}</option>)}<option value="custom">Custom environment (set path below)</option></select></label><label className="flow-check"><input type="checkbox" checked={gpu} onChange={(e) => setGpu(e.target.checked)} /> Reserve GPU and pause chat</label></div>
        <details className="flow-advanced" open={mode === "pro" || !python || undefined}><summary>Advanced execution and JSON settings</summary><div className="rp-controls">
          <label>Python interpreter<input className="field" value={python} onChange={(e) => setPython(e.target.value)} /></label>
          <label>Command arguments (JSON array)<input className="field" value={argv} onChange={(e) => setArgv(e.target.value)} /></label>
          <label>Timeout (seconds)<input className="field" type="number" min={1} value={timeout} onChange={(e) => setTimeoutValue(Number(e.target.value))} /></label>
        </div>
        <label>Job configuration (JSON)<textarea aria-label="Research job configuration" className="field rp-config" value={config} onChange={(e) => setConfig(e.target.value)} /></label></details>
        <p className="row-sub">Dependencies come from your chosen environment; no packages are installed automatically. Code runs with the Blocky account’s permissions. Common model libraries use offline mode; this is not a network or filesystem sandbox.</p>
        <button className="btn" disabled={saving || sourceLoading || !sourceReady || !python} onClick={() => act(run)}>Start research run{parent ? ` (branch of #${parent})` : ""}</button>{parent && <button className="btn ghost" onClick={() => setParent(null)}>Clear parent</button>}</section>
      </div>
      <div className="group pad" hidden={stage !== "results" && stage !== "compare"}><h2>Runs and evidence</h2>
        {jobsLoading ? <p role="status">Loading runs…</p> : !jobsReady ? <p>Run list unavailable. Retrying automatically.</p> : jobs.length === 0 && <p>No runs yet. Start an evaluation to record results.</p>}
        {jobs.map((j) => <div className="rp-controls" key={j.id}><label><input type="checkbox" aria-label={`Compare research job ${j.id}`} checked={compareIds.includes(String(j.id))} onChange={(e) => setCompareIds((ids) => e.target.checked ? [...ids, String(j.id)] : ids.filter((id) => id !== String(j.id)))} /> Compare</label><button className="btn small ghost" onClick={() => { setSelected(j.id); setStage("results"); }}>#{j.id} · {j.status}{j.parent_id ? ` · from #${j.parent_id}` : ""}</button><button className="btn small ghost" onClick={() => { setStage("run"); setParent(j.id); setArgv(JSON.stringify(j.spec.argv)); setConfig(JSON.stringify(j.spec.config, null, 2)); setPython(j.spec.python); setGpu(j.spec.gpu); }}>Branch settings</button>{["queued", "running"].includes(j.status) && <button className="btn small ghost" onClick={() => act(async () => { await api.researchCancel(j.id); setJobs(await api.researchJobs(pid)); })}>Cancel #{j.id}</button>}</div>)}
        {stage === "results" && shown && <><h3>Run #{shown.id} · {shown.status}</h3>{shown.error && <p className="error">{shown.error}</p>}
          {["queued", "running"].includes(shown.status) && <p role="status">{shown.status === "queued" ? "Waiting for the shared hardware queue. Your source and settings have been saved." : "Your program is running. Progress and output update automatically."}</p>}
          {shown.status === "done" && shown.events.some((e) => e.type === "metric") && <div className="flow-next"><h3>Ready to compare this method?</h3><p>Carry this tested result into Experiments as the baseline. Registration happens automatically; the measurement is not rerun.</p><button className="btn" disabled={saving} onClick={() => act(async () => { const result = await api.researchPromote(shown.id, methodName || projects.find((p) => p.id === pid)?.name || ""); window.location.hash = `#/experiments?exp=${result.experiment_id}&stage=compare`; })}>Use this result in Experiments</button></div>}
          {shown.status === "done" && !shown.events.some((e) => e.type === "metric") && <p>The program completed without evaluation metrics. Its artifacts are available below. Report an evaluation metric through the SDK to compare this method in Experiments.</p>}
          {shown.status === "done" && <details className="flow-advanced"><summary>Register separately (advanced)</summary><div className="rp-controls"><label>Registered method name<input className="field" value={methodName} onChange={(e) => setMethodName(e.target.value)} placeholder="My architecture or evaluator" /></label><button className="btn" disabled={saving} onClick={() => act(async () => { const m = await api.researchRegister(shown.id, methodName || projects.find((p) => p.id === pid)?.name || ""); setMessage(`Registered ${m.name}. Open Experiments and choose it as a baseline or evaluation branch.`); })}>Register for Experiments</button></div></details>}
          <p className="row-sub">{shown.directory}</p>
          {shown.events.filter((e) => e.type === "metric").map((m, i) => <details key={i}><summary>{m.name}: {m.value} {m.unit} · {m.kind} · {m.spread?.sd != null ? `sample SD ${m.spread.sd.toPrecision(3)} (n=${m.spread.n})` : "uncertainty not reported"} · reported by project evaluator</summary><pre className="rp-output">{JSON.stringify(m, null, 2)}</pre></details>)}
          {shown.events.filter((e) => e.type === "artifact").map((e, i) => <p key={i}><a href={`/api/research/jobs/${shown.id}/artifact?name=${encodeURIComponent(e.path ?? "")}`}>Download {e.path}</a></p>)}
          <p>{shown.events.filter((e) => e.type === "progress").slice(-1)[0]?.message}</p>
          <details open={shown.status === "running" || mode === "pro" || undefined}><summary>Live output (last 32 KB)</summary><pre className="rp-output">{shown.log || "No output yet."}</pre></details>
          <details open={mode === "pro" || undefined}><summary>Source hashes, command and environment</summary><pre className="rp-output">{JSON.stringify({ spec: shown.spec, provenance: shown.provenance }, null, 2)}</pre><a href={`/api/research/jobs/${shown.id}/artifact?name=environment.txt`}>Download package versions</a></details>
        </>}
      </div>
    </>}
    <div className="group pad" hidden={stage !== "compare"}><h3>Compare methods across projects</h3><p className="row-sub">Select runs in any project. Your selection stays when switching workspaces. Shared measurements must use matching tasks, data, units and procedures.</p>
      <label>Add an existing experiment run<select className="field select" value="" onChange={(e) => { if (e.target.value) setCompareIds((ids) => Array.from(new Set([...ids, e.target.value]))); }}><option value="">Choose a completed profile</option>{profileRuns.map((r) => <option key={r.id} value={`profile:${r.id}`}>#{r.id} {r.experiment}: {r.label}</option>)}</select></label><p className="row-sub">Profile scores retain their original protocols. Missing shared metrics stay unavailable; token rates are not converted to byte rates.</p><p>Selected jobs: {compareIds.join(", ") || "none"}</p><button className="btn" disabled={compareIds.length < 2 || compareIds.length > 8} onClick={() => act(async () => setComparison(await api.researchCompare(compareIds)))}>Compare selected jobs</button><button className="btn ghost" onClick={() => { setCompareIds([]); setComparison(null); }}>Clear selection</button>
      {comparison && <><div className="table-wrap"><table className="data"><thead><tr><th>Metric</th>{comparison.runs.map((r) => <th key={r.id}>Job #{r.id}</th>)}<th>Compatibility</th></tr></thead><tbody>{comparison.metrics.map((m) => <tr key={m.name}><td>{m.name}</td>{m.values.map((v, i) => <td key={i}>{v ? `${v.value} ${v.unit}` : "Unavailable"}</td>)}<td>{m.compatible ? "Matching declared contract" : m.reasons.join("; ")}</td></tr>)}</tbody></table></div><p>{comparison.note}</p></>}
    </div>
  </div></div>;
}
