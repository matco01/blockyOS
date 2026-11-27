/* Projects: the front door. A project is what you are making; everything you do happens inside one. */
import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Boxes, Brain, Code, Database, Download, FileText, FolderKanban, GitBranch, Hammer, Layers, Pencil, Play, Plus, Rocket, Sparkles, Trash2, Wand2 } from "lucide-react";
import { api, Dataset, ExpRun, ModelFile, Project, ProjectCode, ProjectExperiment, ProjectResults, Status } from "./api";
import { DropZone, FilePick, fmtBytes, ProgressBar } from "./Upload";
import ExperimentsPage from "./ExperimentsPage";
import InternalsPage from "./InternalsPage";
import ResearchPage from "./ResearchPage";
import WeightsWorkshop from "./WeightsWorkshop";
import { Tabs } from "./ui";
import { modelTitle } from "./workflow";

type Tab = "overview" | "build" | "data" | "canvas" | "tests" | "ship";
const TABS: Tab[] = ["overview", "build", "data", "canvas", "tests", "ship"];
type Recipe = "finetune" | "train" | "smaller" | "code" | "method";

function ago(t: number) {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  const d = Math.floor(s / 86400);
  return d === 1 ? "yesterday" : `${d} days ago`;
}

/** The two or three numbers that say how a version is doing. */
function KeyNumbers({ r }: { r: ProjectResults | null | undefined }) {
  if (!r) return <span className="muted">not measured yet</span>;
  return (
    <span className="pj-nums">
      {r.mmlu != null && <span><b>{(r.mmlu * 100).toFixed(1)}%</b> accuracy</span>}
      {r.tg != null && <span><b>{r.tg.toFixed(0)}</b> tok/s</span>}
      {r.ppl != null && <span><b>{r.ppl.toFixed(2)}</b> perplexity</span>}
    </span>
  );
}

/** Puts a loose experiment or piece of code into a project. */
function MoveTo({ kind, id, projects, onMoved }: { kind: "experiment" | "code"; id: number; projects: Project[]; onMoved: () => void }) {
  return (
    <select className="field select pj-move" value="" aria-label="Move to project"
      onChange={(e) => { if (e.target.value) api.projectAssign(kind, id, Number(e.target.value)).then(onMoved); }}>
      <option value="">Move to project…</option>
      {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
    </select>
  );
}

export default function ProjectsPage({ projectId, status, open }: { projectId: number | null; status: Status | null; open: (id: number | null) => void }) {
  return projectId != null ? <ProjectView id={projectId} status={status} back={() => open(null)} /> : <ProjectList open={open} />;
}

// ---------- The list ----------

function ProjectList({ open }: { open: (id: number) => void }) {
  const [data, setData] = useState<{ projects: Project[]; unassigned: { experiments: ProjectExperiment[]; code: ProjectCode[] } } | null>(null);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState(""), [goal, setGoal] = useState("");
  const [err, setErr] = useState(""), [busy, setBusy] = useState(false);
  const load = useCallback(() => api.projects().then(setData).catch((e) => setErr((e as Error).message)), []);
  useEffect(() => { load(); }, [load]);

  const create = async () => {
    if (!name.trim() || busy) return;
    setBusy(true); setErr("");
    try { const { id } = await api.projectCreate(name, goal); open(id); }
    catch (e) { setErr((e as Error).message); setBusy(false); }
  };
  const loose = data ? data.unassigned.experiments.length + data.unassigned.code.length : 0;

  return (
    <div className="page"><div className="page-inner wide">
      <div className="pj-head">
        <div>
          <h1 className="page-title">Projects</h1>
          <p className="page-sub">Everything you are making, each in one place: the model, every version you try, the results, and what you ship.</p>
        </div>
        {data && data.projects.length > 0 && <button className="btn" onClick={() => setCreating(true)}><Plus /> New project</button>}
      </div>
      {err && <p className="error" role="alert">{err}</p>}

      {(creating || (data && data.projects.length === 0)) && (
        <form className="group pad pj-new" onSubmit={(e) => { e.preventDefault(); create(); }}>
          <h2>{data && data.projects.length === 0 ? "Start your first project" : "New project"}</h2>
          <label>Name<input className="field" autoFocus value={name} maxLength={80} placeholder="e.g. Support assistant" onChange={(e) => setName(e.target.value)} /></label>
          <label><span>Goal <span className="muted">(optional)</span></span>
            <input className="field" value={goal} maxLength={500} placeholder="e.g. Answers customer questions in Czech and runs on a 4 GB GPU" onChange={(e) => setGoal(e.target.value)} /></label>
          <div className="flow-toolbar">
            <button className="btn" type="submit" disabled={!name.trim() || busy}><FolderKanban /> Create project</button>
            {data && data.projects.length > 0 && <button className="btn ghost" type="button" onClick={() => setCreating(false)}>Cancel</button>}
          </div>
          <p className="row-sub">You choose what to start from inside the project: an existing model, training from scratch, or your own code.</p>
        </form>
      )}

      {!data && !err && <p role="status">Loading projects…</p>}
      {data && data.projects.length > 0 && (
        <div className="pj-grid">
          {data.projects.map((p) => {
            const versions = p.experiments.reduce((n, e) => n + e.done, 0);
            const active = p.experiments.reduce((n, e) => n + e.active, 0);
            const best = p.experiments.find((e) => e.latest?.results || e.baseline?.results);
            return (
              <button key={p.id} className="pj-card" onClick={() => open(p.id)}>
                <span className="pj-card-icon"><FolderKanban /></span>
                <b className="pj-card-name">{p.name}</b>
                {p.goal ? <span className="pj-card-goal">{p.goal}</span> : <span className="pj-card-goal muted">No goal written yet</span>}
                <span className="pj-card-meta">
                  {p.experiments.length} model{p.experiments.length === 1 ? "" : "s"} · {versions} version{versions === 1 ? "" : "s"} measured
                  {p.code.length > 0 && <> · {p.code.length} code</>}
                  {active > 0 && <> · <span className="pj-live">{active} running</span></>}
                </span>
                {best && <KeyNumbers r={(best.latest ?? best.baseline)?.results} />}
                <span className="pj-card-foot">Active {ago(p.last_activity)}</span>
              </button>
            );
          })}
        </div>
      )}

      {data && loose > 0 && (
        <details className="group pad pj-loose" open={data.projects.length === 0 || undefined}>
          <summary>Not in a project yet ({loose})</summary>
          <p className="row-sub">Work from before projects existed. Move each item into a project to see it there; nothing is copied or deleted.</p>
          {data.unassigned.experiments.map((e) => (
            <div className="pj-loose-row" key={`e${e.id}`}>
              <Boxes /><span><b>{e.name}</b><span className="muted"> · {e.done} version{e.done === 1 ? "" : "s"} measured</span></span>
              {data.projects.length > 0 ? <MoveTo kind="experiment" id={e.id} projects={data.projects} onMoved={load} /> : <a href={`#/experiments?exp=${e.id}`}>Open</a>}
            </div>
          ))}
          {data.unassigned.code.map((c) => (
            <div className="pj-loose-row" key={`c${c.id}`}>
              <Code /><span><b>{c.name}</b><span className="muted"> · your code</span></span>
              {data.projects.length > 0 ? <MoveTo kind="code" id={c.id} projects={data.projects} onMoved={load} /> : <a href="#/code">Open</a>}
            </div>
          ))}
        </details>
      )}
    </div></div>
  );
}

// ---------- Inside a project ----------

function ProjectView({ id, status, back }: { id: number; status: Status | null; back: () => void }) {
  const [p, setP] = useState<Project | null>(null);
  const [err, setErr] = useState("");
  const [tab, setTabState] = useState<Tab>(() => {
    const t = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("tab");
    if (t === "versions") return "tests";
    return TABS.includes(t as Tab) ? t as Tab : "overview";
  });
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  const [openExp, setOpenExp] = useState<number | null>(null);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(""), [goal, setGoal] = useState("");
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  const load = useCallback(() => api.project(id).then((x) => { setP(x); setErr(""); }).catch((e) => setErr((e as Error).message)), [id]);
  const loadData = useCallback(() => api.projectData(id).then((d) => setDatasets(d.datasets)).catch(() => {}), [id]);
  useEffect(() => { load(); loadData(); const t = setInterval(load, 10000); return () => clearInterval(t); }, [load, loadData]);
  const setTab = (t: Tab) => {
    setTabState(t);
    window.history.replaceState(null, "", `#/projects/${id}${t === "overview" ? "" : `?tab=${t}`}`);
  };
  const openVersions = (eid: number | null, view: "tests" | "canvas" = "tests") => { setOpenExp(eid); setTab(view); };
  const build = (r: Recipe) => { setRecipe(r); setTab("build"); };

  if (err && !p) return <div className="page"><div className="page-inner"><p className="error">{err}</p><button className="btn ghost" onClick={back}><ArrowLeft /> All projects</button></div></div>;
  if (!p) return <div className="page"><div className="page-inner"><p role="status">Loading project…</p></div></div>;

  const save = async () => { await api.projectUpdate(p.id, { name, goal }); setEditing(false); load(); };
  const remove = async () => {
    if (!confirm(`Delete the project "${p.name}"? Its experiments and code are kept and appear under "Not in a project yet".`)) return;
    await api.projectDelete(p.id); back();
  };
  const empty = p.experiments.length === 0 && p.code.length === 0;
  const versions = p.experiments.reduce((n, e) => n + e.done, 0);

  return (
    <div className="page"><div className="page-inner wide">
      <button className="pj-back" onClick={back}><ArrowLeft /> All projects</button>
      {editing ? (
        <form className="pj-edit" onSubmit={(e) => { e.preventDefault(); save(); }}>
          <input className="field pj-edit-name" autoFocus value={name} maxLength={80} onChange={(e) => setName(e.target.value)} aria-label="Project name" />
          <input className="field" value={goal} maxLength={500} placeholder="Goal (optional)" onChange={(e) => setGoal(e.target.value)} aria-label="Goal" />
          <div className="flow-toolbar"><button className="btn" type="submit" disabled={!name.trim()}>Save</button><button className="btn ghost" type="button" onClick={() => setEditing(false)}>Cancel</button></div>
        </form>
      ) : (
        <div className="pj-head">
          <div>
            <h1 className="page-title">{p.name}</h1>
            <p className="page-sub">{p.goal || <span className="muted">No goal yet. Writing one down helps you decide when a version is good enough.</span>}</p>
          </div>
          <div className="flow-toolbar">
            <button className="icon-btn" title="Rename or change the goal" onClick={() => { setName(p.name); setGoal(p.goal); setEditing(true); }}><Pencil /></button>
            <button className="icon-btn danger" title="Delete project (keeps its experiments and code)" onClick={remove}><Trash2 /></button>
          </div>
        </div>
      )}

      <Tabs value={tab} onChange={setTab} tabs={[
        { id: "overview", label: "Overview" },
        { id: "build", label: "Build" },
        { id: "data", label: "Data", badge: datasets.length || null },
        { id: "canvas", label: "Canvas" },
        { id: "tests", label: "Tests", badge: versions || null },
        { id: "ship", label: "Ship" },
      ]} />

      {tab === "overview" && (empty
        ? <Begin project={p} onStarted={(eid) => { load(); openVersions(eid); }} build={build} />
        : <Overview project={p} openVersions={openVersions} build={build} onChanged={load} />)}
      {tab === "build" && <Build project={p} recipe={recipe} setRecipe={setRecipe} onChanged={load} openVersions={openVersions} datasets={datasets} goData={() => setTab("data")} />}
      {tab === "data" && <DataTab project={p} datasets={datasets} reload={loadData} build={build} />}
      {(tab === "tests" || tab === "canvas") && p.experiments.length === 0 && (
        <div className="group pad pj-empty"><b>No models in this project yet.</b>
          <p>{tab === "canvas" ? "The canvas maps every version you try and what it led to." : "Tests measure each version: quality, speed, memory and energy, with their spread."} Add a starting model on the Overview tab first.</p>
          <button className="btn" onClick={() => setTab("overview")}>Go to Overview</button></div>)}
      {tab === "tests" && p.experiments.length > 0 && <ExperimentsPage status={status} projectId={p.id} openExp={openExp} onChanged={load} view="report" />}
      {tab === "canvas" && p.experiments.length > 0 && <ExperimentsPage status={status} projectId={p.id} openExp={openExp} onChanged={load} view="canvas" />}
      {tab === "ship" && <Ship project={p} openVersions={openVersions} />}
    </div></div>
  );
}

/** An empty project: how do you want to begin? */
function Begin({ project, onStarted, build }: { project: Project; onStarted: (eid: number) => void; build: (r: Recipe) => void }) {
  const [models, setModels] = useState<ModelFile[] | null>(null);
  const [pick, setPick] = useState(""), [busy, setBusy] = useState(false), [err, setErr] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const loadModels = (select?: string) => api.models().then((d) => { setModels(d.models); setPick(select ?? d.models[0]?.file ?? ""); }).catch((e) => setErr((e as Error).message));
  useEffect(() => { loadModels(); }, []);
  const uploadModel = async (files: File[]) => {
    const f = files[0];
    if (!f || busy) return;
    setBusy(true); setErr(""); setProgress(0);
    try { const r = await api.modelUpload(f, setProgress); await loadModels(r.file); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(false); setProgress(null); }
  };
  const start = async () => {
    const m = models?.find((x) => x.file === pick);
    if (!m || busy) return;
    setBusy(true); setErr("");
    try { const { id } = await api.experimentCreate(modelTitle(m), m.file, project.id); onStarted(id); }
    catch (e) { setErr((e as Error).message); setBusy(false); }
  };
  return (
    <div className="pj-begin">
      <h2>How do you want to begin?</h2>
      <p className="row-sub">Pick one now; you can add more starting points to this project later.</p>
      {err && <p className="error">{err}</p>}
      <div className="pj-ways">
        <div className="pj-way">
          <span className="pj-way-icon"><Boxes /></span>
          <b>Start from an existing model</b>
          <span>Build on a model that is already good, then change it until it fits your goal.</span>
          {models === null ? <span className="muted">Loading models…</span> : <>
            {models.length > 0 && <>
              <select className="field select" value={pick} onChange={(e) => setPick(e.target.value)} aria-label="Starting model">
                {models.map((m) => <option key={m.file} value={m.file}>{modelTitle(m)} · {m.quant} · {m.size_label}</option>)}
              </select>
              <button className="btn" disabled={!pick || busy} onClick={start}><Plus /> Use this model</button>
            </>}
            {progress !== null && <ProgressBar value={progress} label="Uploading the model" />}
            <span className="pj-way-more">
              {models.length === 0 ? "Get a model: " : "Or get another: "}
              <a href="#/models">download one</a> · <FilePick label="upload a .gguf file" accept=".gguf" ghost disabled={busy} onFiles={uploadModel} />
            </span>
          </>}
        </div>
        <button className="pj-way" onClick={() => build("train")}>
          <span className="pj-way-icon"><Brain /></span>
          <b>Train from scratch</b>
          <span>Pretrain a new model with a known architecture on your own data, then measure it like any other.</span>
          <em>Set up training →</em>
        </button>
        <button className="pj-way" onClick={() => build("code")}>
          <span className="pj-way-icon"><Code /></span>
          <b>Bring your own code</b>
          <span>A new architecture, your own training loop, or a different kind of AI. Blocky runs it, records it and compares it.</span>
          <em>Open a code workspace →</em>
        </button>
      </div>
    </div>
  );
}

/** A project with work in it: where each model stands, and what to do next. */
function Overview({ project, openVersions, build, onChanged }: { project: Project; openVersions: (eid: number | null, view?: "tests" | "canvas") => void; build: (r: Recipe) => void; onChanged: () => void }) {
  const [adding, setAdding] = useState(false);
  const unmeasured = project.experiments.find((e) => e.done === 0 && e.active === 0);
  const measured = project.experiments.find((e) => e.done > 0);
  return (
    <div className="pj-overview">
      <div className="group pad pj-next">
        <Sparkles />
        {project.experiments.some((e) => e.active > 0)
          ? <div><b>Measuring now.</b><span>Results appear under Versions as each run finishes. Long runs keep the GPU busy; the chat model is paused meanwhile.</span></div>
          : unmeasured
            ? <div><b>Next: measure the starting point of {unmeasured.name}.</b><span>Every change is compared against it, so it comes first.</span><button className="btn small" onClick={() => openVersions(unmeasured.id)}><Play /> Open and measure</button></div>
            : measured
              ? <div><b>Next: try a change and see whether it helps.</b><span>Make a smaller version, fine-tune it on your data, or change one setting in Versions.</span><span className="flow-toolbar"><button className="btn small" onClick={() => build("smaller")}><Layers /> Make it smaller</button><button className="btn small ghost" onClick={() => build("finetune")}><Wand2 /> Fine-tune</button><button className="btn small ghost" onClick={() => openVersions(measured.id)}><GitBranch /> Change a setting</button></span></div>
              : <div><b>Next: build something.</b><span>Your code is here; when it reports results, use them in Versions.</span><button className="btn small" onClick={() => build("code")}>Open your code</button></div>}
      </div>

      {project.experiments.length > 0 && <div className="group-label">Models in this project</div>}
      <div className="pj-lines">
        {project.experiments.map((e) => (
          <div className="group pad pj-line" key={e.id}>
            <div className="pj-line-head">
              <span><b>{e.name}</b><span className="muted"> · {e.model}</span></span>
              <span className="flow-toolbar"><button className="btn small ghost" onClick={() => openVersions(e.id, "canvas")}><GitBranch /> Canvas</button><button className="btn small ghost" onClick={() => openVersions(e.id)}><Play /> Tests</button></span>
            </div>
            <div className="pj-line-body">
              <div><span className="pj-k">Starting point</span>{e.baseline ? <KeyNumbers r={e.baseline.results} /> : <span className="muted">not measured yet</span>}</div>
              {e.latest && <div><span className="pj-k">Latest · {e.latest.label}</span><KeyNumbers r={e.latest.results} /></div>}
            </div>
            <span className="row-sub">{e.done} version{e.done === 1 ? "" : "s"} measured{e.active > 0 && <> · <span className="pj-live">{e.active} running</span></>}</span>
          </div>
        ))}
        {project.code.map((c) => (
          <div className="group pad pj-line" key={`c${c.id}`}>
            <div className="pj-line-head"><span><Code size={15} /> <b>{c.name}</b><span className="muted"> · your code</span></span>
              <button className="btn small ghost" onClick={() => build("code")}>Open</button></div>
          </div>
        ))}
      </div>
      {adding ? <Begin project={project} onStarted={(eid) => { setAdding(false); onChanged(); openVersions(eid); }} build={build} />
        : <button className="btn ghost" onClick={() => setAdding(true)}><Plus /> Add another starting point</button>}
    </div>
  );
}

const RECIPES: { id: Recipe; icon: typeof Hammer; title: string; what: string }[] = [
  { id: "smaller", icon: Layers, title: "Make it smaller or faster", what: "Quantize or prune a model so it fits weaker hardware." },
  { id: "finetune", icon: Wand2, title: "Fine-tune on your data", what: "Teach an existing model your task with LoRA or QLoRA." },
  { id: "train", icon: Brain, title: "Train all weights or from scratch", what: "Full fine-tuning, or pretraining a new model." },
  { id: "code", icon: Code, title: "Your own code", what: "Run your architecture, trainer or evaluator." },
  { id: "method", icon: Sparkles, title: "Try a new quantization method", what: "Prototype it in Python before writing a kernel." },
];

/** Build: every way to make a new version, in one place. */
function Build({ project, recipe, setRecipe, onChanged, openVersions, datasets, goData }: { project: Project; recipe: Recipe | null; setRecipe: (r: Recipe | null) => void; onChanged: () => void; openVersions: (eid: number | null) => void; datasets: Dataset[]; goData: () => void }) {
  const ggufs = Array.from(new Set(project.experiments.map((e) => e.model).filter((m) => m.endsWith(".gguf"))));
  const [model, setModel] = useState(ggufs[0] ?? "");
  const [built, setBuilt] = useState(false);
  return (
    <div className="pj-build">
      <div className="pj-recipes" role="tablist">
        {RECIPES.map((r) => (
          <button key={r.id} role="tab" aria-selected={recipe === r.id} className={`pj-recipe ${recipe === r.id ? "on" : ""}`} onClick={() => setRecipe(r.id)}>
            <r.icon /><b>{r.title}</b><span>{r.what}</span>
          </button>
        ))}
      </div>
      {!recipe && <p className="row-sub">Choose what you want to build. Everything you build becomes a model you can measure under Versions.</p>}

      {recipe === "smaller" && (ggufs.length === 0
        ? <div className="group pad pj-empty"><b>Add a starting model first.</b><p>A smaller version is made from a model in this project. Add one on the Overview tab.</p></div>
        : <div className="group pad">
          {ggufs.length > 1 && <label className="pj-inline">From<select className="field select" value={model} onChange={(e) => setModel(e.target.value)}>{ggufs.map((m) => <option key={m}>{m}</option>)}</select></label>}
          <WeightsWorkshop model={model} onBuilt={() => setBuilt(true)} />
          {built && <p className="pj-hint">Built. To measure it, open <button className="link" onClick={() => openVersions(project.experiments.find((e) => e.model === model)?.id ?? null)}>Versions</button> and choose <i>Change → Weights</i>.</p>}
        </div>)}
      {recipe === "finetune" && <><DataHint datasets={datasets} goData={goData} /><InternalsPage embedded initialTab="ft" datasets={datasets} /></>}
      {recipe === "train" && <><DataHint datasets={datasets} goData={goData} /><InternalsPage embedded initialTab="train" datasets={datasets} /></>}
      {recipe === "method" && <InternalsPage embedded initialTab="play" />}
      {recipe === "code" && <ResearchPage codeIds={project.code.map((c) => c.id)} onCreate={(cid) => api.projectAssign("code", cid, project.id).then(onChanged)} />}
    </div>
  );
}

/** Ship: what you can do with a finished version today. */
function Ship({ project, openVersions }: { project: Project; openVersions: (eid: number | null) => void }) {
  const [runs, setRuns] = useState<Record<number, ExpRun[]>>({});
  const [msg, setMsg] = useState(""), [busy, setBusy] = useState(false);
  useEffect(() => {
    project.experiments.filter((e) => e.done > 0).forEach((e) =>
      api.experiment(e.id).then((x) => setRuns((r) => ({ ...r, [e.id]: (x.runs_list ?? []).filter((run) => run.status === "done") }))).catch(() => {}));
  }, [project]);
  const use = async (file: string) => {
    if (!confirm(`Make ${file} this device's chat model? The current chat model is replaced (you can switch back on the Models page).`)) return;
    setBusy(true); setMsg("");
    try { await api.setRuntime({ model: file }); setMsg(`${file} is now the chat model and the /v1 API on this device. It takes a moment to load.`); }
    catch (e) { setMsg((e as Error).message); } finally { setBusy(false); }
  };
  const ready = project.experiments.filter((e) => e.done > 0);
  if (ready.length === 0) return (
    <div className="group pad pj-empty"><Rocket /><b>Nothing measured to ship yet.</b><p>Ship becomes available once a version has been measured, so what you ship comes with its results.</p></div>
  );
  return (
    <div className="pj-ship">
      {msg && <p role="status" className="pj-hint">{msg}</p>}
      {ready.map((e) => (
        <div className="group pad" key={e.id}>
          <div className="pj-line-head"><b>{e.name}</b>
            <span className="flow-toolbar">
              <a className="btn small ghost" href={`/api/experiments/${e.id}/bundle.zip`} title="Results, provenance, raw outputs and a script that re-runs every measurement on another machine"><Download /> Reproducibility bundle</a>
              <a className="btn small ghost" href={`/api/experiments/${e.id}/export.csv`}><Download /> Results (CSV)</a>
            </span>
          </div>
          {(runs[e.id] ?? []).filter((r, i, all) => all.findIndex((x) => x.config.model === r.config.model) === i).map((r) => {
            const file = r.config.model;
            const local = file.endsWith(".gguf");
            return (
              <div className="pj-ship-row" key={r.id}>
                <span><b>{r.label}</b><span className="muted"> · {file}</span></span>
                <KeyNumbers r={{ mmlu: r.results.mmlu?.acc ?? null, tg: r.results.speed?.tg ?? null, ppl: r.results.ppl ?? null }} />
                {local ? <button className="btn small ghost" disabled={busy} onClick={() => use(file)}><Rocket /> Use on this device</button>
                  : <button className="btn small ghost" onClick={() => openVersions(e.id)}>Details</button>}
              </div>
            );
          })}
        </div>
      ))}
      <p className="row-sub">Coming next: publish to Hugging Face (private or public) with a model card generated from these measurements, export a package, and Ollama.</p>
    </div>
  );
}

/** Above the training forms: where the data comes from. */
function DataHint({ datasets, goData }: { datasets: Dataset[]; goData: () => void }) {
  return (
    <p className="pj-hint">
      {datasets.length > 0
        ? <>Choose one of your {datasets.length} uploaded file{datasets.length === 1 ? "" : "s"} under <b>Your data</b> below, or <button className="link" onClick={goData}>upload more on the Data tab</button>.</>
        : <>Training needs examples of what the model should learn. <button className="link" onClick={goData}>Upload them on the Data tab</button>, or use a Hugging Face dataset below.</>}
    </p>
  );
}

const EXAMPLE_CHAT = `{"instruction": "How do I reset my password?", "output": "Open Settings, choose Account, then Reset password."}
{"instruction": "Do you ship to Slovakia?", "output": "Yes, delivery takes 2 to 4 working days."}`;
const EXAMPLE_MESSAGES = `{"messages": [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello, how can I help?"}]}`;

/** Data: the files this project trains and tests on, uploaded from your computer. */
function DataTab({ project, datasets, reload, build }: { project: Project; datasets: Dataset[]; reload: () => void; build: (r: Recipe) => void }) {
  const [progress, setProgress] = useState<{ value: number; label: string } | null>(null);
  const [err, setErr] = useState(""), [msg, setMsg] = useState("");
  const send = async (files: File[]) => {
    setErr(""); setMsg("");
    for (const f of files) {
      try {
        setProgress({ value: 0, label: `Uploading ${f.name}` });
        await api.projectDataUpload(project.id, f, (value) => setProgress({ value, label: `Uploading ${f.name}` }));
        setMsg(`${f.name} is ready to train on.`);
      } catch (e) { setErr(`${f.name}: ${(e as Error).message}`); }
    }
    setProgress(null); reload();
  };
  const remove = async (d: Dataset) => {
    if (!confirm(`Delete ${d.name} from this project? This removes the file from the device.`)) return;
    await api.projectDataDelete(project.id, d.name); reload();
  };
  return (
    <div className="pj-data">
      <DropZone accept={[".jsonl", ".txt"]} onFiles={send}>
        <Database />
        <b>Drop training files here</b>
        <span>.jsonl (examples of questions and answers) or .txt (plain text)</span>
        <FilePick label="Choose files" accept=".jsonl,.txt" multiple onFiles={send} disabled={!!progress} />
      </DropZone>
      {progress && <ProgressBar value={progress.value} label={progress.label} />}
      {err && <p className="error" role="alert">{err}</p>}
      {msg && <p role="status" className="pj-hint">{msg} <button className="link" onClick={() => build("finetune")}>Fine-tune with it</button></p>}

      {datasets.length > 0 && (
        <div className="group pad">
          {datasets.map((d) => (
            <div className="pj-loose-row" key={d.name}>
              <FileText />
              <span><b>{d.name}</b><span className="muted"> · {fmtBytes(d.bytes)}{d.rows != null ? ` · ${d.rows.toLocaleString()} ${d.kind === "jsonl" ? "examples" : "lines"}` : ""}</span></span>
              <button className="icon-btn danger" title={`Delete ${d.name}`} onClick={() => remove(d)}><Trash2 /></button>
            </div>
          ))}
        </div>
      )}

      <details className="group pad pj-formats" open={datasets.length === 0 || undefined}>
        <summary>What should the files look like?</summary>
        <p><b>To teach a task</b> (fine-tuning): a <code>.jsonl</code> file, one example per line, each with what someone asks and the answer you want.</p>
        <pre className="rp-output">{EXAMPLE_CHAT}</pre>
        <p className="row-sub">Also accepted per line: <code>prompt</code>/<code>response</code>, <code>question</code>/<code>answer</code>, or whole conversations:</p>
        <pre className="rp-output">{EXAMPLE_MESSAGES}</pre>
        <p><b>To teach a language or a field</b> (training from scratch or continued training): a <code>.txt</code> file of plain text, or a <code>.jsonl</code> with a <code>text</code> field per line.</p>
        <p className="row-sub">A few hundred good examples already change a model's behaviour. Keep some examples out of training to test with: put them in a second file.</p>
      </details>
    </div>
  );
}
