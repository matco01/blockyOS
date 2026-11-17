import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, Check, Cpu, Download, Server, FlaskConical, Gauge, Hammer, Loader2, MemoryStick, Play, Plus, Sparkles, Square, Trash2, Upload, Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { api, Device, DevicePreset, RemoteJob, Engine, ExpConfig, Experiment, ExpRun, KernelProfile, ModelFile, Provenance, Status, TorchStatus } from "./api";
import { LineChart, Term, Tabs, useMode } from "./ui";
import { modelTitle } from "./workflow";
import ResearchCanvas from "./ResearchCanvas";
import MetricEvidence from "./MetricEvidence";
import ValidationPanel from "./ValidationPanel";
import InstrumentPanel from "./InstrumentPanel";
import MethodChangePanel from "./MethodChangePanel";
import type { RegisteredMethod } from "./api";
import { evidence } from "./evidence";
import { Planner, RulerTable } from "./Planner";
import WeightsWorkshop from "./WeightsWorkshop";
import EnginesPanel from "./EnginesPanel";
import { allMetrics, colorOf, delta, hwLabel, Metric, n1, nameOf, runTitle, shortModel, adjustedP } from "./expmetrics";
import { Correction, CORRECTION_LABEL, savedCorrection, saveCorrection } from "./stats";

const PHASES = [
  { id: "loading", label: "Load", what: "memory layout, load time" },
  { id: "quality", label: "Quality", what: "MMLU-Pro, 560 questions" },
  { id: "longctx", label: "Long context", what: "needle in a haystack" },
  { id: "speed", label: "Speed", what: "llama-bench, 5 reps" },
  { id: "fidelity", label: "Fidelity", what: "PPL + KLD, 16k tokens" },
];

/* ---------- Pieces ---------- */

/** Frameworks that serve Hugging Face checkpoints (not GGUF files). */
const isCkpt = (e?: string | null) => e === "vllm" || e === "sglang" || e === "trtllm";

function Tile({ icon: Icon, color, label, value, unit, sub }: { icon: LucideIcon; color: string; label: React.ReactNode; value: string; unit?: string; sub?: string }) {
  return (
    <div className="xt" style={{ "--c": color } as React.CSSProperties}>
      <div className="xt-head"><span className="xt-icon"><Icon /></span>{label}</div>
      <div className="xt-value">{value}{unit && <small>{unit}</small>}</div>
      {sub && <div className="xt-sub">{sub}</div>}
    </div>
  );
}

function Profile({ run }: { run: ExpRun }) {
  if (run.config.engine === "research") return <div><b>{run.results.method?.name ?? run.label}</b><p className="row-sub">Registered evaluator · job #{run.results.research_job_id}. Open metric evidence in the results table for samples, contracts and artifacts.</p><div className="xt-grid">{allMetrics([run]).filter((m) => m.researchName).slice(0, 4).map((m) => <div key={m.key}><b>{m.label}</b><p>{m.get(run.results)?.toPrecision(5)} {m.unit}</p></div>)}</div></div>;
  const r = run.results, s = r.speed ?? (r.serving ? { tg: r.serving.tg ?? undefined, tg_sd: r.serving.tg_sd ?? undefined, pp: r.serving.pp ?? undefined, pp_sd: r.serving.pp_sd ?? undefined } : undefined),
    m = r.mmlu, hw = r.hw;
  return (
    <div className="xt-grid">
      <Tile icon={Gauge} color="#7c5cff" label={<Term k="tg">Decode</Term>} value={s?.tg ? n1(s.tg) : "–"} unit={s?.tg_sd ? `± ${s.tg_sd.toFixed(1)} tok/s` : "tok/s"}
        sub={hw?.decode ? `${hw.decode.ms_per_token} ms/token · ${hw.decode.effective_gbps} GB/s effective` : undefined} />
      <Tile icon={Zap} color="#ffa94d" label={<Term k="pp">Prefill</Term>} value={s?.pp ? s.pp.toFixed(0) : "–"} unit={s?.pp_sd ? `± ${s.pp_sd.toFixed(0)} tok/s` : "tok/s"}
        sub={hw?.prefill ? `${hw.prefill.achieved_tflops} TFLOPS of ${hw.peaks.tflops ?? "?"} peak` : undefined} />
      <Tile icon={Sparkles} color="#ff5c9d" label="MMLU-Pro" value={m ? (m.acc * 100).toFixed(1) : "–"} unit="%"
        sub={m ? `95% CI ${(m.ci95[0] * 100).toFixed(0)}–${(m.ci95[1] * 100).toFixed(0)}% · n = ${m.n} · chance 10%` : "Not measured in this run (older profile)"} />
      <Tile icon={MemoryStick} color="#22d3ee" label={<Term k="vram">VRAM</Term>} value={hw?.headroom ? (hw.headroom.vram_model_mb / 1024).toFixed(2) : "–"} unit="GB"
        sub={hw?.headroom ? `${(hw.headroom.vram_free_mb / 1024).toFixed(2)} GB free of ${(hw.headroom.vram_capacity_mb / 1024).toFixed(0)} GB · ${r.memory?.layers?.gpu}/${r.memory?.layers?.total} layers on GPU` : undefined} />
    </div>
  );
}

function Running({ run, onStop }: { run: ExpRun; onStop: () => void }) {
  if (run.config.engine === "research") return <div className="xrun"><b>Evaluating {run.label}</b><p>Registered evaluator · {run.status}. Its supported measurements will appear in the report.</p><button className="btn small ghost" onClick={onStop}>Stop evaluation</button></div>;
  const idx = PHASES.findIndex((p) => p.id === (run.phase === "lm_eval" ? "longctx" : run.phase));
  const mp = run.results.mmlu_progress, np = run.results.needle_progress;
  return (
    <div className="xrun">
      <div className="xrun-head"><Loader2 className="spin" /> <b>Profiling “{runTitle(run)}”</b><span className="row-sub">Chat is paused while the hardware is reserved</span>
        <button className="btn small ghost" onClick={onStop}><Square /> Stop</button></div>
      <div className="xsteps">
        {PHASES.map((p, i) => (
          <div key={p.id} className={`xstep ${i < idx ? "done" : i === idx ? "now" : ""}`}>
            <span>{i < idx ? <Check size={13} /> : i + 1}</span><div><b>{p.label}</b><em>{p.id === "quality" && mp && i === idx ? `${mp.done}/${mp.total} · ${mp.correct} correct` : p.id === "longctx" && np && i === idx ? `${np.done}/${np.total} · ${np.correct} found` : p.id === "longctx" && run.phase === "lm_eval" ? "lm-eval running…" : p.what}</em></div>
          </div>
        ))}
      </div>
    </div>
  );
}

const CTX = [4096, 8192, 16384, 32768, 65536];
const VRAM_SIM = [null, 6144, 4096, 2048];
const RAM_SIM = [null, 16384, 8192, 4096];
const CORES = [2, 4, 8];
const DEVICE_KEYS: (keyof ExpConfig)[] = ["device", "cores", "threads", "ram_limit_mb", "vram_limit_mb"];

type SweepGrid = Partial<Record<keyof ExpConfig, unknown[]>>;

function ChangePanel({ base: rawBase, models, onRun, onSweep, onModels }: { base: ExpConfig; models: ModelFile[]; onRun: (label: string, cfg: Partial<ExpConfig>) => void;
  onSweep: (grid: SweepGrid, repeats: number) => void; onModels: () => void }) {
  const workspaceMode = useMode();
  // Runs made before engines existed have no engine field: they ran on stock.
  const base = useMemo(() => ({ ...rawBase, engine: rawBase.engine ?? "stock" }), [rawBase]);
  const [engineList, setEngineList] = useState<Engine[]>([]);
  const [eng, setEng] = useState(false);
  const loadEngines = () => api.engines().then(setEngineList).catch(() => {});
  const [ckpts, setCkpts] = useState<TorchStatus["checkpoints"]>([]);
  const [presets, setPresets] = useState<Record<string, DevicePreset>>({});
  const [cpuThreads, setCpuThreads] = useState(0);
  const [gpuCount, setGpuCount] = useState(0);
  useEffect(() => { loadEngines(); api.torchStatus().then((s) => setCkpts(s.checkpoints)).catch(() => {});
    api.devicePresets().then((p) => { setPresets(p.presets); setCpuThreads(p.cpu_threads); setGpuCount(p.gpu_count ?? 0); }).catch(() => {}); }, []);
  const [mode, setMode] = useState<"single" | "sweep">("single");
  const [shop, setShop] = useState(() => window.location.hash.includes("workshop=1"));
  const [cfg, setCfg] = useState<ExpConfig>(base);
  const [grid, setGrid] = useState<SweepGrid>({});
  const [repeats, setRepeats] = useState(1);
  useEffect(() => setCfg(base), [base]);
  const baseModel = models.find((m) => m.file === base.model);
  const family = models.filter((m) => baseModel && m.architecture === baseModel.architecture && (m.name ?? "") === (baseModel.name ?? ""));
  const others = models.filter((m) => !family.includes(m) && !m.error);
  // A device preset is one change, even though it sets several settings.
  const presetKeys = cfg.preset && cfg.preset !== base.preset ? Object.keys(presets[cfg.preset]?.config ?? {}) : [];
  const changed = (Object.keys(cfg) as (keyof ExpConfig)[]).filter((k) => String(cfg[k] ?? "") !== String(base[k] ?? "") && !presetKeys.includes(k));
  const set = (patch: Partial<ExpConfig>) => setCfg((c) => {
    let next = { ...c, ...patch };
    if ("preset" in patch) next = patch.preset ? { ...next, ...presets[patch.preset]?.config } : { ...next, device: "gpu", cores: null, threads: null, ram_limit_mb: null, vram_limit_mb: null };
    else if (DEVICE_KEYS.some((k) => k in patch)) next.preset = null;  // customizing the device: no longer the preset
    // Engines run different formats: vLLM and SGLang take checkpoints, llama.cpp takes GGUF. Keep the pair valid.
    if (isCkpt(patch.engine) && !next.model.startsWith("hf:") && ckpts[0]) next.model = `hf:${ckpts[0].id}`;
    if (isCkpt(patch.engine)) next = { ...next, kv: "auto", ngl: "auto", fa: "auto", threads: null, ram_limit_mb: null, device: "gpu", preset: null, split_mode: null, rpc: null };
    if (patch.engine && !isCkpt(patch.engine)) next.pp = null;
    if (patch.engine === "sglang") next.pp = null;
    if (patch.engine && !isCkpt(patch.engine) && next.kv === "auto") next.kv = "q8_0";
    if (patch.engine && !isCkpt(patch.engine) && next.model.startsWith("hf:")) next.model = base.model.startsWith("hf:") ? (family[0]?.file ?? models[0]?.file ?? next.model) : base.model;
    return next;
  });
  const inGrid = (k: keyof ExpConfig, v: unknown) => (grid[k] ?? []).some((x) => String(x ?? "") === String(v ?? ""));
  const toggle = (k: keyof ExpConfig, v: unknown) => setGrid((g) => {
    const cur = g[k] ?? [];
    const next = inGrid(k, v) ? cur.filter((x) => String(x ?? "") !== String(v ?? "")) : [...cur, v];
    return { ...g, [k]: next };
  });
  const Opt = ({ k, v, label }: { k: keyof ExpConfig; v: unknown; label: string }) => mode === "single"
    ? <button disabled={isCkpt(cfg.engine) && ((["kv", "ngl", "fa"].includes(k) && v !== "auto") || (k === "ram_limit_mb" && v != null) || (k === "device" && v === "cpu") || (k === "preset" && !!v) || (k === "split_mode" && v != null))} className={String(cfg[k] ?? "") === String(v ?? "") ? "on" : ""} onClick={() => set({ [k]: v } as Partial<ExpConfig>)}>{label}</button>
    : <button className={inGrid(k, v) ? "on multi" : ""} onClick={() => toggle(k, v)}>{label}</button>;
  const swept = (Object.keys(grid) as (keyof ExpConfig)[]).filter((k) => (grid[k] ?? []).length > 0);
  const nRuns = swept.length ? swept.reduce((n, k) => n * (grid[k]!.length), 1) * repeats : 0;
  return (
    <div className="xchange">
      <div className="xc-mode">
        <div className="segmented"><button className={mode === "single" ? "on" : ""} onClick={() => setMode("single")}>One change</button>
          <button className={mode === "sweep" ? "on" : ""} onClick={() => setMode("sweep")}>Sweep</button></div>
        <span className="row-sub">{mode === "single" ? "Change one setting and profile it against its starting point."
          : "Select several values per setting. Every combination gets queued and profiled one after another; leave it running."}</span>
      </div>
      <div className="xc-row"><div><b>Weights</b><span>Versions of this model's weights: downloaded, or built in the workshop</span></div>
        <div className="ws-opts">
          <div className="segmented wrap">{family.map((m) => <Opt key={m.file} k="model" v={m.file} label={`${m.file.replace(".gguf", "").replace(/^.*?-(?=(UD-)?(I?Q\d|BF16|F16))/i, "")} · ${(m.size_bytes / 2 ** 30).toFixed(1)} GB`} />)}</div>
          <button className={`btn small ${shop ? "" : "ghost"}`} onClick={() => setShop(!shop)}><Hammer /> {shop ? "Close workshop" : "Make a variant"}</button>
        </div></div>
      {shop && <div className="xc-shop"><WeightsWorkshop model={base.model} onBuilt={onModels} /></div>}
      {others.length > 0 && (
        <div className="xc-row"><div><b>A different model</b><span>Same tests, same hardware. Speed, memory and MMLU-Pro always compare; KL divergence and perplexity only between models that share a tokenizer.</span></div>
          <div className="segmented wrap">{others.map((m) => <Opt key={m.file} k="model" v={m.file} label={`${m.file.replace(".gguf", "")} · ${(m.size_bytes / 2 ** 30).toFixed(1)} GB`} />)}</div></div>
      )}
      <div className="xc-row"><div><b>Inference engine</b><span>Which llama.cpp build runs it: stock, or a branch/fork you built</span></div>
        <div className="ws-opts">
          <div className="segmented wrap">{engineList.filter((e) => e.status === "ready").map((e) => <Opt key={e.id} k="engine" v={e.id} label={e.id === "stock" ? "stock" : e.name} />)}</div>
          <button className={`btn small ${eng ? "" : "ghost"}`} onClick={() => setEng(!eng)}><Cpu /> {eng ? "Close" : "Add engine"}</button>
        </div></div>
      {eng && <div className="xc-shop"><EnginesPanel onChange={loadEngines} /></div>}
      {(mode === "sweep" ? (grid.engine ?? []).some((e) => isCkpt(e as string)) : isCkpt(cfg.engine)) && (
        <div className="xc-row"><div><b>Checkpoint (vLLM / SGLang / TensorRT-LLM)</b><span>These frameworks run original Hugging Face weights (16-bit). Download more in Internals → Checkpoints.</span></div>
          <div className="segmented wrap">{ckpts.length ? ckpts.map((c) => <Opt key={c.id} k="model" v={`hf:${c.id}`} label={`${c.repo} · ${(c.size_bytes / 2 ** 30).toFixed(1)} GB`} />)
            : <span className="row-sub" style={{ padding: "4px 10px" }}>No checkpoints yet</span>}</div></div>
      )}
      <div className="xc-row"><div><b><Term k="kv">KV cache type</Term></b><span>-ctk / -ctv</span></div>
        <div className="segmented">{isCkpt(cfg.engine) && <Opt k="kv" v="auto" label="Engine default" />}<Opt k="kv" v="f16" label="f16" /><Opt k="kv" v="q8_0" label="q8_0" /><Opt k="kv" v="q4_0" label="q4_0" /></div></div>
      <div className="xc-row"><div><b><Term k="context">Context size</Term></b><span>-c</span></div>
        <div className="segmented">{CTX.map((c) => <Opt key={c} k="ctx" v={c} label={`${c / 1024}k`} />)}</div></div>
      <details className="flow-advanced" open={workspaceMode === "pro" || mode === "sweep" || undefined}><summary>Device limits and advanced inference settings</summary>
      <div className="xc-row"><div><b>Target device</b><span>Simulate a cheaper device: CPU only, fewer cores, less RAM or VRAM. Capacity and cores are real limits; per-core speed and memory bandwidth aren't, so CPU presets add a projected decode speed for the target's memory.</span></div>
        <div className="segmented wrap"><Opt k="preset" v={null} label="This device" />{Object.entries(presets).map(([id, p]) => <Opt key={id} k="preset" v={id} label={p.label} />)}</div></div>
      <div className="xc-row"><div><b>Processor</b><span>CPU only runs with no GPU at all (-dev none); cores pins the run to the first N CPU threads</span></div>
        <div className="ws-opts">
          <div className="segmented"><Opt k="device" v="gpu" label="GPU" /><Opt k="device" v="cpu" label="CPU only" /></div>
          <div className="segmented"><Opt k="cores" v={null} label="All cores" />{CORES.filter((n) => !cpuThreads || n < cpuThreads).map((n) => <Opt key={n} k="cores" v={n} label={`${n}`} />)}</div>
        </div></div>
      <div className="xc-row"><div><b>RAM limit</b><span>The run fails if the model's CPU-side memory plus a minimal system wouldn't fit</span></div>
        <div className="segmented">{RAM_SIM.map((v) => <Opt key={String(v)} k="ram_limit_mb" v={v} label={v ? `${v / 1024} GB` : "Full"} />)}</div></div>
      <div className="xc-row"><div><b>VRAM limit</b><span>Simulates a smaller GPU (--fit-target)</span></div>
        <div className="segmented">{VRAM_SIM.map((v) => <Opt key={String(v)} k="vram_limit_mb" v={v} label={v ? `${v / 1024} GB` : "Full"} />)}</div></div>
      <div className="xc-row"><div><b><Term k="layers">GPU layers</Term></b><span>-ngl</span></div>
        <div className="segmented"><Opt k="ngl" v="auto" label="Auto" />{[24, 16, 8, 0].map((n) => <Opt key={n} k="ngl" v={n} label={String(n)} />)}</div></div>
      <div className="xc-row"><div><b><Term k="fa">Flash attention</Term></b><span>-fa</span></div>
        <div className="segmented"><Opt k="fa" v="auto" label="Auto" /><Opt k="fa" v="on" label="On" /><Opt k="fa" v="off" label="Off" /></div></div>
      {gpuCount > 1 && <>
        <div className="xc-row"><div><b>GPUs</b><span>Use the first N GPUs (CUDA_VISIBLE_DEVICES). vLLM and SGLang use them as tensor-parallel size; llama.cpp spreads the model by the split mode below</span></div>
          <div className="segmented"><Opt k="gpus" v={null} label="All" />{Array.from({ length: gpuCount }, (_, i) => i + 1).map((n) => <Opt key={n} k="gpus" v={n} label={String(n)} />)}</div></div>
        <div className="xc-row"><div><b>Multi-GPU split</b><span>llama.cpp -sm: layer (pipeline; one GPU works at a time per sequence), row (tensor parallel; every GPU works on every layer), none (one GPU)</span></div>
          <div className="segmented"><Opt k="split_mode" v={null} label="Auto" /><Opt k="split_mode" v="layer" label="Layer" /><Opt k="split_mode" v="row" label="Row" /><Opt k="split_mode" v="none" label="None" /></div></div>
      </>}
      {mode === "single" && !isCkpt(cfg.engine) && <RpcWorkers value={cfg.rpc ?? []} onChange={(rpc) => set({ rpc: rpc.length ? rpc : null })} />}
      {(cfg.engine === "vllm" || cfg.engine === "trtllm") && (
        <div className="xc-row"><div><b>Pipeline stages</b><span>Split the model's layers into stages. vLLM places stages on other machines through a Ray cluster you start first (<code>ray start --head</code> here, <code>ray start --address=…</code> on the others); TensorRT-LLM uses this machine's GPUs</span></div>
          <div className="segmented"><Opt k="pp" v={null} label="1" />{[2, 4, 8].map((n) => <Opt key={n} k="pp" v={n} label={String(n)} />)}</div></div>
      )}
      </details>
      {mode === "single" ? (
        <div className="xc-foot">
          <span className="row-sub">{changed.length === 0 ? "Pick one variable to change against the baseline."
            : changed.length === 1 ? "One variable changed. Repeat both setups to distinguish an effect from run-to-run noise." : `${changed.length} variables changed at once: effects will be confounded.`}</span>
          <div className="xc-btns">
            <button className="btn ghost" onClick={() => onRun("", base)}
              title="Runs this exact setup again: the difference between the two shows how much results drift between sessions on this machine.">Repeat this setup</button>
            <button className="btn" disabled={changed.length === 0} onClick={() => onRun("", cfg)}><Play /> Profile this change</button>
          </div>
        </div>
      ) : (
        <div className="xc-foot">
          <span className="row-sub">
            {nRuns === 0 ? "Select values above. Settings you leave empty stay as in the baseline."
              : <>{swept.map((k) => `${grid[k]!.length} ${k === "vram_limit_mb" ? "VRAM limits" : k === "model" ? "weight files" : k}`).join(" × ")}{repeats > 1 ? ` × ${repeats} repeats` : ""} = <b>{nRuns} runs</b>{nRuns > 32 ? " · limit is 32 per sweep" : ""}</>}
          </span>
          <div className="xc-btns">
            <label className="xrep">Repeats <select className="field select small-select" value={repeats} onChange={(e) => setRepeats(Number(e.target.value))}>{[1, 2, 3].map((n) => <option key={n}>{n}</option>)}</select></label>
            <button className="btn" disabled={nRuns === 0 || nRuns > 32} onClick={() => { onSweep(grid, repeats); setGrid({}); }}><Play /> Queue {nRuns || ""} runs</button>
          </div>
        </div>
      )}
    </div>
  );
}

/** Preregistration: the hypothesis and analysis, fixed before the data. Append-only. */
function PreregPanel({ exp, onSaved }: { exp: Experiment; onSaved: () => void }) {
  const [open, setOpen] = useState(false);
  const [p, setP] = useState({ hypothesis: "", primary_metric: "kld", claim: "difference", direction: "either", alpha: 0.05, min_effect: "", equivalence_margin_pts: "", analysis: "" });
  const [err, setErr] = useState<string | null>(null);
  const plans = exp.preregistrations ?? [];
  const set = (k: string, v: string | number) => setP((x) => ({ ...x, [k]: v }));
  const save = async () => {
    setErr(null);
    try {
      await api.experimentPreregister(exp.id, { ...p, min_effect: p.min_effect === "" ? null : Number(p.min_effect),
        equivalence_margin_pts: p.equivalence_margin_pts === "" ? null : Number(p.equivalence_margin_pts) });
      setOpen(false); onSaved();
    } catch (e) { setErr((e as Error).message); }
  };
  return (
    <details className="group pad xadv" open={open || undefined} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary><b>Preregistration</b> <span className="row-sub">{plans.length ? `${plans.length} plan${plans.length > 1 ? "s" : ""} recorded` : "fix the hypothesis and analysis before running, so results can be reported as confirmatory"}</span></summary>
      {plans.map((pl) => (
        <div key={pl.id} className="row-sub" style={{ marginTop: 8 }}>
          <b>#{pl.id}</b> · {new Date(pl.created_at * 1000).toLocaleString()} · SHA-256 <code>{pl.sha256.slice(0, 12)}…</code><br />
          {pl.plan.hypothesis}<br />
          Primary: {pl.plan.primary_metric} · {pl.plan.claim.replace("_", "-")}{pl.plan.equivalence_margin_pts ? ` within ±${pl.plan.equivalence_margin_pts} points` : ""} · α = {pl.plan.alpha}
          {pl.plan.min_effect != null ? ` · minimum effect ${pl.plan.min_effect}` : ""} · confirmatory runs: {pl.confirmatory_runs.length ? pl.confirmatory_runs.map((r) => `#${r}`).join(", ") : "none yet"}
        </div>
      ))}
      <div className="ws-grid" style={{ marginTop: 10 }}>
        <label><span>Hypothesis</span><textarea className="field" rows={2} value={p.hypothesis} onChange={(e) => set("hypothesis", e.target.value)} placeholder="e.g. Q4_K_M keeps MMLU-Pro accuracy within 1 point of Q8_0 on this model" /></label>
        <label><span>Primary metric · claim · direction</span><div className="ft-row">
          <select className="field select" value={p.primary_metric} onChange={(e) => set("primary_metric", e.target.value)}>{["kld", "mmlu", "same_top", "ppl", "tg", "pp", "srv_tg", "jpt", "needle"].map((m) => <option key={m}>{m}</option>)}</select>
          <select className="field select" value={p.claim} onChange={(e) => set("claim", e.target.value)}><option value="difference">difference</option><option value="equivalence">equivalence</option><option value="non_inferiority">non-inferiority</option></select>
          <select className="field select" value={p.direction} onChange={(e) => set("direction", e.target.value)}><option>either</option><option>higher</option><option>lower</option></select></div></label>
        <label><span>α · minimum effect · equivalence margin (points)</span><div className="ft-row">
          <input className="field" type="number" step={0.01} value={p.alpha} onChange={(e) => set("alpha", Number(e.target.value))} />
          <input className="field" value={p.min_effect} placeholder="optional" onChange={(e) => set("min_effect", e.target.value)} />
          <input className="field" value={p.equivalence_margin_pts} placeholder="for equivalence" onChange={(e) => set("equivalence_margin_pts", e.target.value)} /></div></label>
        <label><span>Analysis plan</span><textarea className="field" rows={2} value={p.analysis} onChange={(e) => set("analysis", e.target.value)} placeholder="Which runs, which test, how many repeats, what counts as support" /></label>
      </div>
      <div className="xc-btns"><button className="btn small" disabled={p.hypothesis.trim().length < 10} onClick={save}>Record plan</button>
        <span className="row-sub">Plans can't be edited or deleted; a new plan is a new entry with its own time.</span></div>
      {err && <div className="error"><AlertCircle /> {err}</div>}
    </details>
  );
}

/** llama.cpp RPC workers: other machines running rpc-server that hold part of the model. */
function RpcWorkers({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  const [text, setText] = useState(value.join(", "));
  useEffect(() => setText(value.join(", ")), [value.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="xc-row"><div><b>RPC workers (other machines)</b><span>llama.cpp --rpc: addresses of machines running rpc-server (Settings → Multi-machine on another Blocky). Layers are spread over this machine and the workers; the network then sets much of the speed</span></div>
      <input className="field" aria-label="RPC workers" placeholder="192.168.1.40:50052, 192.168.1.41:50052" value={text}
        onChange={(e) => setText(e.target.value)} onBlur={() => onChange(text.split(",").map((x) => x.trim()).filter(Boolean))} /></div>
  );
}

function Board({ runs, metric, setMetric }: { runs: ExpRun[]; metric: string; setMetric: (k: string) => void }) {
  const [selected, setSelected] = useState<{ metric: Metric; run: ExpRun } | null>(null);
  const [showUnavailable, setShowUnavailable] = useState(false);
  const base = runs[0];
  const METRICS = allMetrics(runs);
  const [corr, setCorr] = useState<Correction>(savedCorrection);
  // Every run is compared with the baseline at once: correct each metric's p-values for that many comparisons.
  const adj = useMemo(() => new Map(METRICS.map((m) => [m.key, adjustedP(m, runs.slice(1).map((r) => ({ run: r, base })), corr)])),
    [runs, corr]); // eslint-disable-line react-hooks/exhaustive-deps
  const groups = Array.from(new Set(METRICS.map((m) => m.group)));
  const hasData = (m: Metric) => runs.some((r) => m.get(r.results) != null);
  return (
    <div className="table-wrap xboard">
      {selected && <MetricEvidence metric={selected.metric} run={selected.run} onClose={() => setSelected(null)} />}
      <p className="row-sub">Select the evidence label beside a value to inspect its source, samples, commands and retained log. Simulated means measured here with device constraints.</p>
      <label className="row-sub"><input type="checkbox" checked={showUnavailable} onChange={(ev) => setShowUnavailable(ev.target.checked)} /> Show metrics unavailable in every run</label>
      {runs.length > 2 && (
        <div className="xcorr">
          <span>Comparing {runs.length - 1} runs with the baseline: p-values corrected for multiple comparisons</span>
          <div className="segmented">{(["holm", "bh", "none"] as Correction[]).map((c) => (
            <button key={c} className={corr === c ? "on" : ""} title={c === "holm" ? "Controls the chance of any false positive (strict)" : c === "bh" ? "Controls the share of false positives among the hits (for large sweeps)" : "Each comparison on its own: with many runs, some will look significant by chance"}
              onClick={() => { setCorr(c); saveCorrection(c); }}>{CORRECTION_LABEL[c]}</button>))}</div>
        </div>
      )}
      <table className="data">
        <thead>
          <tr><th>Metric</th>{runs.map((r, i) => (
            <th key={r.id} className="r run-label" title={r.stats?.env_mismatch?.length ? `Not directly comparable: ${r.stats.env_mismatch.join(", ")} differ from the baseline` : r.label}>
              <span className="xdot" style={{ background: colorOf(i) }} />{nameOf(r, i)}{r.stats?.env_mismatch?.length ? <span className="xwarn">⚠</span> : null}
              {(r.stats?.hw_diff?.length || r.results.imported_from) ? <span className="xhwtag" title={`Ran on ${hwLabel(r)}${r.results.imported_from ? " (imported from another device)" : ""}`}>on {hwLabel(r)}</span> : null}</th>
          ))}</tr>
        </thead>
        {groups.map((g) => {
          const rows = METRICS.filter((x) => x.group === g);
          const any = rows.some(hasData);
          return (
            <tbody key={g}>
              <tr className="grp-row"><td colSpan={runs.length + 1}>{g}{g === "Energy" && !any && <span className="muted"> · no power reading for these runs: add a wall meter in Settings → Energy & cost, or use hardware whose sensors report power</span>}</td></tr>
              {rows.filter((x) => showUnavailable || hasData(x)).map((x) => (
                <tr key={x.key} className={metric === x.key ? "sel" : ""} onClick={() => setMetric(x.key)} title={x.note ?? "Click to chart"}>
                  <td className="mlabel">{x.term ? <Term k={x.term}>{x.label}</Term> : x.label}{x.unit && <span className="u"> {x.unit}</span>}</td>
                  {runs.map((r, i) => {
                    const v = x.get(r.results), sd = x.sd?.(r.results);
                    const repeats = runs.slice(1).filter((q) => q.label === "Repeat of baseline" && q.id !== r.id);
                    const bv = x.get(base.results);
                    const session = Math.max(0, ...repeats.map((q) => { const qv = x.get(q.results); return qv != null && bv != null ? Math.abs(qv - bv) : 0; }));
                    const d = i === 0 ? null : delta(x, { run: r, base }, session, adj.get(x.key)?.get(r.id));
                    return (
                      <td key={r.id} className="r">
                        {v == null ? "–" : (x.fmt ?? n1)(v)}{sd ? <span className="sd"> ±{(x.fmt ?? n1)(sd)}</span> : null}
                        {d && <span className={`d ${d.tone}`} title={d.title}>{d.text}</span>}
                        <button className="xe-badge" onClick={(ev) => { ev.stopPropagation(); setSelected({ metric: x, run: r }); }} aria-label={`Evidence for ${x.label}, run ${r.id}`}>{evidence(x, r).kind}</button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          );
        })}
      </table>
    </div>
  );
}

function MetricBars({ runs, metric }: { runs: ExpRun[]; metric: string }) {
  const METRICS = allMetrics(runs);
  const m = METRICS.find((x) => x.key === metric) ?? METRICS[0];
  const vals = runs.map((r) => m.get(r.results));
  const max = Math.max(...vals.map((v, i) => (v ?? 0) + (m.sd?.(runs[i].results) ?? 0)), 1e-9);
  return (
    <div className="xchart">
      <div className="mini-title">{m.label}{m.unit && <span className="muted"> ({m.unit}{m.better ? `, ${m.better} is better` : ""})</span>}</div>
      {runs.map((r, i) => {
        const v = vals[i], sd = m.sd?.(r.results);
        return (
          <div className="xbar-row" key={r.id} title={`${r.label}: ${v == null ? "no data" : (m.fmt ?? n1)(v)}${sd ? ` ± ${(m.fmt ?? n1)(sd)}` : ""}`}>
            <span>{nameOf(r, i)}</span>
            <div className="xbar-track">
              <div style={{ width: `${((v ?? 0) / max) * 100}%`, background: colorOf(i) }} />
              {v != null && sd ? <i style={{ left: `${(Math.max(0, v - sd) / max) * 100}%`, width: `${((2 * sd) / max) * 100}%` }} /> : null}
            </div>
            <b>{v == null ? "–" : (m.fmt ?? n1)(v)}</b>
          </div>
        );
      })}
    </div>
  );
}

/* ---------- Results from other devices ---------- */

function ImportRuns({ exp, onDone }: { exp: Experiment | null; onDone: (expId: number) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const run = async (into?: number) => {
    if (!file) return;
    setErr(null);
    try {
      const r = await api.experimentImport(file, into);
      setMsg(`Imported ${r.runs} runs from ${r.from.device ?? r.from.gpu}.`); setFile(null); onDone(r.experiment_id);
    } catch (e) { setErr((e as Error).message); }
  };
  return (
    <>
      <input ref={input} type="file" accept=".json,application/json" hidden onChange={(e) => { setFile(e.target.files?.[0] ?? null); setMsg(null); e.target.value = ""; }} />
      <button className="btn small ghost" onClick={() => input.current?.click()} title="Bring in an experiment exported (JSON) from another Blocky device, to compare hardware"><Upload /> Import</button>
      {(file || msg || err) && (
        <div className="imp">
          {file && <><span className="row-sub">Import <b>{file.name}</b>: results from another device keep their own hardware snapshot.</span>
            <button className="btn small" onClick={() => run()}>As a new experiment</button>
            {exp && <button className="btn small ghost" onClick={() => run(exp.id)}>Into “{exp.name}”</button>}
            <button className="btn small ghost" onClick={() => setFile(null)}>Cancel</button></>}
          {msg && <span className="row-sub">{msg}</span>}
          {err && <span className="error"><AlertCircle /> {err}</span>}
        </div>
      )}
    </>
  );
}

/** Profile chosen setups on another Blocky device; results come back into this experiment. */
function RemoteRuns({ exp, onImported }: { exp: Experiment; onImported: () => void }) {
  const [open, setOpen] = useState(false);
  const [devs, setDevs] = useState<Device[]>([]);
  const [dev, setDev] = useState<number | null>(null);
  const [pick, setPick] = useState<number[]>([]);
  const [jobs, setJobs] = useState<RemoteJob[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const local = (exp.runs_list ?? []).filter((r) => !r.results.imported_from && r.status !== "cancelled");
  useEffect(() => { if (open) api.devices().then((d) => { setDevs(d); setDev((x) => x ?? d[0]?.id ?? null); }).catch(() => {}); }, [open]);
  useEffect(() => {
    let alive = true, wasActive = false;
    const tick = () => api.remoteJobs(exp.id).then((j) => {
      if (!alive) return;
      const active = j.some((x) => x.status === "starting" || x.status === "running");
      if (wasActive && !active) onImported();
      wasActive = active; setJobs(j);
    }).catch(() => {});
    tick();
    const t = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(t); };
  }, [exp.id]);
  const go = async () => {
    setErr(null);
    try { await api.remoteRun(exp.id, dev!, pick); setPick([]); setOpen(false); setJobs(await api.remoteJobs(exp.id)); } catch (e) { setErr((e as Error).message); }
  };
  return (
    <>
      <button className="btn small ghost" onClick={() => setOpen(!open)} title="Profile these setups on another Blocky device and compare hardware"><Server /> Run on another device</button>
      {(open || jobs.length > 0) && (
        <div className="imp" style={{ flexDirection: "column", alignItems: "stretch" }}>
          {open && (devs.length === 0
            ? <span className="row-sub">No devices yet: add one in <a href="#/settings">Settings → Remote access & devices</a>.</span>
            : <>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <span className="row-sub">Device</span>
                <select className="field select" value={dev ?? ""} onChange={(e) => setDev(Number(e.target.value))}>{devs.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</select>
                <span className="row-sub">Setups to profile there (the same model files must be on that device):</span>
              </div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {local.map((r, i) => <button key={r.id} className={`btn small ${pick.includes(r.id) ? "" : "ghost"}`}
                  onClick={() => setPick(pick.includes(r.id) ? pick.filter((x) => x !== r.id) : [...pick, r.id])}>{nameOf(r, i)}</button>)}
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn small" disabled={!dev || pick.length === 0} onClick={go}><Play /> Profile {pick.length || ""} on {devs.find((d) => d.id === dev)?.name ?? "device"}</button>
                <button className="btn small ghost" onClick={() => setOpen(false)}>Close</button>
              </div>
            </>)}
          {jobs.map((j) => (
            <div key={j.id} className="row-sub">
              <b>{j.device}</b>: {j.status}{j.step ? ` · ${j.step}` : ""}{j.imported ? ` · ${j.imported.runs} runs imported` : ""}{j.error ? ` · ${j.error}` : ""}
              {(j.status === "running" || j.status === "starting") && <> · <a onClick={() => api.remoteJobCancel(j.id)}>Cancel</a></>}
            </div>
          ))}
          {err && <span className="error"><AlertCircle /> {err}</span>}
        </div>
      )}
    </>
  );
}

/* ---------- Long context ---------- */

function NeedleGrid({ runs }: { runs: ExpRun[] }) {
  const rs = runs.filter((r) => r.results.needle?.cells?.length);
  if (!rs.length) return null;
  const lengths = Array.from(new Set(rs.flatMap((r) => r.results.needle!.lengths))).sort((a, b) => a - b);
  return (
    <div className="ng">
      <div className="mini-title">Long-context retrieval <span className="muted">(a code hidden at 5 depths; each cell = how many of the 5 were found; hover for depths)</span></div>
      <div className="table-wrap"><table className="data ng-table">
        <thead><tr><th>Run</th>{lengths.map((L) => <th key={L} className="r">{L >= 1024 ? `${L / 1024}k` : L} tokens</th>)}</tr></thead>
        <tbody>{rs.map((r) => (
          <tr key={r.id}><td className="run-label" title={runTitle(r)}>{runTitle(r)}</td>
            {lengths.map((L) => {
              const cells = r.results.needle!.cells.filter((c) => c.length === L);
              if (!cells.length) return <td key={L} className="r muted" title="Longer than this run's context">–</td>;
              const ok = cells.filter((c) => c.ok).length;
              const pf = r.results.needle!.prefill_tps_by_length?.[String(L)];
              return <td key={L} className="r"><span className={`ng-cell ${ok === cells.length ? "all" : ok === 0 ? "none" : "some"}`}
                title={`${cells.map((c) => `${Math.round(c.depth * 100)}% deep: ${c.ok ? "found" : "missed"}`).join(" · ")}${pf ? ` · read at ${pf} tok/s` : ""}`}>{ok}/{cells.length}</span></td>;
            })}</tr>
        ))}</tbody>
      </table></div>
    </div>
  );
}

function TestProtocol({ exp, onSaved }: { exp: Experiment; onSaved: () => void }) {
  const cur = exp.protocol?.lm_eval;
  const [open, setOpen] = useState(false);
  const [tasks, setTasks] = useState(cur?.tasks ?? "");
  const [limit, setLimit] = useState(cur?.limit ?? 100);
  const [kern, setKern] = useState(!!exp.protocol?.kernels);
  const [mmluN, setMmluN] = useState(exp.protocol?.mmlu_per_category ?? 40);
  const [ruler, setRuler] = useState(exp.protocol?.ruler?.samples ?? 0);
  const [chunks, setChunks] = useState(exp.protocol?.ppl_chunks ?? 32);
  const [ftext, setFtext] = useState(exp.protocol?.fidelity_text ?? "");
  const [margin, setMargin] = useState(exp.protocol?.equivalence_margin_pts ?? 0);
  const [clock, setClock] = useState(exp.protocol?.lock_clocks_mhz ?? 0);
  const [cool, setCool] = useState(exp.protocol?.cooldown_s ?? 0);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { setTasks(exp.protocol?.lm_eval?.tasks ?? ""); setLimit(exp.protocol?.lm_eval?.limit ?? 100); setKern(!!exp.protocol?.kernels);
    setMmluN(exp.protocol?.mmlu_per_category ?? 40); setRuler(exp.protocol?.ruler?.samples ?? 0); setChunks(exp.protocol?.ppl_chunks ?? 32);
    setFtext(exp.protocol?.fidelity_text ?? ""); setMargin(exp.protocol?.equivalence_margin_pts ?? 0); setClock(exp.protocol?.lock_clocks_mhz ?? 0); setCool(exp.protocol?.cooldown_s ?? 0); }, [exp.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const save = async () => {
    setErr(null);
    try { await api.experimentProtocol(exp.id, tasks.trim() ? { tasks, limit } : null, kern, mmluN, ruler ? { samples: ruler } : null, chunks,
      { fidelity_text: ftext.trim() || null, equivalence_margin_pts: margin || null, lock_clocks_mhz: clock || null, cooldown_s: cool || null }); setOpen(false); onSaved(); } catch (e) { setErr((e as Error).message); }
  };
  const extra = cur || exp.protocol?.kernels || exp.protocol?.ruler || (exp.protocol?.mmlu_per_category ?? 40) !== 40 || (exp.protocol?.ppl_chunks ?? 32) !== 32 || !!exp.protocol?.fidelity_text || !!exp.protocol?.equivalence_margin_pts || !!exp.protocol?.lock_clocks_mhz || !!exp.protocol?.cooldown_s;
  return (
    <div className="tp">
      <span className="row-sub"><b>Every run measures:</b> memory · <b>MMLU-Pro ({(14 * (exp.protocol?.mmlu_per_category ?? 40)).toLocaleString()})</b> · long context (2k…ctx) · serving speed · llama-bench speed · perplexity & KLD ({(exp.protocol?.ppl_chunks ?? 32) * 512 >= 1000 ? `${Math.round((exp.protocol?.ppl_chunks ?? 32) * 0.512)}k` : (exp.protocol?.ppl_chunks ?? 32) * 512} tokens) · energy · telemetry
        {exp.protocol?.ruler ? <> · <b>long-context suite</b> ({exp.protocol.ruler.samples}/cell)</> : ""}{cur ? <> · <b>lm-eval: {cur.tasks}</b> (limit {cur.limit})</> : ""}{exp.protocol?.kernels ? <> · <b>GPU kernel profile</b></> : ""}</span>
      <button className="btn small ghost" onClick={() => setOpen(!open)}>{open ? "Close" : "Add tests"}</button>
      {open && (
        <div className="tp-edit">
          <div className="row-sub">Run <b>lm-evaluation-harness</b> tasks too, the field's standard tool, against each run's exact server (greedy, seed 0). Use generative tasks such as <code>gsm8k</code>, <code>ifeval</code>, <code>bbh_cot_fewshot</code>, <code>minerva_math</code>. Multiple-choice log-likelihood tasks need prompt log-probs, which these servers don't provide; MMLU-Pro above already covers that style. Applies to runs queued from now on.</div>
          <div className="ws-grid">
            <label><span>MMLU-Pro size</span><select className="field select" value={mmluN} onChange={(e) => setMmluN(Number(e.target.value))}>
              <option value={40}>560 questions (default)</option><option value={100}>1,400 questions</option><option value={250}>3,500 questions</option></select>
              <em>More questions detect smaller accuracy differences (see “How many runs or questions do I need?” in the results). Larger sets contain the default 560, so paired tests still work against older runs.</em></label>
            <label><span>Perplexity / KL divergence sample</span><select className="field select" value={chunks} onChange={(e) => setChunks(Number(e.target.value))}>
              <option value={32}>32 × 512 tokens (default)</option><option value={64}>64 × 512 tokens</option><option value={128}>128 × 512 tokens</option><option value={256}>256 × 512 tokens</option></select>
              <em>More text narrows the KL divergence and perplexity error bars, and catches rare large deviations (99th-percentile KLD). Set this before the baseline: every run must be scored on the same text, so changing it later needs a new experiment.</em></label>
            <label><span>Perplexity / KL divergence text</span><input className="field" placeholder="WikiText-2 test (default) — or /path/to/your-text.txt" value={ftext} onChange={(e) => setFtext(e.target.value)} />
              <em>Quantization damage differs by domain: score code, another language or your own data with a .txt file on this device (50 KB or more). Set before the baseline; every run is scored on the same text and its SHA-256 goes into provenance.</em></label>
            <label><span>Equivalence margin (accuracy points)</span><input className="field" type="number" min={0} step={0.5} value={margin || ""} placeholder="none" onChange={(e) => setMargin(Number(e.target.value))} />
              <em>To claim “no worse than ±X points”, choose X before running (a preregistration fixes it too). Each comparison then reports a TOST equivalence verdict.</em></label>
            <label><span>Noise controls</span><div className="ft-row">
              <input className="field" type="number" min={0} placeholder="lock GPU clock, MHz" value={clock || ""} onChange={(e) => setClock(Number(e.target.value))} />
              <input className="field" type="number" min={0} placeholder="cool-down, s" value={cool || ""} onChange={(e) => setCool(Number(e.target.value))} /></div>
              <em>Locking the GPU clock (below its maximum boost) and pausing between runs remove most boost and heat noise from speed results. Recorded per run; runs with different settings aren't compared. Needs the clock helper from scripts/install.sh.</em></label>
            <label><span>Long-context suite (RULER-style)</span><select className="field select" value={ruler} onChange={(e) => setRuler(Number(e.target.value))}>
              <option value={0}>Off</option><option value={4}>4 samples per task and length (quick)</option><option value={8}>8 samples (standard)</option><option value={20}>20 samples (precise)</option></select>
              <em>Four synthetic tasks harder than one needle: multi-key retrieval with distractors, all values of a key, variable tracking, common words. Lengths 4k up to the run's context. Long contexts make it slow.</em></label>
            <label><span>Tasks</span><input className="field" value={tasks} onChange={(e) => setTasks(e.target.value)} placeholder="gsm8k,ifeval" /></label>
            <label><span>Questions per task (limit)</span><input className="field" type="number" min={1} max={5000} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
              <em>gsm8k at 100 takes a few minutes on a fast 9B setup.</em></label>
            <label className="ws-check"><input type="checkbox" checked={kern} onChange={(e) => setKern(e.target.checked)} />
              <span><b>GPU kernel profile</b><em>Traces every GPU kernel with NVIDIA Nsight Systems during short prefill and decode runs, and groups the time by operation (matrix multiply, attention, normalization…). llama.cpp runs only; needs a system where CUDA kernel tracing works (regular Linux; not WSL).</em></span></label>
          </div>
          <div className="xc-btns"><button className="btn small" onClick={save}>Save</button>{extra && <button className="btn small ghost" onClick={() => { setTasks(""); setKern(false); setMmluN(40); setRuler(0); setChunks(32); setFtext(""); setMargin(0); setClock(0); setCool(0); api.experimentProtocol(exp.id, null, false).then(onSaved); setOpen(false); }}>Back to the standard tests</button>}</div>
          {err && <div className="error"><AlertCircle /> {err}</div>}
        </div>
      )}
    </div>
  );
}

/* ---------- GPU kernels ---------- */

const KCOLORS: Record<string, string> = { "Matrix multiply (weights)": "var(--series-1)", "Attention": "var(--series-2)", "Linear attention / state": "var(--series-4)",
  "Normalization": "var(--series-3)" };

function KernelBars({ runs }: { runs: ExpRun[] }) {
  const rs = runs.filter((r) => r.results.kernels);
  if (!rs.length) return null;
  const cats = Array.from(new Set(rs.flatMap((r) => [...(r.results.kernels?.decode?.categories ?? []), ...(r.results.kernels?.prefill?.categories ?? [])].map((c) => c.category))));
  const bar = (k: KernelProfile | undefined) => k && (
    <div className="xbar-track stack">{k.categories.map((c) => c.share > 0.004 &&
      <div key={c.category} style={{ flex: c.share, background: KCOLORS[c.category] ?? "var(--split-other)" }} title={`${c.category}: ${(c.share * 100).toFixed(1)}% · ${c.ms.toFixed(1)} ms · ${c.calls} launches`} />)}</div>
  );
  return (
    <div className="xsplit2">
      <div className="mini-title">GPU time by operation <span className="muted">(Nsight Systems kernel trace; hover for details)</span></div>
      <div className="xlegend top">{cats.map((c) => <span key={c}><i className="sq" style={{ background: KCOLORS[c] ?? "var(--split-other)" }} />{c}</span>)}</div>
      {rs.map((r, i) => r.results.kernels!.available ? (
        <div key={r.id} className="kb-run">
          <div className="xs-name"><span className="xdot" style={{ background: colorOf(i) }} />{nameOf(r, runs.indexOf(r))}</div>
          <div className="kb-row"><span>Decode</span>{bar(r.results.kernels!.decode)}</div>
          <div className="kb-row"><span>Prefill</span>{bar(r.results.kernels!.prefill)}</div>
          <details className="kb-top"><summary>Top kernels (decode)</summary>
            <table className="data"><tbody>{(r.results.kernels!.decode?.top ?? []).slice(0, 8).map((k) => (
              <tr key={k.name}><td className="mono-cell">{k.name}</td><td>{k.category}</td><td className="r">{(k.share * 100).toFixed(1)}%</td><td className="r">{k.calls}×</td></tr>))}</tbody></table>
          </details>
        </div>
      ) : <div key={r.id} className="row-sub">{nameOf(r, runs.indexOf(r))}: {r.results.kernels!.note}</div>)}
    </div>
  );
}

/* ---------- Reproducibility ---------- */

function Repro({ runs }: { runs: ExpRun[] }) {
  const [cmdRun, setCmdRun] = useState<number | null>(null);
  const rows: { label: string; get: (p: Provenance) => string | null | undefined }[] = [
    { label: "Device", get: (p) => p.host.device ?? "–" },
    { label: "GPU", get: (p) => `${(p.gpu.name ?? "none").replace("NVIDIA GeForce ", "")} · ${p.gpu.vram_mb ? `${Math.round(p.gpu.vram_mb / 1024)} GB` : "?"}${p.gpu.pcie ? ` · ${p.gpu.pcie}` : ""}` },
    { label: "CPU · RAM", get: (p) => `${p.host.cpu} · ${p.host.threads} threads · ${Math.round(p.host.ram_mb / 1024)} GB` },
    { label: "llama.cpp", get: (p) => [p.llama_cpp.version, p.llama_cpp.commit].filter(Boolean).join(" @ ") },
    { label: "Model SHA-256", get: (p) => p.model.sha256?.slice(0, 16) + "…" },
    { label: "GPU driver / CUDA", get: (p) => `${p.gpu.driver ?? "?"} / toolkit ${p.gpu.cuda_toolkit ?? "?"}, driver API ${p.gpu.cuda_driver_api ?? "?"}` },
    { label: "Host", get: (p) => `${p.host.os}${p.host.wsl ? " (WSL2)" : ""} · ${p.host.kernel}` },
    { label: "MMLU-Pro data", get: (p) => `${p.data.mmlu_pro_sha256?.slice(0, 12)}… seed ${p.data.mmlu_seed}` },
    { label: "Blocky", get: (p) => `${p.blocky.commit ?? "?"}${p.blocky.dirty ? " (uncommitted changes)" : ""}` },
    { label: "Started", get: (p) => p.started_at },
  ];
  const withProv = runs.filter((r) => r.results.provenance);
  const sel = runs.find((r) => r.id === cmdRun) ?? withProv[0];
  return (
    <details className="xrepro">
      <summary><b>Reproducibility</b> <span className="row-sub">versions, file hashes and exact commands for every run{runs.some((r) => r.stats?.env_mismatch?.length) ? " · ⚠ some runs used a different environment" : ""}</span></summary>
      {withProv.length === 0 ? <div className="row-sub">{runs.some((r) => r.results.research_provenance) ? "Registered evaluator provenance, source hashes, environment and artifacts are available from each metric's evidence button." : "Runs made before provenance tracking don't have this. New runs record it automatically."}</div> : <>
        <div className="row-sub">Protocol: {withProv[0].results.provenance!.protocol}</div>
        <div className="table-wrap"><table className="data">
          <thead><tr><th />{runs.map((r, i) => <th key={r.id}>{nameOf(r, i)}</th>)}</tr></thead>
          <tbody>{rows.map((row) => (
            <tr key={row.label}><td className="mlabel">{row.label}</td>
              {runs.map((r) => { const p = r.results.provenance; return <td key={r.id} className="mono-cell">{p ? row.get(p) : "–"}</td>; })}</tr>
          ))}
          <tr><td className="mlabel">Differs from baseline</td>{runs.map((r, i) => <td key={r.id} className={r.stats?.env_mismatch?.length ? "warn-cell" : ""}>{i === 0 ? "–" : r.stats?.env_mismatch?.length ? r.stats.env_mismatch.join(", ") : "nothing"}</td>)}</tr>
          </tbody></table></div>
        {sel?.results.commands && <>
          <div className="xroof-bar"><span className="mini-title">Exact commands</span>
            <select className="field select small-select" value={sel.id} onChange={(e) => setCmdRun(Number(e.target.value))}>
              {withProv.map((r) => <option key={r.id} value={r.id}>{nameOf(r, runs.indexOf(r))}</option>)}</select></div>
          <pre className="xcmds">{Object.entries(sel.results.commands).map(([k, v]) => `# ${k}\n${v}`).join("\n\n")}</pre>
        </>}
      </>}
    </details>
  );
}

/* ---------- Hardware views ---------- */

const LIMIT_TEXT: Record<string, { short: string; fix: string }> = {
  "Possible VRAM bandwidth limit": { short: "Possible GPU memory limit", fix: "Estimated weight traffic is near nominal memory bandwidth. Confirm with a kernel profile before attributing the bottleneck." },
  "Possible RAM bandwidth limit": { short: "Possible system RAM limit", fix: "The weight-read model suggests RAM traffic may matter; these times are estimates, not measurements of individual operations." },
  "Bottleneck undetermined": { short: "Undetermined", fix: "Bandwidth utilization alone cannot distinguish compute, attention, transfers, or software overhead." },
  "Roofline estimate exceeded": { short: "Estimate exceeded", fix: "Check weight traffic and hardware assumptions; the simplified roofline is not explaining this run." },
};

function UseBar({ value, max, color, text }: { value: number; max: number; color: string; text: string }) {
  const p = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="xuse">
      <div className="xuse-track"><div style={{ width: `${p}%`, background: color }} /></div>
      <div className="xuse-text"><b>{((value / max) * 100).toFixed(0)}%</b> {text}</div>
    </div>
  );
}

/** One row per run: what limits it, and how much of the hardware's limit it actually uses. */
function LimitTable({ runs }: { runs: ExpRun[] }) {
  return (
    <div className="xlimit-table">
      <div className="xlt-head"><span>Run</span><span>Possible bottleneck (estimate)</span><span>Writing: share of estimated roof</span><span>Reading: share of nominal compute</span></div>
      {runs.map((r, i) => {
        const h = r.results.hw, d = h?.decode, p = h?.prefill, lim = h?.limiter;
        const t = lim ? LIMIT_TEXT[lim.what] : null;
        return (
          <div className="xlt-row" key={r.id}>
            <div className="xlt-run"><span className="xdot" style={{ background: colorOf(i) }} />{nameOf(r, i)}</div>
            <div className="xlt-lim">{lim ? <><span className={`xchip k${lim.what.length % 3}`}>{t?.short ?? lim.what}</span><em title={t?.fix}>{lim.evidence}</em></> : "–"}</div>
            <div>{(() => { const tg = r.results.speed?.tg ?? r.results.serving?.tg; return d?.roof_tps && tg ? <UseBar value={tg} max={d.roof_tps} color="var(--series-1)" text={`${n1(tg)} of ~${n1(d.roof_tps)} tok/s possible`} /> : "–"; })()}</div>
            <div>{p?.roof_tflops ? <UseBar value={p.achieved_tflops} max={p.roof_tflops} color="var(--series-3)" text={`${p.achieved_tflops} of ${p.roof_tflops} TFLOPS`} /> : "–"}</div>
          </div>
        );
      })}
      {runs.some((r) => r.results.hw?.projection) && (
        <div className="xlt-legend">
          {runs.map((r, i) => r.results.hw?.projection && (
            <div key={r.id}><span className="xchip k0">Projection</span>{nameOf(r, i)}: on a {r.results.hw.projection.target} ({r.results.hw.projection.memory}, {r.results.hw.projection.target_ram_gbps} GB/s), decode would reach about <b>{r.results.hw.projection.tg} tok/s</b> at the same {Math.round(r.results.hw.projection.bandwidth_share * 100)}% share of memory bandwidth measured here. Prompt reading is compute-bound and isn't projected.</div>
          ))}
        </div>
      )}
      {runs.some((r) => r.results.hw?.limiter) && (
        <div className="xlt-legend">
          {Array.from(new Set(runs.map((r) => r.results.hw?.limiter?.what).filter(Boolean) as string[])).map((w) => (
            <div key={w}><span className={`xchip k${w.length % 3}`}>{LIMIT_TEXT[w]?.short ?? w}</span>{LIMIT_TEXT[w]?.fix}</div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Where each generated token's time goes. Full-width bars on one shared scale, labels inside the segments. */
function TimeSplit({ runs }: { runs: ExpRun[] }) {
  const rows = runs.map((r, i) => ({ r, i })).filter(({ r }) => {
    const d = r.results.hw?.decode;
    return d && d.roof_util != null && d.roof_util <= 1 && Object.values(d.time_split_ms).every((v) => v != null);
  });
  const max = Math.max(...rows.map(({ r }) => r.results.hw!.decode!.ms_per_token), 1);
  const parts = [
    { k: "vram_reads" as const, label: "Estimated minimum GPU weight-read time", c: "var(--series-1)" },
    { k: "ram_reads" as const, label: "Estimated minimum RAM weight-read time", c: "var(--series-3)" },
    { k: "other" as const, label: "Unexplained remainder", c: "var(--split-other)" },
  ];
  return (
    <div className="xsplit2">
      <div className="xlegend top">{parts.map((p) => <span key={p.k}><i className="sq" style={{ background: p.c }} />{p.label}</span>)}</div>
      {rows.map(({ r, i }) => {
        const s = r.results.hw!.decode!.time_split_ms, tot = r.results.hw!.decode!.ms_per_token;
        return (
          <div className="xs-row" key={r.id}>
            <div className="xs-name"><span className="xdot" style={{ background: colorOf(i) }} />{nameOf(r, i)}<b>{tot.toFixed(1)} ms per token</b></div>
            <div className="xs-bar" style={{ width: `${(tot / max) * 100}%` }}>
              {parts.map((p) => {
                const v = s[p.k] ?? 0;
                if (v < 0.05) return null;
                return <div key={p.k} style={{ flex: v, background: p.c, color: p.k === "other" ? "var(--text)" : "#fff" }} title={`${p.label}: ${v.toFixed(1)} ms`}>{v / max > 0.06 ? `${v.toFixed(1)}` : ""}</div>;
              })}
            </div>
          </div>
        );
      })}
      <div className="row-sub">Only total time is measured. The segments estimate weight-read time from assumed traffic and bandwidth; the remainder is not a measured overhead or a promise of recoverable performance.</div>
    </div>
  );
}

/** Roofline for one run: attainable TFLOPS = min(peak compute, memory bandwidth × FLOP per byte). Log–log, clipped. */
function Roofline({ runs }: { runs: ExpRun[] }) {
  const withHw = runs.filter((r) => r.results.hw?.peaks.tflops);
  const [sel, setSel] = useState<number | null>(null);
  const run = withHw.find((r) => r.id === sel) ?? withHw[0];
  const h = run?.results.hw;
  if (!h?.peaks.tflops || !h.peaks.vram_gbps) return <div className="row-sub">Peak numbers for this GPU aren't in Blocky's table yet.</div>;
  const W = 620, H = 300, L = 48, R = 16, T = 16, B = 40;
  const x0 = Math.log10(0.5), x1 = Math.log10(5000), y0 = Math.log10(0.05), y1 = Math.log10(h.peaks.tflops * 2.5);
  const X = (ai: number) => L + ((Math.log10(ai) - x0) / (x1 - x0)) * (W - L - R);
  const Y = (tf: number) => T + (1 - (Math.log10(tf) - y0) / (y1 - y0)) * (H - T - B);
  const peak = h.peaks.tflops, bw = h.peaks.vram_gbps, ram = h.peaks.ram_gbps;
  const roofPath = (gbps: number, flat: boolean) => {
    const xs = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, (peak * 1000) / gbps].filter((x) => x >= 0.5 && x <= 5000).sort((a, b) => a - b);
    return xs.map((x, k) => `${k ? "L" : "M"}${X(x)},${Y(flat ? Math.min(peak, (gbps * x) / 1000) : (gbps * x) / 1000)}`).join(" ");
  };
  const i = runs.indexOf(run);
  const pts = [
    h.decode?.arithmetic_intensity && h.decode.achieved_tflops ? { x: h.decode.arithmetic_intensity, y: h.decode.achieved_tflops, name: "Writing", roof: Math.min(peak, (bw * h.decode.arithmetic_intensity) / 1000) } : null,
    h.prefill?.arithmetic_intensity ? { x: h.prefill.arithmetic_intensity, y: h.prefill.achieved_tflops, name: "Reading", roof: Math.min(peak, (bw * h.prefill.arithmetic_intensity) / 1000) } : null,
  ].filter(Boolean) as { x: number; y: number; name: string; roof: number }[];
  return (
    <div className="xroof">
      <div className="xroof-bar">
        <select className="field select small-select" value={run.id} onChange={(e) => setSel(Number(e.target.value))}>
          {withHw.map((r) => <option key={r.id} value={r.id}>{nameOf(r, runs.indexOf(r))}</option>)}
        </select>
        <span className="row-sub">The line is the most this GPU can do. Each dot is what the run achieved; the dashed gap below the line is unused hardware.</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Roofline chart">
        <defs><clipPath id="roofclip"><rect x={L} y={T} width={W - L - R} height={H - T - B} /></clipPath></defs>
        {[1, 10, 100, 1000].map((v) => <g key={v}><line x1={X(v)} x2={X(v)} y1={T} y2={H - B} className="grid" /><text x={X(v)} y={H - B + 15} textAnchor="middle" className="tick">{v}</text></g>)}
        {[0.1, 1, 10, 100].filter((v) => Math.log10(v) <= y1).map((v) => <g key={v}><line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} className="grid" /><text x={L - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text></g>)}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="axis">Work per byte of weights (FLOP/byte, log scale)</text>
        <text x={12} y={(T + H - B) / 2} textAnchor="middle" className="axis" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>TFLOPS (log)</text>
        <g clipPath="url(#roofclip)">
          <path d={roofPath(bw, true)} className="roof" />
          {ram && <path d={roofPath(ram, true)} className="roof ram" />}
          {pts.map((p) => <line key={p.name} x1={X(p.x)} x2={X(p.x)} y1={Y(p.roof)} y2={Y(p.y)} className="gap" />)}
        </g>
        {(() => {
          // Labels run along their slanted line (same angle on screen), placed above it.
          const ang = (Math.atan2(Y(1) - Y(10), X(10) - X(1)) * 180) / Math.PI;
          const at = (gbps: number, ai: number, dy: number, text: string) => (
            <text x={X(ai)} y={Y((gbps * ai) / 1000) + dy} className="roof-label" transform={`rotate(${-ang} ${X(ai)} ${Y((gbps * ai) / 1000) + dy})`}>{text}</text>
          );
          return <>
            {at(bw, 6, -8, `GPU memory speed limit · ${bw} GB/s`)}
            {ram && at(ram, 45, 15, `system RAM speed · ${ram} GB/s`)}
          </>;
        })()}
        <text x={W - R - 4} y={Y(peak) - 7} textAnchor="end" className="roof-label">compute limit · {peak} TFLOPS</text>
        {pts.map((p) => {
          const right = X(p.x) > W * 0.6;
          return (
            <g key={p.name}>
              <circle cx={X(p.x)} cy={Y(p.y)} r={6} fill={colorOf(i)} className="pt" />
              <text x={X(p.x) + (right ? -10 : 10)} y={Y(p.y) + 4} textAnchor={right ? "end" : "start"} className="pt-label">{p.name}: {((p.y / p.roof) * 100).toFixed(0)}% of the limit</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function Telemetry({ run }: { run: ExpRun }) {
  const s = run.results.series ?? [];
  if (s.length < 5) return null;
  const pts = (k: "gpu" | "mem_bw" | "power" | "vram_mb" | "sm_clock" | "cpu") => s.filter((x) => x[k] != null).map((x) => ({ x: x.t, y: k === "vram_mb" ? (x[k] as number) / 1024 : (x[k] as number) }));
  const charts = [
    { k: "gpu" as const, t: "GPU busy", u: "%", c: "var(--series-1)", max: 100 },
    { k: "mem_bw" as const, t: "Memory controller busy", u: "%", c: "var(--series-4)", max: 100 },
    { k: "sm_clock" as const, t: "SM clock", u: "MHz", c: "var(--series-3)" },
    { k: "cpu" as const, t: "CPU busy", u: "%", c: "var(--series-2)", max: 100 },
    { k: "vram_mb" as const, t: "VRAM used, all programs", u: "GB", c: "var(--series-2)" },
    { k: "power" as const, t: "GPU power", u: "W", c: "var(--series-3)" },
  ].filter((c) => pts(c.k).length > 2);
  const phases = PHASES.map((p) => { const xs = s.filter((x) => x.phase === p.id); return xs.length ? `${p.label} ${xs[0].t.toFixed(0)}–${xs[xs.length - 1].t.toFixed(0)} s` : null; }).filter(Boolean);
  return (
    <div className="xtl">
      <div className="mini-title">Sampled every 0.5 s <span className="muted">· {phases.join(" · ")}</span></div>
      <div className="tool-charts three">
        {charts.map((c) => <div key={c.k}><div className="mini-title">{c.t} <span className="muted">({c.u})</span></div>
          <LineChart compact points={pts(c.k)} yMax={c.max} height={56} color={c.c} unit={c.u} fmtX={(x) => `${x.toFixed(0)} s`} /></div>)}
      </div>
    </div>
  );
}

/* ---------- Page ---------- */

type ReportStage = "overview" | "setup" | "compare" | "validate" | "hardware" | "all";

export default function ExperimentsPage({ status }: { status: Status | null }) {
  const mode = useMode();
  const pageRef = useRef<HTMLDivElement>(null);
  const [stage, setStage] = useState<ReportStage>(() => {
    const query = new URLSearchParams(window.location.hash.split("?")[1]);
    const linked = query.get("stage") ?? (query.has("workshop") ? "setup" : query.has("roofline") ? "hardware" : null);
    return ["overview", "setup", "compare", "validate", "hardware", "all"].includes(linked ?? "") ? linked as ReportStage : "overview";
  });
  const [listLoading, setListLoading] = useState(true), [choicesLoading, setChoicesLoading] = useState(true), [detailLoading, setDetailLoading] = useState(false);
  useEffect(() => { pageRef.current?.scrollTo({ top: 0 }); }, [stage]);
  const [listReady, setListReady] = useState(false), [pending, setPending] = useState(false);
  const [choicesReady, setChoicesReady] = useState(false);
  const request = useRef(0);
  const [methods, setMethods] = useState<RegisteredMethod[]>([]);
  const [list, setList] = useState<Experiment[]>([]);
  const [id, setId] = useState<number | null>(null);
  const [exp, setExp] = useState<Experiment | null>(null);
  const [models, setModels] = useState<ModelFile[]>([]);
  const [newModel, setNewModel] = useState("");
  const [creating, setCreating] = useState(false);
  const [focus, setFocus] = useState<number | null>(null);
  const [metric, setMetric] = useState("tg");
  const [tab, setTabState] = useState<"report" | "canvas">(() => { if (window.location.hash.includes("view=canvas")) return "canvas"; if (window.location.hash.includes("stage=")) return "report"; try { return localStorage.getItem("blocky.exp.tab") === "canvas" ? "canvas" : "report"; } catch { return "report"; } });
  const setTab = (t: "report" | "canvas") => { setTabState(t); try { localStorage.setItem("blocky.exp.tab", t); } catch { /* private mode */ } };
  const [err, setErr] = useState<string | null>(null);

  const loadList = useCallback(() => api.experiments().then((l) => {
    setList(l); setListReady(true);
    const linked = Number(window.location.hash.match(/[?&]exp=(\d+)/)?.[1]);
    setId((cur) => l.some((e) => e.id === cur) ? cur : l.some((e) => e.id === linked) ? linked : l[0]?.id ?? null);
  }).catch((e) => setErr(e.message)).finally(() => setListLoading(false)), []);
  const loadChoices = useCallback(async () => {
    setChoicesLoading(true);
    const results = await Promise.allSettled([api.researchMethods(), api.models()]);
    setChoicesReady(results.every((result) => result.status === "fulfilled"));
    if (results[0].status === "fulfilled") {
      const m = results[0].value; setMethods(m);
      setNewModel((current) => current || (m[0] ? `method:${m[0].id}` : ""));
    } else setErr(String(results[0].reason));
    if (results[1].status === "fulfilled") {
      const d = results[1].value; setModels(d.models);
      setNewModel((current) => current || (d.models.some((m) => m.file === d.runtime.model) ? d.runtime.model ?? "" : d.models[0]?.file || ""));
    } else setErr(String(results[1].reason));
    setChoicesLoading(false);
  }, []);
  useEffect(() => { loadList(); loadChoices(); }, [loadList, loadChoices]);
  const load = useCallback(async () => {
    if (!id) return;
    const seq = ++request.current;
    try { const next = await api.experiment(id); if (seq === request.current) setExp(next); }
    catch (e) { if (seq === request.current) setErr((e as Error).message); }
    finally { if (seq === request.current) setDetailLoading(false); }
  }, [id]);
  useEffect(() => { setExp(null); setDetailLoading(!!id); load(); return () => { request.current++; }; }, [load]);

  const runs = exp?.runs_list ?? [];
  const running = runs.find((r) => r.status === "running") ?? runs.find((r) => r.status === "queued");
  useEffect(() => {
    const t = setInterval(load, running ? 2000 : 15000);
    return () => clearInterval(t);
  }, [running, load]);

  const done = useMemo(() => runs.filter((r) => r.status === "done"), [runs]);
  const baseline = done[0];
  const shown = done.find((r) => r.id === focus) ?? done[done.length - 1];

  const create = async () => {
    if (!newModel || pending) return;
    setErr(null); setPending(true);
    try {
      const m = models.find((x) => x.file === newModel);
      const { id: nid } = await api.experimentCreate(`${methods.find((m) => `method:${m.id}` === newModel)?.name ?? (m ? modelTitle(m) : newModel)}`, newModel);
      setCreating(false); setId(nid); setStage("overview"); setTab("report"); loadList();
    } catch (e) { setErr((e as Error).message); } finally { setPending(false); }
  };
  const runProfile = async (label: string, cfg: Partial<ExpConfig>) => {
    if (!exp || pending) return;
    setErr(null); setPending(true);
    try { await api.experimentRun(exp.id, label, cfg, baseline?.id); load(); setStage("compare"); window.scrollTo({ top: 0 }); } catch (e) { setErr((e as Error).message); } finally { setPending(false); }
  };
  const runSweep = async (grid: SweepGrid, repeats: number) => {
    if (!exp || pending) return;
    setErr(null); setPending(true);
    try { await api.experimentSweep(exp.id, grid as Record<string, unknown[]>, repeats); load(); setStage("compare"); window.scrollTo({ top: 0 }); } catch (e) { setErr((e as Error).message); } finally { setPending(false); }
  };
  const reloadModels = () => api.models().then((d) => setModels(d.models)).catch(() => {});
  // The canvas reuses the same change panel, but branches from whichever card you picked.
  const branchPanel = (parent: ExpRun, close: () => void) => exp && (
    parent.config.engine === "research" ? <MethodChangePanel methods={methods} models={models} parent={parent} onRun={(label, cfg) => { api.experimentRun(exp.id, label, cfg, parent.id).then(() => { load(); close(); }).catch((e) => setErr((e as Error).message)); }} /> : <ChangePanel base={parent.config} models={models} onModels={reloadModels}
      onRun={(label, cfg) => { setErr(null); api.experimentRun(exp.id, label, cfg, parent.id).then(() => { load(); close(); }).catch((e) => setErr((e as Error).message)); }}
      onSweep={(grid, reps) => { setErr(null); api.experimentSweep(exp.id, grid as Record<string, unknown[]>, reps, parent.id).then(() => { load(); close(); }).catch((e) => setErr((e as Error).message)); }} />
  );
  const L = exp?.limits;
  const queued = runs.filter((r) => r.status === "queued");

  return (
    <div className="page" ref={pageRef}>
      <div className={`page-inner wide ${tab === "canvas" ? "xfull" : ""}`}>
        <h1 className="page-title">Experiments</h1>
        <p className="page-sub">Start from a baseline, branch a change, and compare models or registered research methods. Inspect measurement contracts and uncertainty before drawing conclusions.</p>

        <div className="xbar">
          <select className="field select" value={id ?? ""} aria-label="Experiment" onChange={(e) => { setId(Number(e.target.value)); setFocus(null); }} disabled={list.length === 0}>
            {list.length === 0 && <option>{listLoading ? "Loading experiments..." : listReady ? "No experiments yet" : "Experiments unavailable"}</option>}
            {list.map((e) => <option key={e.id} value={e.id}>{e.name} · {e.id === exp?.id ? runs.length : e.runs} run{(e.id === exp?.id ? runs.length : e.runs) === 1 ? "" : "s"}</option>)}
          </select>
          <button className="btn small ghost" onClick={() => setCreating(!creating)}><Plus /> New experiment</button>
          <details className="flow-study-tools" open={mode === "pro" || undefined}><summary>Study tools and export</summary><div className="flow-toolbar">
          {exp && <>
            <a className="btn small ghost" href={`/api/experiments/${exp.id}/export.csv`}><Download /> CSV</a>
            <a className="btn small ghost" href={`/api/experiments/${exp.id}/export.json`} title="Everything, including hardware: import it on another Blocky device to compare"><Download /> JSON</a>
            <a className="btn small ghost" href={`/api/experiments/${exp.id}/bundle.zip`} title="Reproducibility bundle: experiment, provenance, commands, raw outputs, instrument checks, protocol document and a script that re-runs everything on another Blocky and compares"><Download /> Bundle</a>
            <button className="icon-btn danger" title="Delete experiment" onClick={() => { if (confirm(`Delete "${exp.name}" and all its runs?`)) api.experimentDelete(exp.id).then(() => { setId(null); setExp(null); loadList(); }); }}><Trash2 /></button>
          </>}
          <ImportRuns exp={exp} onDone={(nid) => { loadList(); if (nid !== exp?.id) setId(nid); else load(); }} />
          {exp && <RemoteRuns exp={exp} onImported={load} />}
          </div></details>
        </div>

        {!listLoading && listReady && (creating || list.length === 0) && (
          <div className="group pad xnew">
            <b>Which model do you want to study?</b>
            <div className="row-sub">Choose a local model or registered method. Measure a baseline, evaluate changes, then validate the comparison.</div>
            <div className="xnew-row">
              <select aria-label="Starting model or method" className="field select" disabled={choicesLoading} value={newModel} onChange={(e) => setNewModel(e.target.value)}>
                {!newModel && <option value="">{choicesLoading ? "Loading models and methods..." : "Choose a model or method"}</option>}
                {methods.map((m) => <option key={`method:${m.id}`} value={`method:${m.id}`}>{m.name} · registered method</option>)}
                {models.map((m) => <option key={m.file} value={m.file}>{modelTitle(m)} · {m.quant}</option>)}
              </select>
              <button className="btn" disabled={pending || choicesLoading || !newModel} onClick={create}><FlaskConical /> Start experiment</button>
            </div>
            {!choicesLoading && choicesReady && models.length === 0 && methods.length === 0 && <p>No models or registered methods are available. <a href="#/models">Add a model</a> or <a href="#/projects">develop and test a method</a> first.</p>}
          </div>
        )}
        {listLoading && <p role="status">Loading experiments...</p>}
        {detailLoading && <p role="status">Loading experiment...</p>}
        {err && <div className="error" role="alert" style={{ marginTop: 12 }}><AlertCircle /> {err} <button className="btn small ghost" onClick={() => { setErr(null); loadList(); loadChoices(); load(); }}>Retry loading</button></div>}

        {exp && L && (
          <>
            <details className="flow-device" open={mode === "pro" || undefined}><summary>Device and measured limits</summary><div className="xhwline"><Cpu size={14} />
              <span>{L.gpu?.replace("NVIDIA GeForce ", "") ?? "No GPU"}</span>
              <span>VRAM {(L.vram_mb / 1024).toFixed(0)} GB · {L.gpu_bandwidth_gbps ?? "?"} GB/s</span>
              <span>{L.gpu_tflops ?? "?"} TFLOPS <span className="muted">({L.gpu_tflops_kind ?? "unknown"})</span></span>
              <span>RAM {(L.ram_mb / 1024).toFixed(0)} GB · {L.ram_bandwidth_gbps ?? "?"} GB/s measured</span>
              <span>{L.cpu}</span>
            </div></details>

            {running && <Running run={running} onStop={() => api.experimentCancel(running.id).then(load)} />}
            {queued.filter((r) => r.id !== running?.id).length > 0 && (
              <div className="xqueue">
                <b>{queued.filter((r) => r.id !== running?.id).length} more in the queue</b>
                <span className="row-sub">{queued.filter((r) => r.id !== running?.id).map((r) => runTitle(r)).join(" · ")}</span>
                <span className="row-sub">Runs execute one at a time.</span>
                <button className="btn small ghost" onClick={() => api.experimentCancelQueue(exp.id).then(load)}>Cancel queue</button>
              </div>
            )}

            {exp.model.startsWith("method:") ? <details className="flow-advanced" open={mode === "pro" || undefined}><summary>Evaluator measurement contract</summary><p className="row-sub">Registered methods use their own evaluator contracts. Native profile phases are unavailable unless a compatible evaluator explicitly supplies them.</p></details> : <details className="flow-advanced" open={mode === "pro" || undefined}><summary>Measurement protocol and test settings</summary><TestProtocol exp={exp} onSaved={load} /></details>}
            <div className="xviewbar">
              <div className="segmented">
                <button className={tab === "report" ? "on" : ""} onClick={() => setTab("report")}>Report</button>
                <button className={tab === "canvas" ? "on" : ""} onClick={() => setTab("canvas")}>Canvas</button>
              </div>
              <span className="row-sub">{tab === "report" ? "Follow the steps below, or open the full report for every detail." : "The map of this experiment: what was tried, what it led to. Drag, zoom, annotate."}</span>
            </div>

            {tab === "canvas" ? <ResearchCanvas key={exp.id} exp={exp} runs={runs} branchPanel={branchPanel} onChanged={load}
              onRepeat={(r) => api.experimentRun(exp.id, "", r.config, r.parent_id ?? null).then(load).catch((e) => setErr((e as Error).message))}
              onDelete={(r) => api.experimentDeleteRun(r.id).then(load)} onCancel={(r) => api.experimentCancel(r.id).then(load)} /> : <>
            <Tabs value={baseline ? stage : "overview"} onChange={setStage} tabs={[{id:"overview",label:"Overview"}, ...(baseline ? [{id:"setup" as const,label:"Set up a change"},{id:"compare" as const,label:"Compare results",badge:done.length},{id:"validate" as const,label:"Repeat and validate"},{id:"hardware" as const,label:"Hardware"},{id:"all" as const,label:"Full report"}] : [])]} />
            {baseline && stage === "overview" && <div className="group pad flow-next"><h2>{done.length > 1 ? "Review your comparison" : "Baseline ready. Choose what to change."}</h2><p>{done.length} completed run{done.length === 1 ? "" : "s"}. Compare compatible measurements and repeat both setups before drawing conclusions.</p><div className="flow-toolbar"><button className="btn" onClick={() => setStage(done.length > 1 ? "compare" : "setup")}>{done.length > 1 ? "Compare results" : "Set up a change"}</button><button className="btn ghost" onClick={() => setStage("validate")}>Repeat and validate</button></div></div>}
            <section hidden={!!baseline && stage !== "overview" && stage !== "all"}>
            <div className="group-label">1 · Baseline{baseline ? `: ${baseline.results.method?.name ?? shortModel(baseline)}` : ""}</div>
            {baseline ? (
              <div className="group pad">
                <div className="xcap"><b>{baseline.results.method?.name ?? baseline.config.model.replace(".gguf", "")}</b><span className="row-sub" hidden={baseline.config.engine === "research"}>ctx {baseline.config.ctx} · KV {baseline.config.kv} · ngl {String(baseline.config.ngl)} · fa {baseline.config.fa}</span></div>
                <Profile run={baseline} />
              </div>
            ) : !running && (
              <div className="group pad xcta">
                <div><b>Measure the starting point</b>
                  <span className="row-sub">{exp.model.startsWith("method:") ? "Run the registered evaluator with its saved source and settings. Its supported metrics, logs and artifacts become the baseline; unsupported native phases stay unavailable." : "Load the model, evaluate quality, measure llama-bench speed and perplexity, and save reference logits and telemetry."}</span></div>
                <button className="btn" disabled={pending} onClick={() => runProfile("Baseline", {})}><Play /> {status?.busy ? "Queue baseline" : "Run baseline"}</button>
              </div>
            )}

            </section>
            {runs.filter((r) => !["done", "running", "queued"].includes(r.status)).map((r) => <div key={r.id} className="xfailed" role="alert"><AlertCircle size={14} />{r.label}: {r.status}{r.error ? ` - ${r.error}` : ""}<button className="icon-btn danger" aria-label={`Delete failed run ${r.id}`} onClick={() => api.experimentDeleteRun(r.id).then(load)}><Trash2 /></button></div>)}
            {baseline && (
              <>
                <section hidden={stage !== "setup" && stage !== "all"}>
                <div className="group-label">2 · Change one variable</div>
                {baseline.config.engine !== "research" && <div className="group pad"><ChangePanel base={baseline.config} models={models} onRun={runProfile} onSweep={runSweep} onModels={reloadModels} /></div>}
                {methods.length > 0 && <div className="group pad"><MethodChangePanel key={`${exp.id}-${baseline.id}`} methods={methods} models={models} parent={baseline} onRun={runProfile} /></div>}

                </section>
                <section hidden={stage !== "compare" && stage !== "all"}>
                <div className="group-label">3 · Results</div>
                <div className="group pad">
                  <Board runs={done} metric={metric} setMetric={setMetric} />
                  <div className="row-sub xnote">Speeds are medians; ± is descriptive spread (1.4826 × median absolute deviation), not a standard error. Differences are colored only when a statistical test supports them: paired McNemar for MMLU-Pro, or an approximate normal test using explicitly reported lm-eval standard errors, with multiple-comparison correction (Holm by default). Grey means significance is unavailable or not established; it does not prove equivalence. Different protocols and KL reference baselines are not compared. Protocol 3 serving throughput and energy include the full request, including prefill.</div>
                  {done.length > 1 && <MetricBars runs={done} metric={metric} />}
                  <NeedleGrid runs={done} />
                  <RulerTable runs={done} />
                  <Repro runs={done} />

                </div>

                </section>
                <section hidden={stage !== "validate" && stage !== "all"}>
                <ValidationPanel key={exp.id} exp={exp} runs={runs} onChanged={load} />
                <PreregPanel exp={exp} onSaved={load} />
                {done.length > 0 && <InstrumentPanel expId={exp.id} runsKey={done.map((r) => r.id).join(",")} />}
                {done.length > 0 && <details className="group pad xadv">
                  <summary><b>How many runs or questions do I need?</b> <span className="row-sub">sample sizes for the differences you want to detect, from this experiment's measured noise</span></summary>
                  <Planner runs={done} />
                </details>}

                </section>
                <section hidden={stage !== "hardware" && stage !== "all"}>
                {done.every((r) => r.config.engine === "research") && <p className="row-sub">Registered evaluators supply their own metrics. Native hardware breakdowns are unavailable unless explicitly measured; inspect metric evidence in Compare results for evaluator artifacts.</p>}
                <div className="group-label">4 · What limits each run</div>
                <div className="group pad"><LimitTable runs={done} /></div>

                <div className="group-label">Where the time goes, per generated token</div>
                <div className="group pad"><TimeSplit runs={done} />{done.some((r) => r.results.kernels) && <div className="kb-wrap"><KernelBars runs={done} /></div>}</div>

                <details className="group pad xadv" open={new URLSearchParams(window.location.hash.split("?")[1] ?? "").has("roofline") || undefined}>
                  <summary><b>Roofline</b> <span className="row-sub">the standard performance-engineering chart, one run at a time</span></summary>
                  <Roofline runs={done} />
                </details>

                <div className="group-label history-head">
                  <span>5 · Telemetry</span>
                  <select className="field select small-select" value={shown?.id ?? ""} onChange={(e) => setFocus(Number(e.target.value))}>
                    {done.map((r, i) => <option key={r.id} value={r.id}>{nameOf(r, i)}</option>)}
                  </select>
                </div>
                <div className="group pad">{shown && <Telemetry run={shown} />}
                  {shown?.results.energy && !shown.results.energy.primary && <div className="row-sub xnote">Power: {shown.results.energy.note ?? "no reading for this run."}</div>}
                  {shown?.results.energy?.primary && <div className="row-sub xnote">Energy measured on: {shown.results.energy.primary_label}, over {shown.results.energy.seconds} s of generation ({shown.results.energy.tokens} tokens) after {">"}6 s idle with the model loaded.</div>}
                </div>
                </section>
              </>
            )}
            </>}
          </>
        )}
      </div>
    </div>
  );
}
