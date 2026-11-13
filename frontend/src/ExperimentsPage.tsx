import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, Check, Cpu, Download, Server, FlaskConical, Gauge, Hammer, Loader2, MemoryStick, Play, Plus, Sparkles, Square, Trash2, Upload, Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { api, Device, DevicePreset, RemoteJob, Engine, ExpConfig, Experiment, ExpRun, KernelProfile, ModelFile, Provenance, Status, TorchStatus } from "./api";
import { LineChart, Term } from "./ui";
import ResearchCanvas from "./ResearchCanvas";
import WeightsWorkshop from "./WeightsWorkshop";
import EnginesPanel from "./EnginesPanel";
import { allMetrics, colorOf, delta, hwLabel, Metric, n1, nameOf, runTitle, shortModel } from "./expmetrics";

const PHASES = [
  { id: "loading", label: "Load", what: "memory layout, load time" },
  { id: "quality", label: "Quality", what: "MMLU-Pro, 560 questions" },
  { id: "longctx", label: "Long context", what: "needle in a haystack" },
  { id: "speed", label: "Speed", what: "llama-bench, 5 reps" },
  { id: "fidelity", label: "Fidelity", what: "PPL + KLD, 16k tokens" },
];

/* ---------- Pieces ---------- */

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
  const idx = PHASES.findIndex((p) => p.id === (run.phase === "lm_eval" ? "longctx" : run.phase));
  const mp = run.results.mmlu_progress, np = run.results.needle_progress;
  return (
    <div className="xrun">
      <div className="xrun-head"><Loader2 className="spin" /> <b>Profiling “{runTitle(run)}”</b><span className="row-sub">4–8 minutes · chat is paused</span>
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
  // Runs made before engines existed have no engine field: they ran on stock.
  const base = useMemo(() => ({ ...rawBase, engine: rawBase.engine ?? "stock" }), [rawBase]);
  const [engineList, setEngineList] = useState<Engine[]>([]);
  const [eng, setEng] = useState(false);
  const loadEngines = () => api.engines().then(setEngineList).catch(() => {});
  const [ckpts, setCkpts] = useState<TorchStatus["checkpoints"]>([]);
  const [presets, setPresets] = useState<Record<string, DevicePreset>>({});
  const [cpuThreads, setCpuThreads] = useState(0);
  useEffect(() => { loadEngines(); api.torchStatus().then((s) => setCkpts(s.checkpoints)).catch(() => {});
    api.devicePresets().then((p) => { setPresets(p.presets); setCpuThreads(p.cpu_threads); }).catch(() => {}); }, []);
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
    // Engines run different formats: vLLM takes checkpoints, llama.cpp takes GGUF. Keep the pair valid.
    if (patch.engine === "vllm" && !next.model.startsWith("hf:") && ckpts[0]) next.model = `hf:${ckpts[0].id}`;
    if (patch.engine && patch.engine !== "vllm" && next.model.startsWith("hf:")) next.model = base.model.startsWith("hf:") ? (family[0]?.file ?? models[0]?.file ?? next.model) : base.model;
    return next;
  });
  const inGrid = (k: keyof ExpConfig, v: unknown) => (grid[k] ?? []).some((x) => String(x ?? "") === String(v ?? ""));
  const toggle = (k: keyof ExpConfig, v: unknown) => setGrid((g) => {
    const cur = g[k] ?? [];
    const next = inGrid(k, v) ? cur.filter((x) => String(x ?? "") !== String(v ?? "")) : [...cur, v];
    return { ...g, [k]: next };
  });
  const Opt = ({ k, v, label }: { k: keyof ExpConfig; v: unknown; label: string }) => mode === "single"
    ? <button className={String(cfg[k] ?? "") === String(v ?? "") ? "on" : ""} onClick={() => set({ [k]: v } as Partial<ExpConfig>)}>{label}</button>
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
      {(mode === "sweep" ? (grid.engine ?? []).includes("vllm") : cfg.engine === "vllm") && (
        <div className="xc-row"><div><b>Checkpoint (vLLM)</b><span>vLLM runs original Hugging Face weights (16-bit). Download more in Internals → Checkpoints.</span></div>
          <div className="segmented wrap">{ckpts.length ? ckpts.map((c) => <Opt key={c.id} k="model" v={`hf:${c.id}`} label={`${c.repo} · ${(c.size_bytes / 2 ** 30).toFixed(1)} GB`} />)
            : <span className="row-sub" style={{ padding: "4px 10px" }}>No checkpoints yet</span>}</div></div>
      )}
      <div className="xc-row"><div><b>Target device</b><span>Simulate a cheaper device: CPU only, fewer cores, less RAM or VRAM. Capacity and cores are real limits; per-core speed and memory bandwidth aren't, so CPU presets add a projected decode speed for the target's memory.</span></div>
        <div className="segmented wrap"><Opt k="preset" v={null} label="This device" />{Object.entries(presets).map(([id, p]) => <Opt key={id} k="preset" v={id} label={p.label} />)}</div></div>
      <div className="xc-row"><div><b>Processor</b><span>CPU only runs with no GPU at all (-dev none); cores pins the run to the first N CPU threads</span></div>
        <div className="ws-opts">
          <div className="segmented"><Opt k="device" v="gpu" label="GPU" /><Opt k="device" v="cpu" label="CPU only" /></div>
          <div className="segmented"><Opt k="cores" v={null} label="All cores" />{CORES.filter((n) => !cpuThreads || n < cpuThreads).map((n) => <Opt key={n} k="cores" v={n} label={`${n}`} />)}</div>
        </div></div>
      <div className="xc-row"><div><b>RAM limit</b><span>The run fails if the model's CPU-side memory plus a minimal system wouldn't fit</span></div>
        <div className="segmented">{RAM_SIM.map((v) => <Opt key={String(v)} k="ram_limit_mb" v={v} label={v ? `${v / 1024} GB` : "Full"} />)}</div></div>
      <div className="xc-row"><div><b><Term k="kv">KV cache type</Term></b><span>-ctk / -ctv</span></div>
        <div className="segmented"><Opt k="kv" v="f16" label="f16" /><Opt k="kv" v="q8_0" label="q8_0" /><Opt k="kv" v="q4_0" label="q4_0" /></div></div>
      <div className="xc-row"><div><b><Term k="context">Context size</Term></b><span>-c</span></div>
        <div className="segmented">{CTX.map((c) => <Opt key={c} k="ctx" v={c} label={`${c / 1024}k`} />)}</div></div>
      <div className="xc-row"><div><b>VRAM limit</b><span>Simulates a smaller GPU (--fit-target)</span></div>
        <div className="segmented">{VRAM_SIM.map((v) => <Opt key={String(v)} k="vram_limit_mb" v={v} label={v ? `${v / 1024} GB` : "Full"} />)}</div></div>
      <div className="xc-row"><div><b><Term k="layers">GPU layers</Term></b><span>-ngl</span></div>
        <div className="segmented"><Opt k="ngl" v="auto" label="Auto" />{[24, 16, 8, 0].map((n) => <Opt key={n} k="ngl" v={n} label={String(n)} />)}</div></div>
      <div className="xc-row"><div><b><Term k="fa">Flash attention</Term></b><span>-fa</span></div>
        <div className="segmented"><Opt k="fa" v="auto" label="Auto" /><Opt k="fa" v="on" label="On" /><Opt k="fa" v="off" label="Off" /></div></div>
      {mode === "single" ? (
        <div className="xc-foot">
          <span className="row-sub">{changed.length === 0 ? "Pick one variable to change against the baseline."
            : changed.length === 1 ? "One variable changed: differences can be attributed to it." : `${changed.length} variables changed at once: effects will be confounded.`}</span>
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
              : <>{swept.map((k) => `${grid[k]!.length} ${k === "vram_limit_mb" ? "VRAM limits" : k === "model" ? "weight files" : k}`).join(" × ")}{repeats > 1 ? ` × ${repeats} repeats` : ""} = <b>{nRuns} runs</b>, about {Math.round(nRuns * 4.5)} min{nRuns > 32 ? " · limit is 32 per sweep" : ""}</>}
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

function Board({ runs, metric, setMetric }: { runs: ExpRun[]; metric: string; setMetric: (k: string) => void }) {
  const base = runs[0];
  const METRICS = allMetrics(runs);
  const groups = Array.from(new Set(METRICS.map((m) => m.group)));
  const hasData = (m: Metric) => runs.some((r) => m.get(r.results) != null);
  return (
    <div className="table-wrap xboard">
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
              {rows.filter(hasData).map((x) => (
                <tr key={x.key} className={metric === x.key ? "sel" : ""} onClick={() => setMetric(x.key)} title={x.note ?? "Click to chart"}>
                  <td className="mlabel">{x.term ? <Term k={x.term}>{x.label}</Term> : x.label}{x.unit && <span className="u"> {x.unit}</span>}</td>
                  {runs.map((r, i) => {
                    const v = x.get(r.results), sd = x.sd?.(r.results);
                    const repeats = runs.slice(1).filter((q) => q.label === "Repeat of baseline" && q.id !== r.id);
                    const bv = x.get(base.results);
                    const session = Math.max(0, ...repeats.map((q) => { const qv = x.get(q.results); return qv != null && bv != null ? Math.abs(qv - bv) : 0; }));
                    const d = i === 0 ? null : delta(x, { run: r, base }, session);
                    return (
                      <td key={r.id} className="r">
                        {v == null ? "–" : (x.fmt ?? n1)(v)}{sd ? <span className="sd"> ±{(x.fmt ?? n1)(sd)}</span> : null}
                        {d && <span className={`d ${d.tone}`} title={d.title}>{d.text}</span>}
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
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { setTasks(exp.protocol?.lm_eval?.tasks ?? ""); setLimit(exp.protocol?.lm_eval?.limit ?? 100); setKern(!!exp.protocol?.kernels); }, [exp.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const save = async () => {
    setErr(null);
    try { await api.experimentProtocol(exp.id, tasks.trim() ? { tasks, limit } : null, kern); setOpen(false); onSaved(); } catch (e) { setErr((e as Error).message); }
  };
  return (
    <div className="tp">
      <span className="row-sub"><b>Every run measures:</b> memory · MMLU-Pro (560) · long context (2k…ctx) · serving speed · llama-bench speed · perplexity & KLD · telemetry
        {cur ? <> · <b>lm-eval: {cur.tasks}</b> (limit {cur.limit})</> : ""}{exp.protocol?.kernels ? <> · <b>GPU kernel profile</b></> : ""}</span>
      <button className="btn small ghost" onClick={() => setOpen(!open)}>{open ? "Close" : "Add tests"}</button>
      {open && (
        <div className="tp-edit">
          <div className="row-sub">Run <b>lm-evaluation-harness</b> tasks too, the field's standard tool, against each run's exact server (greedy, seed 0). Use generative tasks such as <code>gsm8k</code>, <code>ifeval</code>, <code>bbh_cot_fewshot</code>, <code>minerva_math</code>. Multiple-choice log-likelihood tasks need prompt log-probs, which these servers don't provide; MMLU-Pro above already covers that style. Applies to runs queued from now on.</div>
          <div className="ws-grid">
            <label><span>Tasks</span><input className="field" value={tasks} onChange={(e) => setTasks(e.target.value)} placeholder="gsm8k,ifeval" /></label>
            <label><span>Questions per task (limit)</span><input className="field" type="number" min={1} max={5000} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
              <em>gsm8k at 100 takes a few minutes on a fast 9B setup.</em></label>
            <label className="ws-check"><input type="checkbox" checked={kern} onChange={(e) => setKern(e.target.checked)} />
              <span><b>GPU kernel profile</b><em>Traces every GPU kernel with NVIDIA Nsight Systems during short prefill and decode runs, and groups the time by operation (matrix multiply, attention, normalization…). llama.cpp runs only; needs a system where CUDA kernel tracing works (regular Linux; not WSL).</em></span></label>
          </div>
          <div className="xc-btns"><button className="btn small" onClick={save}>Save</button>{(cur || exp.protocol?.kernels) && <button className="btn small ghost" onClick={() => { setTasks(""); setKern(false); api.experimentProtocol(exp.id, null, false).then(onSaved); setOpen(false); }}>Remove extra tests</button>}</div>
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
      {withProv.length === 0 ? <div className="row-sub">Runs made before provenance tracking don't have this. New runs record it automatically.</div> : <>
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
  "VRAM bandwidth": { short: "GPU memory speed", fix: "Normal for local AI. Only a smaller/more-compressed model or a GPU with faster memory makes writing much faster." },
  "System-RAM bandwidth": { short: "System RAM speed", fix: "Part of the model doesn't fit on the GPU and is read from much slower RAM. Fit it on the GPU: smaller quantization, shorter context, or more VRAM." },
  "Software overhead": { short: "Software overhead", fix: "The hardware has spare capacity, but time goes to things other than reading weights (attention, kernel launches, cache decompression)." },
};

function UseBar({ value, max, color, text }: { value: number; max: number; color: string; text: string }) {
  const p = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="xuse">
      <div className="xuse-track"><div style={{ width: `${p}%`, background: color }} /></div>
      <div className="xuse-text"><b>{p.toFixed(0)}%</b> {text}</div>
    </div>
  );
}

/** One row per run: what limits it, and how much of the hardware's limit it actually uses. */
function LimitTable({ runs }: { runs: ExpRun[] }) {
  return (
    <div className="xlimit-table">
      <div className="xlt-head"><span>Run</span><span>What limits writing</span><span>Writing: share of the speed limit</span><span>Reading: share of the GPU's compute</span></div>
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
  const rows = runs.map((r, i) => ({ r, i })).filter(({ r }) => r.results.hw?.decode);
  const max = Math.max(...rows.map(({ r }) => r.results.hw!.decode!.ms_per_token), 1);
  const parts = [
    { k: "vram_reads" as const, label: "Reading weights from GPU memory", c: "var(--series-1)" },
    { k: "ram_reads" as const, label: "Reading weights from system RAM", c: "var(--series-3)" },
    { k: "other" as const, label: "Everything else", c: "var(--split-other)" },
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
      <div className="row-sub">The purple part is the physical minimum: the time to stream the weights once at the GPU's full memory speed. Everything beyond it is where speed could still be gained.</div>
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

export default function ExperimentsPage({ status }: { status: Status | null }) {
  const [list, setList] = useState<Experiment[]>([]);
  const [id, setId] = useState<number | null>(null);
  const [exp, setExp] = useState<Experiment | null>(null);
  const [models, setModels] = useState<ModelFile[]>([]);
  const [newModel, setNewModel] = useState("");
  const [creating, setCreating] = useState(false);
  const [focus, setFocus] = useState<number | null>(null);
  const [metric, setMetric] = useState("tg");
  const [tab, setTabState] = useState<"report" | "canvas">(() => { if (window.location.hash.includes("view=canvas")) return "canvas"; try { return localStorage.getItem("blocky.exp.tab") === "canvas" ? "canvas" : "report"; } catch { return "report"; } });
  const setTab = (t: "report" | "canvas") => { setTabState(t); try { localStorage.setItem("blocky.exp.tab", t); } catch { /* private mode */ } };
  const [err, setErr] = useState<string | null>(null);

  const loadList = useCallback(() => api.experiments().then((l) => { setList(l); setId((cur) => cur ?? l[0]?.id ?? null); }).catch(() => {}), []);
  useEffect(() => {
    loadList();
    api.models().then((d) => { setModels(d.models); setNewModel(d.runtime.model ?? d.models[0]?.file ?? ""); }).catch(() => {});
  }, [loadList]);
  const load = useCallback(() => { if (id) api.experiment(id).then(setExp).catch(() => {}); }, [id]);
  useEffect(() => { setExp(null); load(); }, [load]);

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
    setErr(null);
    try {
      const m = models.find((x) => x.file === newModel);
      const { id: nid } = await api.experimentCreate(`${m?.name ?? newModel}`, newModel);
      setCreating(false); setId(nid); loadList();
    } catch (e) { setErr((e as Error).message); }
  };
  const runProfile = async (label: string, cfg: Partial<ExpConfig>) => {
    if (!exp) return;
    setErr(null);
    try { await api.experimentRun(exp.id, label, cfg); load(); window.scrollTo({ top: 0 }); } catch (e) { setErr((e as Error).message); }
  };
  const runSweep = async (grid: SweepGrid, repeats: number) => {
    if (!exp) return;
    setErr(null);
    try { await api.experimentSweep(exp.id, grid as Record<string, unknown[]>, repeats); load(); window.scrollTo({ top: 0 }); } catch (e) { setErr((e as Error).message); }
  };
  const reloadModels = () => api.models().then((d) => setModels(d.models)).catch(() => {});
  // The canvas reuses the same change panel, but branches from whichever card you picked.
  const branchPanel = (parent: ExpRun, close: () => void) => exp && (
    <ChangePanel base={parent.config} models={models} onModels={reloadModels}
      onRun={(label, cfg) => { setErr(null); api.experimentRun(exp.id, label, cfg, parent.id).then(() => { load(); close(); }).catch((e) => setErr((e as Error).message)); }}
      onSweep={(grid, reps) => { setErr(null); api.experimentSweep(exp.id, grid as Record<string, unknown[]>, reps, parent.id).then(() => { load(); close(); }).catch((e) => setErr((e as Error).message)); }} />
  );
  const L = exp?.limits;
  const queued = runs.filter((r) => r.status === "queued");

  return (
    <div className="page">
      <div className={`page-inner wide ${tab === "canvas" ? "xfull" : ""}`}>
        <h1 className="page-title">Experiments</h1>
        <p className="page-sub">Start from a baseline, branch one change at a time, and compare setups or whole different models. Every run is measured the same way, with its noise shown, so you can tell real differences from luck.</p>

        <div className="xbar">
          <select className="field select" value={id ?? ""} onChange={(e) => { setId(Number(e.target.value)); setFocus(null); }} disabled={list.length === 0}>
            {list.length === 0 && <option>No experiments yet</option>}
            {list.map((e) => <option key={e.id} value={e.id}>{e.name} · {e.runs} run{e.runs === 1 ? "" : "s"}</option>)}
          </select>
          <button className="btn small ghost" onClick={() => setCreating(!creating)}><Plus /> New experiment</button>
          {exp && <>
            <a className="btn small ghost" href={`/api/experiments/${exp.id}/export.csv`}><Download /> CSV</a>
            <a className="btn small ghost" href={`/api/experiments/${exp.id}/export.json`} title="Everything, including hardware: import it on another Blocky device to compare"><Download /> JSON</a>
            <button className="icon-btn danger" title="Delete experiment" onClick={() => { if (confirm(`Delete "${exp.name}" and all its runs?`)) api.experimentDelete(exp.id).then(() => { setId(null); setExp(null); loadList(); }); }}><Trash2 /></button>
          </>}
          <ImportRuns exp={exp} onDone={(nid) => { loadList(); if (nid !== exp?.id) setId(nid); else load(); }} />
          {exp && <RemoteRuns exp={exp} onImported={load} />}
        </div>

        {(creating || list.length === 0) && (
          <div className="group pad xnew">
            <b>Which model do you want to study?</b>
            <div className="row-sub">An experiment follows one model: a baseline profile first, then variations of its setup.</div>
            <div className="xnew-row">
              <select className="field select" value={newModel} onChange={(e) => setNewModel(e.target.value)}>
                {models.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file} · {m.quant}</option>)}
              </select>
              <button className="btn" onClick={create}><FlaskConical /> Start experiment</button>
            </div>
          </div>
        )}
        {err && <div className="error" style={{ marginTop: 12 }}><AlertCircle /> {err}</div>}

        {exp && L && (
          <>
            <div className="xhwline"><Cpu size={14} />
              <span>{L.gpu?.replace("NVIDIA GeForce ", "") ?? "No GPU"}</span>
              <span>VRAM {(L.vram_mb / 1024).toFixed(0)} GB · {L.gpu_bandwidth_gbps ?? "?"} GB/s</span>
              <span>{L.gpu_tflops ?? "?"} TFLOPS <span className="muted">({L.gpu_tflops_kind ?? "unknown"})</span></span>
              <span>RAM {(L.ram_mb / 1024).toFixed(0)} GB · {L.ram_bandwidth_gbps ?? "?"} GB/s measured</span>
              <span>{L.cpu}</span>
            </div>

            {running && <Running run={running} onStop={() => api.experimentCancel(running.id).then(load)} />}
            {queued.filter((r) => r.id !== running?.id).length > 0 && (
              <div className="xqueue">
                <b>{queued.filter((r) => r.id !== running?.id).length} more in the queue</b>
                <span className="row-sub">{queued.filter((r) => r.id !== running?.id).map((r) => runTitle(r)).join(" · ")}</span>
                <span className="row-sub">· done in about {Math.round(queued.length * 4.5)} min</span>
                <button className="btn small ghost" onClick={() => api.experimentCancelQueue(exp.id).then(load)}>Cancel queue</button>
              </div>
            )}

            <TestProtocol exp={exp} onSaved={load} />
            <div className="xviewbar">
              <div className="segmented">
                <button className={tab === "report" ? "on" : ""} onClick={() => setTab("report")}>Report</button>
                <button className={tab === "canvas" ? "on" : ""} onClick={() => setTab("canvas")}>Canvas</button>
              </div>
              <span className="row-sub">{tab === "report" ? "Every number, in order: baseline → change → results → hardware." : "The map of this experiment: what was tried, what it led to. Drag, zoom, annotate."}</span>
            </div>

            {tab === "canvas" ? <ResearchCanvas key={exp.id} exp={exp} runs={runs} branchPanel={branchPanel} onChanged={load}
              onRepeat={(r) => api.experimentRun(exp.id, "", r.config, r.parent_id ?? null).then(load).catch((e) => setErr((e as Error).message))}
              onDelete={(r) => api.experimentDeleteRun(r.id).then(load)} onCancel={(r) => api.experimentCancel(r.id).then(load)} /> : <>
            <div className="group-label">1 · Baseline{baseline ? `: ${shortModel(baseline)}` : ""}</div>
            {baseline ? (
              <div className="group pad">
                <div className="xcap"><b>{baseline.config.model.replace(".gguf", "")}</b><span className="row-sub">ctx {baseline.config.ctx} · KV {baseline.config.kv} · ngl {String(baseline.config.ngl)} · fa {baseline.config.fa}</span></div>
                <Profile run={baseline} />
              </div>
            ) : !running && (
              <div className="group pad xcta">
                <div><b>Measure the starting point</b>
                  <span className="row-sub">Load (memory layout from llama.cpp's own log) → MMLU-Pro, 560 questions, 5-shot log-likelihood → llama-bench decode and prefill at 0 and 4k context, 5 repetitions (median, robust spread) → perplexity on 16k tokens of WikiText-2, saving logits for the KL divergence of later runs. GPU and CPU telemetry throughout.</span></div>
                <button className="btn" onClick={() => runProfile("Baseline", {})}><Play /> {status?.busy ? "Queue baseline" : "Run baseline"}</button>
              </div>
            )}

            {baseline && (
              <>
                <div className="group-label">2 · Change one variable</div>
                <div className="group pad"><ChangePanel base={baseline.config} models={models} onRun={runProfile} onSweep={runSweep} onModels={reloadModels} /></div>

                <div className="group-label">3 · Results</div>
                <div className="group pad">
                  <Board runs={done} metric={metric} setMetric={setMetric} />
                  <div className="row-sub xnote">Speeds are the median of the repetitions; ± is the robust spread (1.4826 × median absolute deviation). A difference is colored only when it is bigger than the noise: 2σ for speeds, a paired McNemar test (p &lt; 0.05) for MMLU-Pro. Grey = not significant. If you add a “Repeat this setup” run, changes must also exceed twice the drift between the two repeats (other programs using the same GPU add noise). KL divergence measures how far each token's probability distribution drifted from the baseline's.</div>
                  {done.length > 1 && <MetricBars runs={done} metric={metric} />}
                  <NeedleGrid runs={done} />
                  <Repro runs={done} />
                  {runs.filter((r) => !["done", "running", "queued"].includes(r.status)).map((r) => (
                    <div key={r.id} className="xfailed"><AlertCircle size={14} /> “{r.label}” {r.status}{r.error ? `: ${r.error}` : ""}
                      <button className="icon-btn danger" onClick={() => api.experimentDeleteRun(r.id).then(load)}><Trash2 /></button></div>
                  ))}
                </div>

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
              </>
            )}
            </>}
          </>
        )}
      </div>
    </div>
  );
}
