import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, Check, Cpu, FlaskConical, Gauge, Loader2, MemoryStick, Play, Plus, Sparkles, Square, Target, Trash2, Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { api, ExpConfig, Experiment, ExpResults, ExpRun, ModelFile, Status } from "./api";
import { LineChart, Term } from "./ui";

/* ---------- Metrics shown on the compare board (direction decides green/red) ---------- */

type Metric = { key: string; group: string; label: string; term?: string; unit: string; better: "higher" | "lower"; get: (r: ExpResults) => number | null | undefined; fmt?: (v: number) => string };
const n1 = (v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(1));
const METRICS: Metric[] = [
  { key: "tg", group: "Speed", label: "Writing", term: "tg", unit: "tok/s", better: "higher", get: (r) => r.speed?.tg },
  { key: "tg_long", group: "Speed", label: "Writing, long chat (4k)", unit: "tok/s", better: "higher", get: (r) => r.speed?.tg_d4096 },
  { key: "pp", group: "Speed", label: "Reading", term: "pp", unit: "tok/s", better: "higher", get: (r) => r.speed?.pp, fmt: (v) => v.toFixed(0) },
  { key: "ttft", group: "Speed", label: "First word after", term: "ttft", unit: "ms", better: "lower", get: (r) => r.ttft_ms, fmt: (v) => v.toFixed(0) },
  { key: "quality", group: "Quality", label: "Fresh test score", unit: "%", better: "higher", get: (r) => (r.quality_score != null ? r.quality_score * 100 : null), fmt: (v) => v.toFixed(0) },
  { key: "same_top", group: "Quality", label: "Same top word as baseline", unit: "%", better: "higher", get: (r) => r.same_top, fmt: (v) => v.toFixed(1) },
  { key: "ppl", group: "Quality", label: "Perplexity", unit: "", better: "lower", get: (r) => r.ppl, fmt: (v) => v.toFixed(2) },
  { key: "vram", group: "Memory", label: "GPU memory used", term: "vram", unit: "GB", better: "lower", get: (r) => (r.memory?.vram_used_mb ?? r.telemetry?.quality?.vram_max_mb ?? 0) / 1024 || null, fmt: (v) => v.toFixed(2) },
  { key: "ram", group: "Memory", label: "Model in system RAM", unit: "GB", better: "lower", get: (r) => (r.memory ? Object.values(r.memory.ram).reduce((a, b) => a + b, 0) / 1024 : null), fmt: (v) => v.toFixed(2) },
  { key: "layers", group: "Memory", label: "Layers on GPU", term: "layers", unit: "%", better: "higher", get: (r) => (r.memory?.layers ? (r.memory.layers.gpu / r.memory.layers.total) * 100 : null), fmt: (v) => v.toFixed(0) },
  { key: "load", group: "Memory", label: "Load time", unit: "s", better: "lower", get: (r) => r.load_seconds },
  { key: "power", group: "Energy", label: "GPU power while writing", unit: "W", better: "lower", get: (r) => r.telemetry?.speed?.power_avg, fmt: (v) => v.toFixed(0) },
  { key: "tpj", group: "Energy", label: "Efficiency", term: "tpj", unit: "tok/J", better: "higher", get: (r) => (r.speed?.tg && r.telemetry?.speed?.power_avg ? r.speed.tg / r.telemetry.speed.power_avg : null), fmt: (v) => v.toFixed(2) },
  { key: "hw", group: "Hardware", label: "Share of hardware limit", unit: "%", better: "higher", get: (r) => (r.bottleneck?.efficiency != null ? r.bottleneck.efficiency * 100 : null), fmt: (v) => v.toFixed(0) },
];

const PHASES = [
  { id: "loading", label: "Loading", what: "memory layout & load time" },
  { id: "quality", label: "Quality", what: "40 fresh questions" },
  { id: "speed", label: "Speed", what: "reading, writing, long context" },
  { id: "fidelity", label: "Fidelity", what: "perplexity & drift vs baseline" },
];

function delta(m: Metric, v: number | null | undefined, base: number | null | undefined) {
  if (v == null || base == null || base === 0) return null;
  const d = (v - base) / Math.abs(base);
  if (Math.abs(d) < 0.02) return { text: "same", tone: "same" };
  const good = m.better === "higher" ? d > 0 : d < 0;
  return { text: `${d > 0 ? "+" : ""}${(d * 100).toFixed(0)}%`, tone: good ? "good" : "bad" };
}

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
  const r = run.results;
  const tg = r.speed?.tg, power = r.telemetry?.speed?.power_avg;
  return (
    <div className="xt-grid">
      <Tile icon={Gauge} color="#7c5cff" label={<Term k="tg">Writing speed</Term>} value={tg ? n1(tg) : "–"} unit="tok/s"
        sub={r.speed?.tg_d4096 ? `${n1(r.speed.tg_d4096)} tok/s with 4k words of chat history` : undefined} />
      <Tile icon={Sparkles} color="#ff5c9d" label="Fresh test score" value={r.quality_score != null ? `${Math.round(r.quality_score * 100)}` : "–"} unit="%"
        sub={r.quality ? Object.entries(r.quality).map(([k, v]) => `${k} ${v.correct}/${v.total}`).join(" · ") : undefined} />
      <Tile icon={MemoryStick} color="#22d3ee" label={<Term k="vram">GPU memory</Term>} value={r.memory?.vram_used_mb ? (r.memory.vram_used_mb / 1024).toFixed(1) : "–"} unit="GB"
        sub={r.memory?.layers ? `${r.memory.layers.gpu}/${r.memory.layers.total} layers on GPU · loads in ${r.load_seconds}s` : undefined} />
      <Tile icon={Zap} color="#ffa94d" label={<Term k="tpj">Efficiency</Term>} value={tg && power ? (tg / power).toFixed(2) : "–"} unit="tok/J"
        sub={power ? `${power.toFixed(0)} W while writing · peak ${r.telemetry?.speed?.temp_max ?? "–"}°C`
          : `The GPU driver doesn't report power here${r.telemetry?.speed?.temp_max ? ` · peak ${r.telemetry.speed.temp_max}°C` : ""}`} />
    </div>
  );
}

function Running({ run, onStop }: { run: ExpRun; onStop: () => void }) {
  const idx = PHASES.findIndex((p) => p.id === run.phase);
  return (
    <div className="xrun">
      <div className="xrun-head"><Loader2 className="spin" /> <b>Profiling “{run.label}”</b><span className="row-sub">about 4–6 minutes · chat is paused</span>
        <button className="btn small ghost" onClick={onStop}><Square /> Stop</button></div>
      <div className="xsteps">
        {PHASES.map((p, i) => (
          <div key={p.id} className={`xstep ${i < idx ? "done" : i === idx ? "now" : ""}`}>
            <span>{i < idx ? <Check size={13} /> : i + 1}</span><div><b>{p.label}</b><em>{p.what}</em></div>
          </div>
        ))}
      </div>
      {run.results.quality && <div className="row-sub">So far: {Object.entries(run.results.quality).map(([k, v]) => `${k} ${v.correct}/${v.total}`).join(" · ")}</div>}
    </div>
  );
}

const CTX = [4096, 8192, 16384, 32768, 65536];
const VRAM_SIM = [null, 6144, 4096, 2048];

function ChangePanel({ base, models, onRun, disabled }: { base: ExpConfig; models: ModelFile[]; onRun: (label: string, cfg: Partial<ExpConfig>) => void; disabled: boolean }) {
  const [cfg, setCfg] = useState<ExpConfig>(base);
  useEffect(() => setCfg(base), [base]);
  const baseModel = models.find((m) => m.file === base.model);
  const family = models.filter((m) => baseModel && m.architecture === baseModel.architecture && m.layers === baseModel.layers);
  const changed = (Object.keys(cfg) as (keyof ExpConfig)[]).filter((k) => String(cfg[k] ?? "") !== String(base[k] ?? ""));
  const set = (patch: Partial<ExpConfig>) => setCfg((c) => ({ ...c, ...patch }));
  const Opt = ({ k, v, label }: { k: keyof ExpConfig; v: unknown; label: string }) => (
    <button className={String(cfg[k] ?? "") === String(v ?? "") ? "on" : ""} onClick={() => set({ [k]: v } as Partial<ExpConfig>)}>{label}</button>
  );
  return (
    <div className="xchange">
      <div className="xc-row"><div><b>Model version</b><span>Same model, different <Term k="quant">compression</Term>{family.length < 2 && <> · make more in <a href="#/models?tab=compress">Models → Compress</a></>}</span></div>
        <div className="segmented wrap">{family.map((m) => <Opt key={m.file} k="model" v={m.file} label={`${m.quant} · ${(m.size_bytes / 2 ** 30).toFixed(1)} GB`} />)}</div></div>
      <div className="xc-row"><div><b><Term k="kv">Context memory precision</Term></b><span>Smaller = less VRAM, maybe less accurate</span></div>
        <div className="segmented"><Opt k="kv" v="f16" label="f16" /><Opt k="kv" v="q8_0" label="q8_0" /><Opt k="kv" v="q4_0" label="q4_0" /></div></div>
      <div className="xc-row"><div><b><Term k="context">Context size</Term></b><span>How much text it can hold</span></div>
        <div className="segmented">{CTX.map((c) => <Opt key={c} k="ctx" v={c} label={`${c / 1024}k`} />)}</div></div>
      <div className="xc-row"><div><b>Simulate a smaller GPU</b><span>Test how it would run on weaker hardware</span></div>
        <div className="segmented">{VRAM_SIM.map((v) => <Opt key={String(v)} k="vram_limit_mb" v={v} label={v ? `${v / 1024} GB` : "Full"} />)}</div></div>
      <div className="xc-row"><div><b><Term k="layers">GPU layers</Term></b><span>Force some layers onto the CPU</span></div>
        <div className="segmented"><Opt k="ngl" v="auto" label="Auto" />{[24, 16, 8, 0].map((n) => <Opt key={n} k="ngl" v={n} label={String(n)} />)}</div></div>
      <div className="xc-row"><div><b><Term k="fa">Flash attention</Term></b><span>Faster attention math</span></div>
        <div className="segmented"><Opt k="fa" v="auto" label="Auto" /><Opt k="fa" v="on" label="On" /><Opt k="fa" v="off" label="Off" /></div></div>
      <div className="xc-foot">
        <span className="row-sub">{changed.length === 0 ? "Pick what to change. Changing one thing at a time makes results clear."
          : changed.length === 1 ? "Good: one change, so any difference is caused by it." : `${changed.length} changes at once: differences can't be attributed to one of them.`}</span>
        <button className="btn" disabled={disabled || changed.length === 0} onClick={() => onRun("", cfg)}><Play /> Profile this change</button>
      </div>
    </div>
  );
}

function Board({ runs }: { runs: ExpRun[] }) {
  const done = runs.filter((r) => r.status === "done");
  const base = done[0];
  const [metric, setMetric] = useState("tg");
  if (!base) return null;
  const groups = Array.from(new Set(METRICS.map((m) => m.group)));
  const m = METRICS.find((x) => x.key === metric)!;
  const vals = done.map((r) => m.get(r.results) ?? 0);
  const max = Math.max(...vals, 1e-9);
  return (
    <>
      <div className="table-wrap xboard">
        <table className="data">
          <thead>
            <tr><th />{groups.map((g) => <th key={g} colSpan={METRICS.filter((x) => x.group === g).length} className="grp">{g}</th>)}</tr>
            <tr><th>Run</th>{METRICS.map((x) => <th key={x.key} className={`r ${metric === x.key ? "sel" : ""}`} onClick={() => setMetric(x.key)} title="Click to chart">{x.label}{x.unit && <span className="u"> {x.unit}</span>}</th>)}</tr>
          </thead>
          <tbody>
            {done.map((r, i) => (
              <tr key={r.id}>
                <td className="run-label">{i === 0 ? <b>{r.label}</b> : r.label}</td>
                {METRICS.map((x) => {
                  const v = x.get(r.results);
                  const d = i === 0 ? null : delta(x, v, x.get(base.results));
                  const best = done.length > 1 && v != null && v === (x.better === "higher" ? Math.max : Math.min)(...done.map((q) => x.get(q.results) ?? (x.better === "higher" ? -Infinity : Infinity)));
                  return (
                    <td key={x.key} className={`r ${best ? "best" : ""} ${metric === x.key ? "sel" : ""}`}>
                      {v == null ? "–" : (x.fmt ?? n1)(v)}
                      {d && <span className={`d ${d.tone}`}>{d.text}</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="xchart">
        <div className="mini-title">{m.label} {m.unit && <span className="muted">({m.unit}, {m.better} is better)</span>}</div>
        {done.map((r, i) => (
          <div className="memo-line" key={r.id}>
            <span title={r.label}>{r.label}</span>
            <div className="memo-track"><div style={{ width: `${((vals[i] || 0) / max) * 100}%`, background: i === 0 ? "var(--series-3)" : "var(--series-1)" }} /></div>
            <b>{vals[i] ? (m.fmt ?? n1)(vals[i]) : "–"}</b>
          </div>
        ))}
      </div>
      <div className="row-sub">Green = better than the baseline, red = worse, bold = best in its column. Differences under 2% show as “same”. Click a column to chart it.</div>
    </>
  );
}

function Bottlenecks({ run }: { run: ExpRun }) {
  const b = run.results.bottleneck;
  const s = run.results.series ?? [];
  if (!b) return null;
  const pct = b.efficiency != null ? Math.min(100, b.efficiency * 100) : null;
  const pts = (k: "gpu" | "mem_bw" | "power" | "vram_mb") => s.filter((x) => x[k] != null).map((x) => ({ x: x.t, y: k === "vram_mb" ? (x[k] as number) / 1024 : (x[k] as number) }));
  return (
    <div className="xbn">
      {b.ceiling_tps && run.results.speed?.tg && (
        <div className="xceil">
          <div className="xceil-head"><Target /> <b>How close to the hardware limit?</b></div>
          <div className="xceil-bar"><div style={{ width: `${pct}%` }} /><span style={{ left: "100%" }} /></div>
          <div className="xceil-legend">
            <span>You get <b>{n1(run.results.speed.tg)} tok/s</b></span>
            <span>Physical limit ≈ <b>{n1(b.ceiling_tps)} tok/s</b> ({b.gpu_bandwidth_gbps} GB/s ÷ {b.gb_per_token} GB read per word)</span>
          </div>
        </div>
      )}
      <div className="xfind">
        {b.findings.map((f, i) => (
          <div key={i} className={`xf ${f.severity}`}>
            <b>{f.severity === "major" ? "Main bottleneck: " : ""}{f.title}</b>
            <span>{f.text}</span>
          </div>
        ))}
        {b.findings.length === 0 && <div className="row-sub">No clear bottleneck detected.</div>}
      </div>
      {s.length > 4 && (
        <div className="xtl">
          <div className="mini-title">During the profile <span className="muted">(phases: {PHASES.map((p) => p.label).join(" → ")})</span></div>
          <div className="tool-charts">
            <div><div className="mini-title">GPU load <span className="muted">(%)</span></div><LineChart compact points={pts("gpu")} yMax={100} height={60} color="var(--series-1)" unit="%" fmtX={(x) => `${x.toFixed(0)} s`} /></div>
            <div><div className="mini-title">Memory-controller load <span className="muted">(%)</span></div><LineChart compact points={pts("mem_bw")} yMax={100} height={60} color="var(--series-4)" unit="%" fmtX={(x) => `${x.toFixed(0)} s`} /></div>
            {pts("power").length > 2 && <div><div className="mini-title">Power <span className="muted">(W)</span></div><LineChart compact points={pts("power")} height={60} color="var(--series-3)" unit="W" fmtX={(x) => `${x.toFixed(0)} s`} /></div>}
            <div><div className="mini-title">GPU memory <span className="muted">(GB)</span></div><LineChart compact points={pts("vram_mb")} height={60} color="var(--series-2)" unit="GB" fmtX={(x) => `${x.toFixed(0)} s`} /></div>
          </div>
        </div>
      )}
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
  const [err, setErr] = useState<string | null>(null);

  const loadList = useCallback(() => api.experiments().then((l) => { setList(l); setId((cur) => cur ?? l[0]?.id ?? null); }).catch(() => {}), []);
  useEffect(() => {
    loadList();
    api.models().then((d) => { setModels(d.models); setNewModel(d.runtime.model ?? d.models[0]?.file ?? ""); }).catch(() => {});
  }, [loadList]);
  const load = useCallback(() => { if (id) api.experiment(id).then(setExp).catch(() => {}); }, [id]);
  useEffect(() => { setExp(null); load(); }, [load]);

  const runs = exp?.runs_list ?? [];
  const running = runs.find((r) => r.status === "running" || r.status === "queued");
  useEffect(() => {
    const t = setInterval(load, running ? 2000 : 15000);
    return () => clearInterval(t);
  }, [running, load]);

  const baseline = runs.find((r) => r.status === "done");
  const shown = useMemo(() => runs.find((r) => r.id === focus) ?? [...runs].reverse().find((r) => r.status === "done"), [runs, focus]);

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

  const steps = [
    { n: 1, label: "Baseline", done: !!baseline },
    { n: 2, label: "Change one thing", done: runs.filter((r) => r.status === "done").length > 1 },
    { n: 3, label: "Compare", done: runs.filter((r) => r.status === "done").length > 1 },
    { n: 4, label: "Bottlenecks", done: !!baseline?.results.bottleneck },
  ];

  return (
    <div className="page">
      <div className="page-inner wide">
        <h1 className="page-title">Experiments</h1>
        <p className="page-sub">Profile a model on this hardware, change one thing, and see exactly what got better or worse, and what's holding it back.</p>

        <div className="xbar">
          <select className="field select" value={id ?? ""} onChange={(e) => { setId(Number(e.target.value)); setFocus(null); }} disabled={list.length === 0}>
            {list.length === 0 && <option>No experiments yet</option>}
            {list.map((e) => <option key={e.id} value={e.id}>{e.name} · {e.runs} run{e.runs === 1 ? "" : "s"}</option>)}
          </select>
          <button className="btn small ghost" onClick={() => setCreating(!creating)}><Plus /> New experiment</button>
          {exp && <button className="icon-btn danger" title="Delete experiment" onClick={() => { if (confirm(`Delete "${exp.name}" and all its runs?`)) api.experimentDelete(exp.id).then(() => { setId(null); setExp(null); loadList(); }); }}><Trash2 /></button>}
        </div>

        {(creating || list.length === 0) && (
          <div className="group pad xnew">
            <b>Which model do you want to study?</b>
            <div className="row-sub">An experiment follows one model: first a baseline profile, then variations of its setup.</div>
            <div className="xnew-row">
              <select className="field select" value={newModel} onChange={(e) => setNewModel(e.target.value)}>
                {models.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file} · {m.quant}</option>)}
              </select>
              <button className="btn" onClick={create}><FlaskConical /> Start experiment</button>
            </div>
          </div>
        )}
        {err && <div className="error" style={{ marginTop: 12 }}><AlertCircle /> {err}</div>}

        {exp && (
          <>
            <div className="xstepper">
              {steps.map((s) => (
                <a key={s.n} href={`#step${s.n}`} onClick={(e) => { e.preventDefault(); document.getElementById(`step${s.n}`)?.scrollIntoView({ behavior: "smooth" }); }}
                  className={s.done ? "done" : ""}><span>{s.done ? <Check size={13} /> : s.n}</span>{s.label}</a>
              ))}
              <span className="xhw"><Cpu size={14} /> {exp.limits?.gpu?.replace("NVIDIA GeForce ", "") ?? "CPU"} · {exp.limits?.gpu_bandwidth_gbps ?? "?"} GB/s VRAM · RAM {exp.limits?.ram_bandwidth_gbps ?? "?"} GB/s</span>
            </div>

            {running && <Running run={running} onStop={() => api.experimentCancel(running.id).then(load)} />}

            <div id="step1" className="group-label">1 · Baseline</div>
            {baseline ? (
              <div className="group pad">
                <div className="xcap"><b>{baseline.label}</b><span className="row-sub">{baseline.config.model.replace(".gguf", "")} · ctx {baseline.config.ctx / 1024}k · KV {baseline.config.kv} · GPU layers {String(baseline.config.ngl)}</span></div>
                <Profile run={baseline} />
              </div>
            ) : !running && (
              <div className="group pad xcta">
                <div><b>Measure the starting point</b>
                  <span className="row-sub">One profile measures memory layout, load time, quality on 40 fresh questions, reading/writing speed (also with a long chat), perplexity, power and temperature, then explains what limits it. About 5 minutes.</span></div>
                <button className="btn" disabled={!!status?.busy} onClick={() => runProfile("Baseline", {})}><Play /> Run baseline profile</button>
              </div>
            )}

            {baseline && (
              <>
                <div id="step2" className="group-label">2 · Change one thing</div>
                <div className="group pad"><ChangePanel base={baseline.config} models={models} onRun={runProfile} disabled={!!running || !!status?.busy} /></div>

                <div id="step3" className="group-label">3 · Compare everything</div>
                <div className="group pad">
                  {runs.filter((r) => r.status === "done").length > 1 ? <Board runs={runs} />
                    : <div className="row-sub">Profile a change in step 2: every run then appears here next to the baseline, with what improved and what got worse.</div>}
                  {runs.filter((r) => r.status !== "done" && r.status !== "running" && r.status !== "queued").map((r) => (
                    <div key={r.id} className="xfailed"><AlertCircle size={14} /> “{r.label}” {r.status}{r.error ? `: ${r.error}` : ""}
                      <button className="icon-btn danger" onClick={() => api.experimentDeleteRun(r.id).then(load)}><Trash2 /></button></div>
                  ))}
                </div>

                <div id="step4" className="group-label history-head">
                  <span>4 · What's holding it back</span>
                  <select className="field select small-select" value={shown?.id ?? ""} onChange={(e) => setFocus(Number(e.target.value))}>
                    {runs.filter((r) => r.status === "done").map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}
                  </select>
                </div>
                <div className="group pad">{shown && <Bottlenecks run={shown} />}</div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
