import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, ChevronDown, Download, FlaskConical, Loader2, Play, Square, Trash2 } from "lucide-react";
import { api, BenchResult, BenchRun, ModelFile, Status } from "./api";
import { Pro, speedVerdict, Tabs, Term, useMode } from "./ui";
import { Speculative, Throughput } from "./SpeedTools";

/* Chart series colors: brand hues stepped for data marks (validated: lightness band,
   chroma, CVD + normal-vision separation, >=3:1 contrast, light and dark). */
const SLOTS = 4;

type Preset = { id: string; label: string; hint: string; p: number[]; n: number[]; d: number[]; r: number };
const PRESETS: Preset[] = [
  { id: "quick", label: "Quick", hint: "Prompt + generation speed. ~20 s.", p: [512], n: [128], d: [0], r: 3 },
  { id: "context", label: "Long context", hint: "How speed drops as the conversation grows.", p: [512], n: [128], d: [0, 4096, 16384], r: 3 },
  { id: "prompt", label: "Prompt sweep", hint: "Prompt processing at different batch sizes.", p: [128, 512, 2048, 8192], n: [0], d: [0], r: 3 },
  { id: "gen", label: "Long generation", hint: "Sustained generation (thermals, power).", p: [0], n: [128, 512, 1024], d: [0], r: 2 },
];

const parseList = (s: string) => s.split(/[,\s]+/).filter(Boolean).map(Number).filter((n) => Number.isFinite(n) && n >= 0);
const fmtDate = (t: number) => new Date(t * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
const fmt = (n: number | null | undefined, d = 1) => (n == null ? "–" : n >= 1000 ? Math.round(n).toLocaleString() : n.toFixed(d));
const isPrompt = (r: BenchResult) => !!r.n_prompt && !r.n_gen;

function headline(run: BenchRun) {
  const res = run.results ?? [];
  const tg = res.find((r) => r.n_gen && !r.n_prompt && !r.n_depth) ?? res.find((r) => r.n_gen && !r.n_prompt);
  const pp = res.find((r) => isPrompt(r) && !r.n_depth) ?? res.find(isPrompt);
  return { tg, pp };
}

/** Grouped horizontal bars: one group per test, one bar per selected run. */
function CompareChart({ title, runs, slotOf, pick, unit }: {
  title: React.ReactNode; runs: BenchRun[]; slotOf: (id: number) => number;
  pick: (r: BenchResult) => number | null; unit: string;
}) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string[] } | null>(null);
  const tests = useMemo(() => {
    const seen: string[] = [];
    runs.forEach((run) => run.results?.forEach((r) => { if (pick(r) != null && !seen.includes(r.test)) seen.push(r.test); }));
    return seen;
  }, [runs, pick]);
  const max = Math.max(1, ...runs.flatMap((run) => (run.results ?? []).map((r) => pick(r) ?? 0)));
  const step = niceStep(max);
  const top = Math.ceil(max / step) * step;
  const ticks = Array.from({ length: Math.round(top / step) + 1 }, (_, i) => i * step);
  if (tests.length === 0) return null;

  return (
    <div className="chart">
      <div className="chart-title">{title} <span className="muted">({unit})</span></div>
      <div className="chart-body" onMouseLeave={() => setTip(null)}>
        <div className="chart-grid">
          {ticks.map((t) => <span key={t} style={{ left: `${(t / top) * 100}%` }}><i>{fmt(t, step < 1 ? 2 : 0)}</i></span>)}
        </div>
        {tests.map((test) => (
          <div className="chart-group" key={test}>
            <div className="chart-label mono" title={test}>{test}</div>
            <div className="chart-bars">
              {runs.map((run) => {
                const r = run.results?.find((x) => x.test === test);
                const v = r ? pick(r) : null;
                if (v == null) return <div key={run.id} className="chart-bar empty" />;
                return (
                  <div key={run.id} className="chart-bar"
                    style={{ width: `${(v / top) * 100}%`, background: `var(--series-${slotOf(run.id) + 1})` }}
                    onMouseMove={(e) => {
                      const box = (e.currentTarget.closest(".chart-body") as HTMLElement).getBoundingClientRect();
                      setTip({ x: e.clientX - box.left, y: e.clientY - box.top, text: [
                        `${run.model} · #${run.id}`, `${test}: ${fmt(v, 2)} ${unit}`,
                        r?.stddev_ts != null && unit === "tok/s" ? `± ${fmt(r.stddev_ts, 1)} (${r.samples_ts?.length ?? "?"} runs)` : "",
                      ].filter(Boolean) });
                    }} />
                );
              })}
            </div>
          </div>
        ))}
        {tip && <div className="chart-tip" style={{ left: tip.x + 12, top: tip.y + 12 }}>{tip.text.map((t, i) => <div key={i} className={i ? "" : "b"}>{t}</div>)}</div>}
      </div>
    </div>
  );
}

function niceStep(max: number) {
  const raw = max / 5;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const n = raw / mag;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
}

/** "tg128 @ d4096" -> "Writing 128 tokens, with 4k already in the chat" */
export function friendlyTest(r: BenchResult) {
  const depth = r.n_depth ? `, with ${r.n_depth >= 1024 ? `${Math.round(r.n_depth / 1024)}k` : r.n_depth} tokens already in the chat` : "";
  if (r.n_prompt && !r.n_gen) return `Reading a ${r.n_prompt}-token prompt${depth}`;
  if (r.n_gen && !r.n_prompt) return `Writing ${r.n_gen} tokens${depth}`;
  return `${r.test}${depth}`;
}

function ResultsTable({ results }: { results: BenchResult[] }) {
  const mode = useMode();
  if (mode === "simple") {
    return (
      <div className="simple-results">
        {results.map((r, i) => {
          const writing = !!r.n_gen && !r.n_prompt;
          const v = writing ? speedVerdict(r.avg_ts) : null;
          return (
            <div key={i} className="simple-result">
              <span>{friendlyTest(r)}</span>
              <b>{fmt(r.avg_ts, 0)} <small>tok/s</small></b>
              <span className={`verdict tone-${v?.tone ?? "great"}`}>{v ? v.label : r.avg_ts > 500 ? "Very fast" : "OK"}</span>
            </div>
          );
        })}
      </div>
    );
  }
  return (
    <div className="table-wrap">
      <table className="data">
        <thead><tr><th>Test</th><th className="r">tok/s</th><th className="r">± sd</th><th className="r">Power</th><th className="r">tok/J</th><th className="r">Peak VRAM</th><th className="r">GPU layers</th><th>KV</th><th className="r">Threads</th></tr></thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i}>
              <td className="mono">{r.test}</td>
              <td className="r b">{fmt(r.avg_ts, 1)}</td>
              <td className="r muted">{fmt(r.stddev_ts, 1)}</td>
              <td className="r">{r.power_w != null ? `${fmt(r.power_w, 0)} W` : "–"}</td>
              <td className="r">{fmt(r.tokens_per_joule, 2)}</td>
              <td className="r">{r.peak_vram_mb != null ? `${(r.peak_vram_mb / 1024).toFixed(2)} GB` : "–"}</td>
              <td className="r">{r.n_gpu_layers < 0 ? "auto" : r.n_gpu_layers}{r.n_cpu_moe ? ` (moe→cpu ${r.n_cpu_moe})` : ""}</td>
              <td className="mono">{r.type_k}</td>
              <td className="r">{r.n_threads}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function BenchPage({ status }: { status: Status | null }) {
  const [models, setModels] = useState<ModelFile[]>([]);
  const [runs, setRuns] = useState<BenchRun[]>([]);
  const [preset, setPreset] = useState<string>("quick");
  const [form, setForm] = useState({ model: "", p: "512", n: "128", d: "0", r: 3, kv: "q8_0", fa: "auto", ngl: "", threads: "", ncmoe: "" });
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<number | null>(null);
  // runId -> color slot (follows the run, not its rank). Seeded from a shareable link: #/bench?compare=3,4
  const [slots, setSlots] = useState<Record<number, number>>(() => {
    const ids = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("compare")?.split(",").map(Number).filter(Boolean) ?? [];
    return Object.fromEntries(ids.slice(0, SLOTS).map((id, i) => [id, i]));
  });

  useEffect(() => {
    const ids = Object.entries(slots).sort((a, b) => a[1] - b[1]).map(([id]) => id);
    const want = ids.length ? `#/bench?compare=${ids.join(",")}` : "#/bench";
    if (window.location.hash !== want) window.history.replaceState(null, "", want);
  }, [slots]);
  const [metric, setMetric] = useState<"speed" | "efficiency">("speed");
  const pollRef = useRef<number | undefined>(undefined);
  const [stopping, setStopping] = useState<number | null>(null);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const mode = useMode();

  const loadRuns = useCallback(() => api.benchRuns().then(setRuns).catch(() => {}), []);

  useEffect(() => {
    api.models().then((d) => {
      setModels(d.models);
      setActiveModel(d.runtime.model);
      setForm((f) => ({ ...f, model: f.model || d.runtime.model || d.models[0]?.file || "", kv: d.runtime.kv }));
    }).catch(() => {});
    loadRuns();
  }, [loadRuns]);

  const running = runs.find((r) => r.status === "running" || r.status === "queued");
  const [tab, setTab] = useState<"run" | "results" | "users" | "draft">(window.location.hash.includes("compare=") ? "results" : "run");
  // When a run finishes, jump to the results.
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !running) setTab("results");
    wasRunning.current = !!running;
  }, [running]);
  useEffect(() => {
    window.clearInterval(pollRef.current);
    pollRef.current = window.setInterval(loadRuns, running ? 1500 : 10000);
    return () => window.clearInterval(pollRef.current);
  }, [running, loadRuns]);

  const choosePreset = (p: Preset) => {
    setPreset(p.id);
    setForm((f) => ({ ...f, p: p.p.join(","), n: p.n.join(","), d: p.d.join(","), r: p.r }));
  };

  const start = async () => {
    setError(null);
    try {
      const { id } = await api.startBench({
        model: form.model, n_prompt: parseList(form.p), n_gen: parseList(form.n), depth: parseList(form.d),
        repetitions: form.r, kv: form.kv, flash_attn: form.fa, ngl: form.ngl || "auto",
        threads: form.threads ? Number(form.threads) : undefined, n_cpu_moe: form.ncmoe ? Number(form.ncmoe) : undefined,
      });
      setOpen(id);
      loadRuns();
    } catch (e) { setError((e as Error).message); }
  };

  const toggleCompare = (id: number) => {
    setSlots((s) => {
      if (id in s) { const { [id]: _, ...rest } = s; return rest; }
      const used = new Set(Object.values(s));
      const free = [...Array(SLOTS).keys()].find((k) => !used.has(k));
      return free == null ? s : { ...s, [id]: free };
    });
  };

  const compared = runs.filter((r) => r.id in slots).sort((a, b) => slots[a.id] - slots[b.id]);
  const expected = parseList(form.p).filter(Boolean).length + parseList(form.n).filter(Boolean).length;
  const runningTotal = running ? ((running.params.n_prompt as number[] ?? []).filter(Boolean).length + (running.params.n_gen as number[] ?? []).filter(Boolean).length) * ((running.params.depth as number[] ?? [0]).length) : 0;
  const pickSpeed = useCallback((r: BenchResult) => r.avg_ts, []);
  const pickEff = useCallback((r: BenchResult) => r.tokens_per_joule, []);

  return (
    <div className="page">
      <div className="page-inner wide">
        <h1 className="page-title">Speed tests</h1>
        <p className="page-sub">How fast is a model on this machine? Measures reading speed, writing speed, power and efficiency.</p>
        <Tabs value={tab} onChange={setTab} tabs={[
          { id: "run", label: running ? "Running…" : "New test" },
          { id: "results", label: "Results", badge: runs.filter((r) => r.status === "done").length },
          { id: "users", label: "Many users" },
          { id: "draft", label: "Draft speed-up" },
        ]} />
        {tab === "users" && <Throughput models={models} active={activeModel} status={status} />}
        {tab === "draft" && <Speculative models={models} active={activeModel} status={status} />}

        {tab === "run" && (<>
        {/* New run */}
        <div className="group-label">New run</div>
        <div className="group">
          <div className="row">
            <div className="row-main"><div>Preset</div></div>
            <div className="segmented wrap">
              {PRESETS.map((p) => <button key={p.id} className={preset === p.id ? "on" : ""} title={p.hint} onClick={() => choosePreset(p)}>{p.label}</button>)}
            </div>
          </div>
          <div className="row">
            <div className="row-main"><div>Model</div></div>
            <select className="field select" value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })}>
              {models.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file} · {m.quant}</option>)}
            </select>
          </div>
          {mode === "simple" && (
            <div className="row"><div className="row-main row-sub">{PRESETS.find((p) => p.id === preset)?.hint ?? "Custom test."} Switch to Pro for every llama-bench option.</div></div>
          )}
          <Pro><div className="row form-grid">
            <label><span>Prompt tokens (pp)</span><input className="field mono" value={form.p} onChange={(e) => { setForm({ ...form, p: e.target.value }); setPreset("custom"); }} /></label>
            <label><span>Generated tokens (tg)</span><input className="field mono" value={form.n} onChange={(e) => { setForm({ ...form, n: e.target.value }); setPreset("custom"); }} /></label>
            <label><span>Context depth</span><input className="field mono" value={form.d} onChange={(e) => { setForm({ ...form, d: e.target.value }); setPreset("custom"); }} /></label>
            <label><span>Repetitions</span><input className="field mono" type="number" min={1} max={20} value={form.r} onChange={(e) => setForm({ ...form, r: Number(e.target.value) || 1 })} /></label>
            <label><span>KV cache</span>
              <select className="field select" value={form.kv} onChange={(e) => setForm({ ...form, kv: e.target.value })}>{["f16", "q8_0", "q4_0"].map((k) => <option key={k}>{k}</option>)}</select></label>
            <label><span>Flash attention</span>
              <select className="field select" value={form.fa} onChange={(e) => setForm({ ...form, fa: e.target.value })}>{["auto", "on", "off"].map((k) => <option key={k}>{k}</option>)}</select></label>
            <label><span>GPU layers</span><input className="field mono" placeholder="auto (fit)" value={form.ngl} onChange={(e) => setForm({ ...form, ngl: e.target.value.replace(/\D/g, "") })} /></label>
            <label><span>Threads</span><input className="field mono" placeholder="auto" value={form.threads} onChange={(e) => setForm({ ...form, threads: e.target.value.replace(/\D/g, "") })} /></label>
            <label><span>MoE layers on CPU</span><input className="field mono" placeholder="0" value={form.ncmoe} onChange={(e) => setForm({ ...form, ncmoe: e.target.value.replace(/\D/g, "") })} /></label>
          </div></Pro>
          <div className="row apply-row">
            <div className="row-main row-sub">
              {expected * parseList(form.d).length} tests × {form.r} repetitions. Comma-separate values to sweep. Chat pauses while it runs.
            </div>
            {running ? (
              <button className="btn ghost" disabled={stopping === running.id} onClick={() => { setStopping(running.id); api.cancelBench(running.id).then(loadRuns); }}>
                {stopping === running.id ? <><Loader2 className="spin" /> Stopping…</> : <><Square /> Stop</>}
              </button>
            ) : (
              <button className="btn" disabled={!form.model || !!status?.busy} onClick={start}><Play /> Run benchmark</button>
            )}
          </div>
        </div>
        {error && <div className="error" style={{ marginTop: 12 }}><AlertCircle /> {error}</div>}

        {/* Live run */}
        {running && (
          <>
            <div className="group-label">Running · #{running.id}</div>
            <div className="group">
              <div className="row">
                <Loader2 className="spin" style={{ color: "var(--accent)" }} />
                <div className="row-main">
                  <div>{running.model}</div>
                  <div className="mem-bar"><div className="ok" style={{ width: `${runningTotal ? ((running.results?.length ?? 0) / runningTotal) * 100 : 5}%` }} /></div>
                  <div className="row-sub">{running.results?.length ?? 0} of {runningTotal || "?"} tests done</div>
                </div>
              </div>
              {running.results && running.results.length > 0 && <div className="row block"><ResultsTable results={running.results} /></div>}
              <div className="row block"><pre className="log small">{(running.log ?? "").split("\n").slice(-6).join("\n") || "Starting…"}</pre></div>
            </div>
          </>
        )}

        </>)}

        {tab === "results" && (<>
        {/* Compare */}
        {compared.length > 0 && (
          <>
            <div className="group-label">Compare</div>
            <div className="group viz-root">
              <div className="row">
                <div className="legend">
                  {compared.map((r) => (
                    <span key={r.id}><i style={{ background: `var(--series-${slots[r.id] + 1})` }} />{r.model.replace(/\.gguf$/, "")} <span className="muted">#{r.id}</span></span>
                  ))}
                </div>
                <div className="segmented">
                  <button className={metric === "speed" ? "on" : ""} onClick={() => setMetric("speed")}>Speed</button>
                  <button className={metric === "efficiency" ? "on" : ""} onClick={() => setMetric("efficiency")}>Efficiency</button>
                </div>
              </div>
              <div className="row block">
                {metric === "speed" ? (
                  <>
                    <CompareChart title={<><Term k="pp">Reading speed</Term> (prompt processing)</>} unit="tok/s" runs={compared} slotOf={(id) => slots[id]} pick={(r) => (isPrompt(r) ? pickSpeed(r) : null)} />
                    <CompareChart title={<><Term k="tg">Writing speed</Term> (generation)</>} unit="tok/s" runs={compared} slotOf={(id) => slots[id]} pick={(r) => (!isPrompt(r) ? pickSpeed(r) : null)} />
                  </>
                ) : (
                  <>
                    <CompareChart title={<>Reading <Term k="tpj">efficiency</Term></>} unit="tok/J" runs={compared} slotOf={(id) => slots[id]} pick={(r) => (isPrompt(r) ? pickEff(r) : null)} />
                    <CompareChart title={<>Writing <Term k="tpj">efficiency</Term></>} unit="tok/J" runs={compared} slotOf={(id) => slots[id]} pick={(r) => (!isPrompt(r) ? pickEff(r) : null)} />
                  </>
                )}
              </div>
            </div>
          </>
        )}

        {/* History */}
        <div className="group-label history-head">
          <span>All runs <span className="muted" style={{ fontWeight: 400 }}>· tick up to 4 to compare them in a chart</span></span>
          {runs.some((r) => r.results?.length) && (
            <span className="exports">
              <a href="/api/bench/export?format=csv"><Download size={12} /> CSV</a>
              <a href="/api/bench/export?format=json"><Download size={12} /> JSON</a>
            </span>
          )}
        </div>
        <div className="group">
          {runs.length === 0 && (
            <div className="row"><div className="empty-state" style={{ width: "100%" }}><FlaskConical /><br />No benchmarks yet. Run a Quick one to get a baseline.</div></div>
          )}
          {runs.map((run) => {
            const { tg, pp } = headline(run);
            const expanded = open === run.id;
            return (
              <div key={run.id} className="run">
                <div className="row run-row">
                  <input type="checkbox" className="check" title="Compare" checked={run.id in slots}
                    disabled={!(run.id in slots) && (Object.keys(slots).length >= SLOTS || !run.results?.length)}
                    onChange={() => toggleCompare(run.id)}
                    style={run.id in slots ? ({ "--c": `var(--series-${slots[run.id] + 1})` } as React.CSSProperties) : undefined} />
                  <div className="row-main" onClick={() => setOpen(expanded ? null : run.id)} style={{ cursor: "pointer" }}>
                    <div>{run.model.replace(/\.gguf$/, "")} <span className={`status-tag ${run.status}`}>{run.status}</span></div>
                    <div className="row-sub">#{run.id} · {fmtDate(run.created_at)} · {run.hardware?.gpus[0]?.name.replace("NVIDIA GeForce ", "") ?? "CPU"}{run.error ? ` · ${run.error}` : ""}</div>
                  </div>
                  <div className="kpis">
                    {pp && <div><b>{fmt(pp.avg_ts, 0)}</b><span>pp tok/s</span></div>}
                    {tg && <div><b>{fmt(tg.avg_ts, 1)}</b><span>tg tok/s</span></div>}
                    {tg?.tokens_per_joule != null && <div><b>{fmt(tg.tokens_per_joule, 2)}</b><span>tg tok/J</span></div>}
                  </div>
                  <button className="icon-btn" onClick={() => setOpen(expanded ? null : run.id)}><ChevronDown style={{ transform: expanded ? "rotate(180deg)" : undefined, transition: "transform .2s" }} /></button>
                  <button className="icon-btn danger" title="Delete run" disabled={run.status === "running"} onClick={() => api.deleteBench(run.id).then(loadRuns)}><Trash2 /></button>
                </div>
                {expanded && (
                  <div className="row block run-detail">
                    {run.results && run.results.length > 0 ? <ResultsTable results={run.results} /> : <div className="muted small">No results.</div>}
                    <div className="fingerprint mono small">
                      {run.hardware && <>
                        {run.hardware.gpus.map((g) => `${g.name} (${(g.vram_mb / 1024).toFixed(0)} GB, driver ${g.driver}${g.power_limit_w ? `, ${g.power_limit_w} W limit` : ""})`).join(", ") || "No GPU"}<br />
                        {run.hardware.cpu} · {run.hardware.cpu_cores}C/{run.hardware.cpu_threads}T · {(run.hardware.ram_mb / 1024).toFixed(1)} GB RAM<br />
                        {run.hardware.os} · kernel {run.hardware.kernel}{run.hardware.wsl ? " (WSL2)" : ""} · llama.cpp {run.hardware.llama_cpp ?? "?"}<br />
                      </>}
                      params: {JSON.stringify(run.params)}
                    </div>
                    {run.log && <details><summary className="small muted">llama-bench log</summary><pre className="log small">{run.log}</pre></details>}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        <p className="row-sub" style={{ marginTop: 12 }}>Power is whole-GPU board power sampled every 0.5 s while each test runs (includes the desktop's share on a shared GPU). tok/J = tokens per second ÷ watts.</p>
        </>)}
      </div>
    </div>
  );
}
