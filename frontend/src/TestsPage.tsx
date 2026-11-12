import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, BookOpen, Brain, Calculator, Check, ClipboardCheck, Code2, Download, FlaskConical, GraduationCap, ListChecks, Loader2, Play, Plus, Puzzle, Sigma, Square, Trash2, Trophy, User, X } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { api, BenchmarkInfo, EvalItem, EvalRun, ModelFile, Status, SuiteInfo } from "./api";
import { ImportBenchmark, OutsideTools } from "./BenchImport";
import { BUDGETS } from "./Inspector";
import { Pro, Tabs, Term, useMode } from "./ui";

const ICONS: Record<string, LucideIcon> = {
  brain: Brain, calculator: Calculator, puzzle: Puzzle, book: BookOpen, "list-checks": ListChecks, code: Code2, user: User,
  flask: FlaskConical, graduation: GraduationCap, sigma: Sigma, trophy: Trophy,
};
const SIZES = [25, 50, 100, 250, 0]; // 0 = all

/** Display info for a suite spec like "math" or "bench:gsm8k@50". */
type SuiteMeta = { title: string; icon: string; color: string; baseline?: number | null };
function metaFor(spec: string, suites: Record<string, SuiteInfo>, benches: Record<string, BenchmarkInfo>): SuiteMeta {
  const [key, n] = spec.split("@");
  const b = benches[key];
  if (b) return { title: `${b.title}${n ? ` · ${n}` : " · all"}`, icon: b.icon, color: b.color, baseline: b.baseline };
  const s = suites[key];
  return s ? { title: s.title, icon: s.icon, color: s.color } : { title: key, icon: "user", color: "var(--accent)" };
}
const SLOTS = 4;
const runLabel = (r: EvalRun) => `${r.model.replace(/\.gguf$/, "")}${r.think ? ` · thinking${r.options?.think_budget ? ` ${r.options.think_budget / 1024}k` : ""}` : ""}`;
/** One-line preview of a question: drop LaTeX markers so lists stay readable. */
const plain = (s: string) => s.replace(/\\[()[\]]|\$+/g, "").replace(/\\(?:left|right|t?d?frac|mathcal|overline)/g, "").replace(/\s+/g, " ");
const pct = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(v * 100)}%`);

function grade(score: number | null | undefined) {
  if (score == null) return { label: "–", tone: "meh" };
  if (score >= 0.9) return { label: "Excellent", tone: "great" };
  if (score >= 0.75) return { label: "Strong", tone: "good" };
  if (score >= 0.5) return { label: "Mixed", tone: "meh" };
  return { label: "Weak", tone: "bad" };
}

/** Scorecard for one run: headline score + one bar per test set + drill-down. */
function Scorecard({ run, meta }: { run: EvalRun; meta: (spec: string) => SuiteMeta }) {
  const mode = useMode();
  const [open, setOpen] = useState<string | null>(null);
  const s = run.summary;
  if (!s) return null;
  const g = grade(s.overall);
  return (
    <div className="scorecard">
      <div className="sc-head">
        <div className="sc-score">
          <div className="sc-ring" style={{ "--p": `${(s.overall ?? 0) * 360}deg` } as React.CSSProperties}><span>{pct(s.overall)}</span></div>
          <div>
            <div className="sc-model">{runLabel(run)}</div>
            <span className={`verdict tone-${g.tone}`}>{g.label}</span>
            <div className="row-sub">{s.correct} of {s.total} correct · {Math.round(s.seconds)} s{s.energy_j ? ` · ${s.energy_j >= 3600 ? `${(s.energy_j / 3600).toFixed(2)} Wh` : `${Math.round(s.energy_j)} J`}` : ""}
              {mode === "pro" && s.avg_tps ? ` · ${s.avg_tps.toFixed(0)} tok/s · ${s.tokens.toLocaleString()} tokens` : ""}</div>
            {run.think && <div className="row-sub">Thinking budget {run.options?.think_budget ? `${run.options.think_budget.toLocaleString()} tokens` : "unlimited"}
              {s.budget_hits ? ` · reached on ${s.budget_hits} question${s.budget_hits === 1 ? "" : "s"} (forced to answer)` : " · never reached"}</div>}
          </div>
        </div>
      </div>
      <div className="sc-bars">
        {run.suites.map((key) => {
          const p = s.per_suite[key];
          const info = meta(key);
          const Icon = ICONS[info.icon] ?? User;
          const items = run.results.filter((r) => r.suite === key);
          return (
            <div key={key} className="sc-suite" style={{ "--c": info.color } as React.CSSProperties}>
              <button className="sc-row" onClick={() => setOpen(open === key ? null : key)}>
                <span className="sc-icon"><Icon /></span>
                <span className="sc-name">{info.title}</span>
                <span className="sc-track">
                  <span style={{ width: `${(p?.score ?? 0) * 100}%` }} />
                  {info.baseline != null && <i className="sc-baseline" style={{ left: `${info.baseline * 100}%` }} title={`Random guessing: ${pct(info.baseline)}`} />}
                </span>
                <span className="sc-pct">{p ? pct(p.score) : "…"}</span>
                <span className="sc-count">{p ? `${p.correct}/${p.total}` : ""}</span>
              </button>
              {open === key && (
                <div className="sc-items">
                  {items.map((it) => <QuestionRow key={it.index} it={it} />)}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="row-sub">Click a test set to see every question, the model's answer and why it was marked right or wrong.
        {run.suites.some((k) => meta(k).baseline != null) && " The thin line marks the random-guessing score."}</div>
    </div>
  );
}

function QuestionRow({ it }: { it: EvalItem }) {
  const [show, setShow] = useState(false);
  return (
    <div className={`q-row ${it.correct ? "ok" : "bad"}`}>
      <button className="q-main" onClick={() => setShow(!show)}>
        <span className="q-mark">{it.correct ? <Check /> : <X />}</span>
        <span className="q-text">{plain(it.question)}</span>
        {it.budget_hit && <span className="badge" title="Thinking budget reached: the model was forced to answer">budget</span>}
        {it.tag && <span className="q-tag">{it.tag}</span>}
        <span className="q-meta mono">{it.seconds.toFixed(1)} s</span>
      </button>
      {show && (
        <div className="q-detail">
          <div><span>Expected</span><b>{it.expected}</b></div>
          <div><span>{it.kind === "rules" || it.kind === "code" ? "Result" : "Model's answer"}</span><b>{it.got || "(nothing)"}</b></div>
          {it.truncated && <div className="warn small">Cut off: hit the length limit.</div>}
          {it.reasoning_chars > 0 && <div className="small muted">Thought for {it.reasoning_chars.toLocaleString()} characters first.</div>}
          <pre className="log small">{it.output || "(empty)"}</pre>
        </div>
      )}
    </div>
  );
}

/** Grouped bars: one group per test set, one bar per compared run. */
function CompareSuites({ runs, slotOf, meta }: { runs: EvalRun[]; slotOf: (id: number) => number; meta: (spec: string) => SuiteMeta }) {
  const keys = Array.from(new Set(runs.flatMap((r) => r.suites)));
  return (
    <div className="chart">
      <div className="chart-body">
        <div className="chart-grid">
          {[0, 25, 50, 75, 100].map((t) => <span key={t} style={{ left: `${t}%` }}><i>{t}%</i></span>)}
        </div>
        {["overall", ...keys].map((k) => (
          <div className="chart-group" key={k}>
            <div className="chart-label" title={k === "overall" ? "Overall" : meta(k).title}>{k === "overall" ? <b>Overall</b> : meta(k).title}</div>
            <div className="chart-bars">
              {runs.map((r) => {
                const v = k === "overall" ? r.summary?.overall : r.summary?.per_suite[k]?.score;
                return v == null ? <div key={r.id} className="chart-bar empty" /> : (
                  <div key={r.id} className="chart-bar" title={`${runLabel(r)} · ${pct(v)}`}
                    style={{ width: `${v * 100}%`, background: `var(--series-${slotOf(r.id) + 1})` }} />
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Scatter: speed (x) vs score (y). Only runs over the *same* test sets are comparable,
 * so we pick the most recent test-set combination that has 2+ model setups.
 */
function SmartVsFast({ runs, meta }: { runs: EvalRun[]; meta: (spec: string) => SuiteMeta }) {
  const [hover, setHover] = useState<number | null>(null);

  const { latest, suites } = useMemo(() => {
    const done = runs.filter((r) => r.status === "done" && r.summary?.avg_tps);
    for (const r of done) {  // newest first
      const sig = [...r.suites].sort().join("|");
      const seen = new Map<string, EvalRun>();
      for (const x of done) if ([...x.suites].sort().join("|") === sig && !seen.has(runLabel(x))) seen.set(runLabel(x), x);
      if (seen.size >= 2) return { latest: [...seen.values()].slice(0, 3), suites: r.suites }; // dataviz cap: 3 slots
    }
    return { latest: [] as EvalRun[], suites: [] as string[] };
  }, [runs]);

  if (latest.length < 2) return <div className="row-sub">Run the same tests on at least two models (or thinking on and off) to see them plotted here.</div>;
  return <ScatterPlot runs={latest} hover={hover} setHover={setHover} suites={suites.map((s) => meta(s).title)} />;
}

function niceMax(v: number) {
  const mag = 10 ** Math.floor(Math.log10(v));
  return Math.ceil(v / mag) * mag;
}

/** The plot itself: SVG for grid + dots, HTML cards for labels and the hover card. */
function ScatterPlot({ runs, hover, setHover, suites }: {
  runs: EvalRun[]; hover: number | null; setHover: (id: number | null) => void; suites: string[];
}) {
  // Draw at the container's real pixel width, so text stays its true size.
  const plotRef = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(0);
  useEffect(() => {
    const el = plotRef.current;
    if (!el) return;
    const measure = () => setW(el.clientWidth);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = 300, pad = { l: 56, r: 24, t: 20, b: 44 };
  const xs = runs.map((r) => r.summary!.avg_tps!);
  const ys = runs.map((r) => r.summary!.overall ?? 0);
  const xMax = niceMax(Math.max(...xs) * 1.2);
  const yMin = Math.max(0, Math.floor((Math.min(...ys) - 0.15) * 10) / 10); // zoom to where the scores are
  const sx = (v: number) => pad.l + (v / xMax) * (W - pad.l - pad.r);
  const sy = (v: number) => pad.t + (1 - (v - yMin) / (1 - yMin)) * (H - pad.t - pad.b);
  const yTicks = Array.from({ length: Math.round((1 - yMin) / 0.1) + 1 }, (_, i) => yMin + i * 0.1).filter((_, i, a) => a.length <= 6 || i % 2 === 0);
  const xTicks = Array.from({ length: 5 }, (_, i) => (xMax / 4) * i);
  const midX = sx(xMax / 2), midY = sy((1 + yMin) / 2);

  // Label cards (~175×44 px): try right of the dot, then left, then nudge down,
  // never covering another dot or another card.
  const LW = 175, LH = 44;
  const dots = runs.map((r) => ({ id: r.id, x: sx(r.summary!.avg_tps!), y: sy(r.summary!.overall ?? 0) }));
  const placed: { id: number; x: number; y: number; left: boolean }[] = [];
  const box = (x: number, y: number, left: boolean) => ({ x0: left ? x - 16 - LW : x + 16, x1: left ? x - 16 : x + 16 + LW, y0: y - 20, y1: y - 20 + LH });
  const clashes = (id: number, b: ReturnType<typeof box>) =>
    dots.some((d) => d.id !== id && d.x > b.x0 - 10 && d.x < b.x1 + 10 && d.y > b.y0 - 10 && d.y < b.y1 + 10) ||
    placed.some((p) => { const o = box(p.x, p.y, p.left); return b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0; }) ||
    b.x0 < pad.l || b.x1 > W - 4;
  [...dots].sort((a, b) => a.y - b.y).forEach((d) => {
    let best: { y: number; left: boolean } | null = null;
    for (let shift = 0; shift <= 4 && !best; shift++) {
      for (const left of [false, true]) {
        const y = d.y + shift * (LH + 4);
        if (!clashes(d.id, box(d.x, y, left))) { best = { y, left }; break; }
      }
    }
    placed.push({ id: d.id, x: d.x, ...(best ?? { y: d.y, left: d.x > W * 0.6 }) });
  });

  const hr = runs.find((r) => r.id === hover);
  return (
    <div className="scatter">
      <div className="scatter-plot" ref={plotRef} style={{ height: H }}>
        {W > 0 && <>
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Score versus speed per model setup">
          <defs>
            <linearGradient id="sweet" x1="0" y1="1" x2="1" y2="0">
              <stop offset="0%" stopColor="var(--success)" stopOpacity="0" />
              <stop offset="100%" stopColor="var(--success)" stopOpacity="0.14" />
            </linearGradient>
          </defs>
          <rect x={midX} y={pad.t} width={W - pad.r - midX} height={midY - pad.t} fill="url(#sweet)" rx="10" />
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} className="lc-grid" />
              <text x={pad.l - 10} y={sy(t) + 4} textAnchor="end" className="sc-tick">{Math.round(t * 100)}%</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} x={sx(t)} y={H - pad.b + 18} textAnchor="middle" className="sc-tick">{Math.round(t)}</text>
          ))}
          <line x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} className="sc-axis" />
          <text x={(pad.l + W - pad.r) / 2} y={H - 6} textAnchor="middle" className="sc-axis-label">Writing speed (tok/s) → faster</text>
          <text x={14} y={(pad.t + H - pad.b) / 2} textAnchor="middle" className="sc-axis-label" transform={`rotate(-90 14 ${(pad.t + H - pad.b) / 2})`}>Score → smarter</text>
          <text x={W - pad.r - 10} y={pad.t + 18} textAnchor="end" className="sc-sweet">Sweet spot: fast & smart</text>
          {runs.map((r, i) => (
            <g key={r.id} onMouseEnter={() => setHover(r.id)} onMouseLeave={() => setHover(null)}>
              <circle cx={sx(r.summary!.avg_tps!)} cy={sy(r.summary!.overall ?? 0)} r={18} fill="transparent" />
              <circle cx={sx(r.summary!.avg_tps!)} cy={sy(r.summary!.overall ?? 0)} r={hover === r.id ? 11 : 9}
                fill={`var(--series-${i + 1})`} stroke="var(--surface)" strokeWidth={3} className="sc-dot" />
            </g>
          ))}
        </svg>
        {placed.map((p) => {
          const r = runs.find((x) => x.id === p.id)!;
          const i = runs.indexOf(r);
          const [name, quant] = r.model.replace(/\.gguf$/, "").split(/-(?=(?:UD-)?(?:IQ|Q|BF|F)\d)/);
          return (
            <div key={p.id} className={`sc-label ${p.left ? "left" : ""} ${hover === p.id ? "on" : ""}`}
              style={{ left: p.left ? undefined : p.x + 16, right: p.left ? W - p.x + 16 : undefined, top: p.y - 20, "--c": `var(--series-${i + 1})` } as React.CSSProperties}
              onMouseEnter={() => setHover(p.id)} onMouseLeave={() => setHover(null)}>
              <b>{name}{r.think ? " · thinking" : ""}</b>
              <span>{pct(r.summary!.overall)} · {r.summary!.avg_tps!.toFixed(0)} tok/s{quant ? ` · ${quant}` : ""}</span>
            </div>
          );
        })}
        </>}
      </div>
      {hr && (
        <div className="sc-detail">
          <b>{runLabel(hr)}</b>
          <span>Score <b>{pct(hr.summary!.overall)}</b> ({hr.summary!.correct}/{hr.summary!.total})</span>
          <span>Speed <b>{hr.summary!.avg_tps!.toFixed(0)} tok/s</b></span>
          <span><b>{hr.summary!.seconds_per_question?.toFixed(1)} s</b> per question</span>
          {hr.summary!.energy_j && <span><b>{Math.round(hr.summary!.energy_j / hr.summary!.total)} J</b> per question</span>}
        </div>
      )}
      <div className="row-sub">Same tests for every dot: {suites.join(", ")}.</div>
    </div>
  );
}

function CustomEditor({ onSaved, onClose }: { onSaved: () => void; onClose: () => void }) {
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const lines = text.split("\n").map((l) => l.split("|")).filter((p) => p.length >= 2 && p[0].trim() && p[1].trim());
  const save = async () => {
    setErr(null);
    try {
      await api.evalCreateSuite(name, lines.map(([q, ...a]) => ({ q: q.trim(), answer: a.join("|").trim() })));
      onSaved();
    } catch (e) { setErr((e as Error).message); }
  };
  return (
    <div className="group pad custom-editor">
      <div className="row-sub">One question per line: <code>question | expected answer</code>. Numbers are compared as numbers; text answers can list alternatives with <code>/</code> (e.g. <code>Who painted the Mona Lisa? | Leonardo / da Vinci</code>).</div>
      <input className="field" placeholder="Name, e.g. My work questions" value={name} onChange={(e) => setName(e.target.value)} />
      <textarea className="field mono" rows={7} value={text} onChange={(e) => setText(e.target.value)}
        placeholder={"What is the capital of Czechia? | Prague\nHow many legs does a spider have? | 8"} />
      {err && <div className="error"><AlertCircle /> {err}</div>}
      <div className="custom-actions">
        <span className="row-sub">{lines.length} question{lines.length === 1 ? "" : "s"}</span>
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn" disabled={!name.trim() || lines.length === 0} onClick={save}><Plus /> Save test set</button>
      </div>
    </div>
  );
}

export default function TestsPage({ status }: { status: Status | null }) {
  const [models, setModels] = useState<ModelFile[]>([]);
  const [suites, setSuites] = useState<SuiteInfo[]>([]);
  const [model, setModel] = useState("");
  const [think, setThink] = useState(false);
  const [picked, setPicked] = useState<string[]>(["math", "logic", "hard", "knowledge", "instructions", "coding"]);
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [viewId, setViewId] = useState<number | null>(null);
  const [slots, setSlots] = useState<Record<number, number>>({});
  const [editor, setEditor] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [benches, setBenches] = useState<BenchmarkInfo[]>([]);
  const [benchPick, setBenchPick] = useState<Record<string, number>>({}); // benchmark id -> sample size (0 = all)
  const [budget, setBudget] = useState(4096);
  const [importing, setImporting] = useState(false);
  const [stopping, setStopping] = useState<number | null>(null);

  const load = useCallback(() => {
    api.evalRuns().then((rs) => { setRuns(rs); setViewId((v) => v ?? rs.find((r) => r.status !== "error")?.id ?? null); }).catch(() => {});
    api.evalSuites().then(setSuites).catch(() => {});
    api.benchmarks().then(setBenches).catch(() => {});
  }, []);

  // Poll while a benchmark downloads.
  const downloading = benches.some((b) => b.download?.status === "downloading");
  useEffect(() => {
    if (!downloading) return;
    const t = setInterval(() => api.benchmarks().then(setBenches).catch(() => {}), 1000);
    return () => clearInterval(t);
  }, [downloading]);

  useEffect(() => {
    api.models().then((d) => { setModels(d.models); setModel((m) => m || d.runtime.model || d.models[0]?.file || ""); }).catch(() => {});
    load();
  }, [load]);

  const running = runs.find((r) => r.status === "running" || r.status === "queued");
  const [tab, setTab] = useState<"run" | "results" | "tools">("run");
  // When a run finishes, jump to its results.
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !running) setTab("results");
    wasRunning.current = !!running;
  }, [running]);
  useEffect(() => {
    const t = setInterval(load, running ? 1500 : 15000);
    return () => clearInterval(t);
  }, [running, load]);

  const byKey = useMemo(() => Object.fromEntries(suites.map((s) => [s.key, s])), [suites]);
  const benchByKey = useMemo(() => Object.fromEntries(benches.map((b) => [b.key, b])), [benches]);
  const meta = useCallback((spec: string) => metaFor(spec, byKey, benchByKey), [byKey, benchByKey]);
  const specCount = useCallback((spec: string) => {
    const [key, n] = spec.split("@");
    const total = byKey[key]?.count ?? benchByKey[key]?.count ?? 0;
    return n ? Math.min(Number(n), total) : total;
  }, [byKey, benchByKey]);

  const benchSpecs = Object.entries(benchPick).map(([id, n]) => (n ? `bench:${id}@${n}` : `bench:${id}`));
  const specs = [...picked, ...benchSpecs];
  const view = runs.find((r) => r.id === viewId) ?? null;
  const qCount = specs.reduce((a, k) => a + specCount(k), 0);
  const perQ = think ? Math.min(budget, 8192) / 55 + 3 : 3; // rough: tokens / ~55 tok/s
  const estSec = qCount * perQ + (models.find((m) => m.active)?.file === model ? 0 : 10);

  const start = async () => {
    setError(null);
    try {
      const { id } = await api.evalStart({ model, think, suites: specs, think_budget: think ? budget : undefined });
      setViewId(id);
      load();
    } catch (e) { setError((e as Error).message); }
  };

  const toggleCompare = (id: number) => setSlots((s) => {
    if (id in s) { const { [id]: _, ...rest } = s; return rest; }
    const used = new Set(Object.values(s));
    const free = [...Array(SLOTS).keys()].find((k) => !used.has(k));
    return free == null ? s : { ...s, [id]: free };
  });
  const compared = runs.filter((r) => r.id in slots).sort((a, b) => slots[a.id] - slots[b.id]);

  return (
    <div className="page">
      <div className="page-inner wide">
        <h1 className="page-title">Quality tests</h1>
        <p className="page-sub">How smart is a model? Run question sets with known answers. Code checks every answer, so scores are objective and repeatable.</p>
        <Tabs value={tab} onChange={setTab} tabs={[
          { id: "run", label: running ? "Running…" : "New test" },
          { id: "results", label: "Results", badge: runs.filter((r) => r.status === "done").length },
          { id: "tools", label: "Other tools" },
        ]} />

        {tab === "run" && (<>
        {/* Setup */}
        <div className="group-label">New test run</div>
        <div className="group">
          <div className="row">
            <div className="row-main"><div>Model</div></div>
            <select className="field select" value={model} onChange={(e) => setModel(e.target.value)}>
              {models.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file} · {m.quant}</option>)}
            </select>
          </div>
          <label className="row">
            <div className="row-main"><div>Think first</div><div className="row-sub">Usually more accurate on math and logic, but much slower and uses more energy.</div></div>
            <input type="checkbox" className="switch" checked={think} onChange={(e) => setThink(e.target.checked)} />
          </label>
          {think && (
            <div className="row">
              <div className="row-main"><div>Thinking budget</div><div className="row-sub">After this many thinking tokens the model must answer. Bigger = better on hard problems, but slower.</div></div>
              <div className="segmented wrap">
                {BUDGETS.map((b) => <button key={b} className={budget === b ? "on" : ""} onClick={() => setBudget(b)}>{b / 1024}k</button>)}
              </div>
            </div>
          )}
          <div className="row block">
            <div className="sub-head">Quick checks <span className="muted">· Blocky's own small sets</span></div>
            <div className="suite-grid">
              {suites.map((s) => {
                const Icon = ICONS[s.icon] ?? User;
                const on = picked.includes(s.key);
                return (
                  <button key={s.key} disabled={!s.available} className={`suite-card ${on ? "on" : ""}`} style={{ "--c": s.color } as React.CSSProperties}
                    onClick={() => setPicked((p) => (on ? p.filter((k) => k !== s.key) : [...p, s.key]))}>
                    <span className="suite-icon"><Icon /></span>
                    <span className="suite-body"><b>{s.title}</b><span>{s.about}</span><em>{s.count} questions{!s.available ? " · sandbox missing" : ""}</em></span>
                    <span className="suite-check">{on && <Check />}</span>
                    {s.custom && <span className="suite-del" title="Delete test set" onClick={(e) => { e.stopPropagation(); if (confirm(`Delete "${s.title}"?`)) api.evalDeleteSuite(s.key).then(load); }}><Trash2 /></span>}
                  </button>
                );
              })}
              <button className="suite-card add" onClick={() => setEditor(true)}>
                <span className="suite-icon"><Plus /></span>
                <span className="suite-body"><b>Your own questions</b><span>Paste questions with expected answers.</span></span>
              </button>
            </div>
          </div>
          {editor && <div className="row block"><CustomEditor onClose={() => setEditor(false)} onSaved={() => { setEditor(false); load(); }} /></div>}

          <div className="row block">
            <div className="sub-head">Established benchmarks <span className="muted">· the sets researchers publish scores on</span></div>
            {([["general", "Math, reasoning & knowledge"], ["coding", "Coding"], ["custom", "Imported by you"]] as const).map(([group, heading]) => {
              const list = benches.filter((b) => (b.group ?? "general") === group);
              if (list.length === 0 && group !== "custom") return null;
              return (
            <div key={group} className="bench-group">
            <div className="bench-group-head">
              <span>{heading}</span>
              {group === "custom" && !importing && <button className="btn small ghost" onClick={() => setImporting(true)}><Plus /> Import from Hugging Face</button>}
            </div>
            {group === "custom" && importing && <ImportBenchmark onClose={() => setImporting(false)} onDone={() => { setImporting(false); load(); }} />}
            {group === "custom" && list.length === 0 && !importing && <div className="row-sub">Any public Hugging Face dataset with questions and answers can become a benchmark.</div>}
            <div className="bench-list">
              {list.map((b) => {
                const Icon = ICONS[b.icon] ?? User;
                const size = benchPick[b.id];
                const on = size !== undefined;
                const dl = b.download?.status === "downloading" ? b.download : null;
                return (
                  <div key={b.id} className={`bench-item ${on ? "on" : ""}`} style={{ "--c": b.color } as React.CSSProperties}>
                    <span className="suite-icon"><Icon /></span>
                    <div className="bench-body">
                      <div className="bench-title"><b>{b.title}</b><span className={`diff diff-${b.difficulty.replace(/\s/g, "").toLowerCase()}`}>{b.difficulty}</span>
                        {b.downloaded && <span className="muted small">{b.count?.toLocaleString()} questions</span>}
                        {b.date_from && <span className="muted small" title="If a model was trained after these dates it may have seen the problems">· problems from {b.date_from.slice(0, 7)} to {b.date_to?.slice(0, 7)}</span>}
                        {b.custom && <button className="icon-btn danger tiny" title="Remove" onClick={() => { if (confirm(`Remove "${b.title}"?`)) api.benchmarkDelete(b.id).then(load); }}><Trash2 /></button>}</div>
                      <div className="row-sub">{b.about}</div>
                      {dl && <div className="mem-bar"><div className="ok" style={{ width: `${(dl.done / Math.max(1, dl.total)) * 100}%` }} /></div>}
                      {b.download?.status === "error" && <div className="row-sub" style={{ color: "var(--danger)" }}>{b.download.error}</div>}
                      {b.downloaded && on && (
                        <div className="size-pick">
                          <span className="row-sub">Questions:</span>
                          <div className="segmented">
                            {SIZES.filter((n) => n === 0 || n < (b.count ?? 0)).map((n) => (
                              <button key={n} className={size === n ? "on" : ""} onClick={() => setBenchPick((p) => ({ ...p, [b.id]: n }))}>{n || `All ${b.count?.toLocaleString()}`}</button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    {!b.downloaded ? (
                      <button className="btn small ghost" disabled={!!dl} onClick={() => api.benchmarkDownload(b.id).then(load)}>
                        {dl ? <><Loader2 className="spin" /> {Math.round((dl.done / Math.max(1, dl.total)) * 100)}%</> : <><Download /> Get</>}
                      </button>
                    ) : (
                      <button className={`btn small ${on ? "" : "ghost"}`}
                        onClick={() => setBenchPick((p) => { if (on) { const { [b.id]: _, ...rest } = p; return rest; } return { ...p, [b.id]: Math.min(50, b.count ?? 50) >= (b.count ?? 0) ? 0 : 50 }; })}>
                        {on ? <><Check /> Added</> : <><Plus /> Add</>}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
            </div>
              );
            })}
            <div className="row-sub" style={{ marginTop: 8 }}>Downloaded once from Hugging Face, then stored on this device. The same questions are used every run (fixed shuffle), so runs stay comparable.</div>
          </div>

          <div className="row apply-row">
            <div className="row-main row-sub">{qCount.toLocaleString()} questions · about {estSec < 90 ? `${Math.round(estSec)} s` : estSec < 5400 ? `${Math.round(estSec / 60)} min` : `${(estSec / 3600).toFixed(1)} h`}. Chat pauses while it runs.</div>
            {running ? (
              <button className="btn ghost" disabled={stopping === running.id}
                onClick={() => { setStopping(running.id); api.evalCancel(running.id).then(load); }}>
                {stopping === running.id ? <><Loader2 className="spin" /> Stopping…</> : <><Square /> Stop</>}
              </button>
            ) : (
              <button className="btn" disabled={!model || specs.length === 0 || !!status?.busy} onClick={start}><Play /> Run tests</button>
            )}
          </div>
        </div>
        {error && <div className="error" style={{ marginTop: 12 }}><AlertCircle /> {error}</div>}

        {/* Live */}
        {running && (
          <>
            <div className="group-label">Running · {runLabel(running)}</div>
            <div className="group pad">
              <div className="live-phase"><Loader2 className="spin" size={15} /> Question {running.results.length + 1} of {running.suites.reduce((a, k) => a + specCount(k), 0)}
                <b className="mono">{running.summary ? `${running.summary.correct} correct so far` : ""}</b></div>
              <div className="mem-bar"><div className="ok" style={{ width: `${(running.results.length / Math.max(1, running.suites.reduce((a, k) => a + specCount(k), 0))) * 100}%` }} /></div>
              <div className="ticker">
                {running.results.slice(-6).reverse().map((it) => (
                  <div key={`${it.suite}-${it.index}`} className={`tick ${it.correct ? "ok" : "bad"}`}>{it.correct ? <Check size={13} /> : <X size={13} />} {plain(it.question)}</div>
                ))}
              </div>
            </div>
          </>
        )}

        </>)}

        {tab === "results" && (<>
        {/* Scorecard */}
        {view && view.summary && view.status !== "running" && (
          <>
            <div className="group-label">Scorecard · #{view.id}{view.status === "cancelled" ? " (stopped early)" : view.status === "error" ? " (failed)" : ""}</div>
            <Scorecard run={view} meta={meta} />
          </>
        )}

        {/* Smart vs fast */}
        <div className="group-label">Smart vs fast</div>
        <div className="group pad">
          <SmartVsFast runs={runs} meta={meta} />
          <div className="row-sub" style={{ marginTop: 8 }}>Up and to the right is best. Bigger models usually move up (smarter) and left (slower). <Term k="tps">tok/s</Term> is the average writing speed during the test.</div>
        </div>

        {/* Compare */}
        {compared.length > 0 && (
          <>
            <div className="group-label">Compare</div>
            <div className="group pad viz-root">
              <div className="legend" style={{ marginBottom: 12 }}>
                {compared.map((r) => <span key={r.id}><i style={{ background: `var(--series-${slots[r.id] + 1})` }} />{runLabel(r)} <span className="muted">#{r.id}</span></span>)}
              </div>
              <CompareSuites runs={compared} slotOf={(id) => slots[id]} meta={meta} />
            </div>
          </>
        )}

        </>)}

        {tab === "tools" && (<>
        <div className="group-label">Beyond Blocky's tests</div>
        <OutsideTools />
        </>)}

        {tab === "results" && (<>
        {/* History */}
        <div className="group-label">All runs</div>
        <div className="group">
          {runs.length === 0 && <div className="row"><div className="empty-state" style={{ width: "100%" }}><ClipboardCheck /><br />No test runs yet.</div></div>}
          {runs.map((r) => (
            <div key={r.id} className={`row run-row hist-row ${viewId === r.id ? "selected" : ""}`} onClick={() => setViewId(r.id)}>
              <input type="checkbox" className="check" title="Compare" checked={r.id in slots} onClick={(e) => e.stopPropagation()}
                disabled={!(r.id in slots) && (Object.keys(slots).length >= SLOTS || !r.summary?.overall)}
                onChange={() => toggleCompare(r.id)} style={r.id in slots ? ({ "--c": `var(--series-${slots[r.id] + 1})` } as React.CSSProperties) : undefined} />
              <div className="row-main">
                <div>{runLabel(r)} <span className={`status-tag ${r.status}`}>{r.status}</span></div>
                <div className="row-sub">#{r.id} · {new Date(r.created_at * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })} · {r.suites.map((k) => meta(k).title).join(", ")}{r.error ? ` · ${r.error}` : ""}</div>
              </div>
              <div className="kpis">
                <div><b>{pct(r.summary?.overall)}</b><span>score</span></div>
                <Pro><div><b>{r.summary?.seconds_per_question?.toFixed(1) ?? "–"}</b><span>s / question</span></div></Pro>
              </div>
              <button className="icon-btn danger" title="Delete" disabled={r.status === "running"} onClick={(e) => { e.stopPropagation(); api.evalDelete(r.id).then(load); }}><Trash2 /></button>
            </div>
          ))}
        </div>
        </>)}
      </div>
    </div>
  );
}
