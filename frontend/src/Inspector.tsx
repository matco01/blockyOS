import { useEffect, useState, useSyncExternalStore } from "react";
import { RotateCcw, X } from "lucide-react";
import { api, MessageStats, Sampling } from "./api";
import { live, rateSeries } from "./live";
import { LineChart, Point, speedVerdict, Term, useMode, waitVerdict } from "./ui";

type Field = { key: keyof Sampling; label: string; term?: string; hint: string; step: number; min: number; max: number; def: string; simple?: boolean };
const FIELDS: Field[] = [
  { key: "temperature", label: "Temperature", term: "temperature", hint: "Randomness. 0 = always the most likely word.", step: 0.05, min: 0, max: 2, def: "0.8", simple: true },
  { key: "top_p", label: "Top-p", hint: "Only sample from the most likely words covering this probability.", step: 0.01, min: 0, max: 1, def: "0.95" },
  { key: "top_k", label: "Top-k", hint: "Only sample from the k most likely words.", step: 1, min: 0, max: 200, def: "40" },
  { key: "min_p", label: "Min-p", hint: "Drop words less likely than this × the top word.", step: 0.01, min: 0, max: 1, def: "0.05" },
  { key: "repeat_penalty", label: "Repeat penalty", hint: "> 1 discourages repeating recent words.", step: 0.01, min: 0.5, max: 2, def: "1.0" },
  { key: "presence_penalty", label: "Presence penalty", hint: "> 0 pushes toward new topics.", step: 0.1, min: -2, max: 2, def: "0" },
  { key: "max_tokens", label: "Max tokens", hint: "Hard cap on the answer length (incl. thinking).", step: 1, min: 1, max: 262144, def: "∞" },
  { key: "seed", label: "Seed", term: "seed", hint: "Same seed + same settings = same answer. Great for A/B tests.", step: 1, min: -1, max: 2 ** 31, def: "random", simple: true },
];

const PRESETS: { label: string; s: Sampling }[] = [
  { label: "Default", s: {} },
  { label: "Precise", s: { temperature: 0.2, top_p: 0.9 } },
  { label: "Creative", s: { temperature: 1.0, top_p: 0.95, min_p: 0.05 } },
  { label: "Qwen thinking", s: { temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0 } },
  { label: "Deterministic", s: { temperature: 0, seed: 42 } },
];

const ms = (v: number | null | undefined) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`);
const n = (v: number | null | undefined, d = 1) => (v == null ? "–" : v >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(d));

/** Live speed while an answer streams, plus GPU power/load on separate charts (never one dual-axis chart). */
function LiveView() {
  const st = useSyncExternalStore(live.subscribe, live.get);
  const [, tick] = useState(0);
  const [gpu, setGpu] = useState<{ t: number; p: number; u: number }[]>([]);

  useEffect(() => {
    if (!st.active) return;
    setGpu([]);
    const started = performance.now();
    const t1 = setInterval(() => tick((x) => x + 1), 250);
    const t2 = setInterval(() => {
      api.system().then((s) => {
        const g = s.gpus[0];
        if (g) setGpu((h) => [...h, { t: (performance.now() - started) / 1000, p: g.power_w ?? 0, u: g.util_percent ?? 0 }]);
      }).catch(() => {});
    }, 500);
    return () => { clearInterval(t1); clearInterval(t2); };
  }, [st.active]);

  const series = rateSeries(st, st.active ? performance.now() : (st.tokens[st.tokens.length - 1]?.t ?? st.started));
  if (!st.started) return <div className="row-sub">Send a message to watch it generate live.</div>;
  const phase = { idle: "Done", waiting: "Reading your message…", thinking: "Thinking…", writing: "Writing…" }[st.phase];
  const now = series[series.length - 1]?.y ?? 0;

  return (
    <div className="live-view">
      <div className="live-phase"><span className={`state-dot ${st.active ? "pulse" : "ok"}`} /> {phase}
        <b className="mono">{st.active ? `${now.toFixed(0)} tok/s` : `${st.tokens.length} tokens`}</b></div>
      <div className="mini-title">Speed <span className="muted">(tok/s)</span></div>
      <LineChart points={series} height={110} color="var(--series-1)" unit="tok/s" xLabel="seconds" fmtX={(x) => `${x.toFixed(1)} s`} />
      {gpu.length > 1 && (
        <div className="mini-pair">
          <div><div className="mini-title">GPU power <span className="muted">(W)</span></div>
            <LineChart compact points={gpu.map((g): Point => ({ x: g.t, y: g.p }))} height={40} color="var(--series-3)" unit="W" fmtX={(x) => `${x.toFixed(1)} s`} /></div>
          <div><div className="mini-title">GPU load <span className="muted">(%)</span></div>
            <LineChart compact points={gpu.map((g): Point => ({ x: g.t, y: g.u }))} yMax={100} height={40} color="var(--series-4)" unit="%" fmtX={(x) => `${x.toFixed(1)} s`} /></div>
        </div>
      )}
    </div>
  );
}

/** What fills the context window: system prompt + memories, chat history, the answer, free space. */
function ContextMeter({ s, ctx }: { s: MessageStats; ctx: number }) {
  const prompt = (s.prompt_n ?? 0) + (s.cache_n ?? 0);
  const sys = Math.min(s.system_n ?? 0, prompt);
  const segs = [
    { k: "sys", label: "Instructions + memories", v: sys, c: "var(--series-1)" },
    { k: "hist", label: "Conversation", v: Math.max(0, prompt - sys), c: "var(--series-2)" },
    { k: "ans", label: "Last answer", v: s.predicted_n ?? 0, c: "var(--series-3)" },
  ];
  const used = segs.reduce((a, b) => a + b.v, 0);
  const free = Math.max(0, ctx - used);
  const pct = (used / ctx) * 100;
  return (
    <div className="ctx-meter">
      <div className="row-sub"><Term k="context">Context window</Term>: {used.toLocaleString()} of {ctx.toLocaleString()} tokens ({pct.toFixed(1)}%)</div>
      <div className="mm-bar thin">
        {segs.filter((x) => x.v > 0).map((x) => <div key={x.k} className="mm-seg" style={{ flexGrow: x.v, background: x.c }} title={`${x.label}: ${x.v.toLocaleString()} tokens`} />)}
        <div className="mm-seg free" style={{ flexGrow: free }} title={`Free: ${free.toLocaleString()} tokens`} />
      </div>
      <div className="mm-legend small">
        {segs.map((x) => <span key={x.k}><i style={{ background: x.c }} />{x.label} <b>{x.v.toLocaleString()}</b></span>)}
      </div>
      {pct > 85 && <div className="row-sub warn">Almost full: older messages will start falling out of what the model can see.</div>}
    </div>
  );
}

export const BUDGETS = [1024, 2048, 4096, 8192, 16384];

export default function Inspector({ sampling, setSampling, systemPrompt, setSystemPrompt, thinkBudget, setThinkBudget, last, ctx, onClose }: {
  sampling: Sampling; setSampling: (s: Sampling) => void;
  systemPrompt: string; setSystemPrompt: (s: string) => void;
  thinkBudget: number; setThinkBudget: (n: number) => void;
  last: MessageStats | null; ctx: number | null; onClose: () => void;
}) {
  const mode = useMode();
  const isDefault = Object.keys(sampling).length === 0 && !systemPrompt;
  const fields = mode === "pro" ? FIELDS : FIELDS.filter((f) => f.simple);

  return (
    <aside className="inspector">
      <div className="insp-head">
        <b>Inspector</b>
        {!isDefault && <button className="icon-btn" title="Reset all" onClick={() => { setSampling({}); setSystemPrompt(""); }}><RotateCcw /></button>}
        <button className="icon-btn" title="Close" onClick={onClose}><X /></button>
      </div>

      <div className="insp-body">
        <div className="insp-section">Live</div>
        <LiveView />

        <div className="insp-section">Last answer</div>
        {last ? (
          <>
            <div className="insp-stats">
              <div className="wide"><span><Term k="tps">Writing speed</Term></span>
                <b className="big">{n(last.predicted_per_second)} <small>tok/s · {speedVerdict(last.predicted_per_second).label}</small></b></div>
              <div><span><Term k="ttft">First word after</Term></span><b>{ms(last.ttft_ms)}</b><em>{waitVerdict(last.ttft_ms)}</em></div>
              <div><span>Total time</span><b>{ms(last.total_ms)}</b></div>
              {mode === "pro" && <>
                <div><span><Term k="pp">Prompt</Term> (new)</span><b>{n(last.prompt_n, 0)} tok</b></div>
                <div><span>Prompt speed</span><b>{n(last.prompt_per_second)} tok/s</b></div>
                <div><span>Reused from cache</span><b>{n(last.cache_n, 0)} tok</b></div>
                <div><span>Generated</span><b>{n(last.predicted_n, 0)} tok</b></div>
              </>}
            </div>
            {ctx && <ContextMeter s={last} ctx={ctx} />}
          </>
        ) : <div className="row-sub">Send a message to see timings.</div>}

        <div className="insp-section">Sampling</div>
        <div className="chips">
          {PRESETS.map((p) => (
            <button key={p.label} className={`chip ${JSON.stringify(p.s) === JSON.stringify(sampling) ? "on" : ""}`} onClick={() => setSampling(p.s)}>{p.label}</button>
          ))}
        </div>
        {fields.map((f) => (
          <label className="insp-field" key={f.key} title={f.hint}>
            <span>{f.term ? <Term k={f.term}>{f.label}</Term> : f.label}</span>
            {(f.key === "temperature" || f.key === "top_p" || f.key === "min_p") && (
              <input type="range" min={f.min} max={f.max} step={f.step}
                value={sampling[f.key] ?? Number(f.def)}
                onChange={(e) => setSampling({ ...sampling, [f.key]: Number(e.target.value) })} />
            )}
            <input className="field num" type="number" step={f.step} min={f.min} max={f.max} placeholder={f.def}
              value={sampling[f.key] ?? ""}
              onChange={(e) => {
                const next = { ...sampling };
                if (e.target.value === "") delete next[f.key];
                else next[f.key] = Number(e.target.value);
                setSampling(next);
              }} />
          </label>
        ))}
        <div className="row-sub" style={{ marginTop: 6 }}>
          Empty = the model's recommended settings{last?.sampling_recommended ? ` (now: ${Object.entries(last.sampling_recommended).map(([k, v]) => `${k} ${v}`).join(", ")})` : ""}.
          {mode === "simple" ? " Switch to Pro for every setting." : " Hover a setting for what it does."}
        </div>

        <div className="insp-section">Thinking budget</div>
        <div className="segmented wrap budget-seg">
          {BUDGETS.map((b) => <button key={b} className={thinkBudget === b ? "on" : ""} onClick={() => setThinkBudget(b)}>{b >= 1024 ? `${b / 1024}k` : b}</button>)}
        </div>
        <div className="row-sub" style={{ marginTop: 6 }}>
          Max tokens the model may think before it must answer. Stops endless second-guessing; bigger budgets help on hard problems.
        </div>

        <div className="insp-section">System prompt</div>
        <textarea className="field insp-prompt" rows={6} value={systemPrompt}
          placeholder="Empty = Blocky's default persona. Memories and today's date are still added."
          onChange={(e) => setSystemPrompt(e.target.value)} />
      </div>
    </aside>
  );
}
