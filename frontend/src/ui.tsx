/* Shared UI: Simple/Pro mode, glossary terms, plain-language verdicts, charts. */
import { createContext, useContext, useId, useMemo, useRef, useState } from "react";

// ---------- Mode ----------

export type Mode = "simple" | "pro";
export const ModeContext = createContext<Mode>("simple");
export const useMode = () => useContext(ModeContext);
/** Renders children only in Pro mode. */
export function Pro({ children }: { children: React.ReactNode }) {
  return useMode() === "pro" ? <>{children}</> : null;
}

// ---------- Tabs ----------

/** Splits a long page into a few clearly named views. */
export function Tabs<T extends string>({ tabs, value, onChange }: {
  tabs: { id: T; label: string; badge?: string | number | null }[]; value: T; onChange: (t: T) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} className={value === t.id ? "on" : ""} onClick={() => onChange(t.id)}>
          {t.label}
          {t.badge != null && t.badge !== 0 && <span className="tab-badge">{t.badge}</span>}
        </button>
      ))}
    </div>
  );
}

// ---------- Glossary ----------

export const GLOSSARY: Record<string, { title: string; what: string; why: string }> = {
  tps: { title: "Tokens per second", what: "How many tokens (word pieces, ~¾ of a word) the model produces each second.",
    why: "You read at ~5–8 tok/s. Above ~20 it feels instant." },
  ttft: { title: "Time to first token", what: "How long until the first word of the answer appears.",
    why: "This is the 'waiting' feeling. Long prompts and thinking make it longer." },
  pp: { title: "Prompt processing", what: "Reading your message and the conversation before answering. Done in big parallel batches, so it's much faster than writing.",
    why: "Matters for long documents and long chats." },
  tg: { title: "Generation", what: "Writing the answer, one token at a time. Limited mostly by memory speed.",
    why: "This is the speed you see while text streams in." },
  vram: { title: "VRAM (GPU memory)", what: "The graphics card's own fast memory. Models run fastest when they fit entirely in it.",
    why: "VRAM size decides which models run well on your machine." },
  quant: { title: "Quantization", what: "Compressing a model's numbers (e.g. from 16-bit to 4-bit) so it's ~4× smaller.",
    why: "Q4_K_M is the usual sweet spot: big savings, small quality loss. Q8 is near-original, Q3 and below start to hurt." },
  context: { title: "Context window", what: "How much text the model can 'see' at once: system prompt, memories, the whole chat so far, and the answer.",
    why: "When a chat outgrows it, the oldest parts fall out. Bigger windows cost memory." },
  kv: { title: "KV cache (context memory)", what: "Where the model stores what it has already read, so it doesn't re-read the whole chat for every word.",
    why: "Grows with the context window. q8_0 halves it with almost no quality loss." },
  layers: { title: "Layers", what: "A model is a stack of layers that each refine the answer. Each one can live on the GPU (fast) or CPU (slow).",
    why: "All layers on the GPU = full speed. Each layer on the CPU slows things down." },
  moe: { title: "Mixture of Experts (MoE)", what: "A model made of many 'experts' where only a few are used per word. E.g. 35B total, 3B active.",
    why: "Big-model knowledge at small-model speed. Experts can sit in normal RAM without much slowdown." },
  tpj: { title: "Tokens per joule", what: "Energy efficiency: tokens produced per joule of GPU energy.",
    why: "Higher = cheaper and cooler to run. Useful for comparing models and power limits." },
  temperature: { title: "Temperature", what: "How adventurous word choice is. 0 = always the most likely word, 1+ = more random.",
    why: "Low for facts and code, higher for creative writing." },
  seed: { title: "Seed", what: "The starting point of the randomness.",
    why: "Same seed + same settings = the exact same answer. Essential for fair A/B tests." },
  fa: { title: "Flash attention", what: "A faster, memory-saving way to compute attention.",
    why: "Usually on. Needed for compressed (q8_0/q4_0) context memory." },
  compute: { title: "Working buffers", what: "Scratch memory the GPU needs while computing.", why: "Fixed overhead, usually a few hundred MB." },
  other: { title: "Other apps", what: "GPU memory used by the desktop, browsers and other programs.", why: "Not available to the AI. Closing apps can free some." },
};

/** A technical term with a dotted underline that explains itself on hover/tap. */
export function Term({ k, children }: { k: keyof typeof GLOSSARY | string; children?: React.ReactNode }) {
  const g = GLOSSARY[k];
  const [open, setOpen] = useState(false);
  const id = useId();
  if (!g) return <>{children}</>;
  return (
    <span className="term" tabIndex={0} aria-describedby={id}
      onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}
      onClick={(e) => { e.stopPropagation(); setOpen((o) => !o); }}>
      {children ?? g.title}
      {open && (
        <span className="term-pop" role="tooltip" id={id}>
          <b>{g.title}</b>
          <span>{g.what}</span>
          <span className="why">{g.why}</span>
        </span>
      )}
    </span>
  );
}

// ---------- Plain-language verdicts ----------

export function speedVerdict(tps: number | null | undefined): { label: string; tone: "bad" | "meh" | "good" | "great" } {
  if (tps == null) return { label: "–", tone: "meh" };
  if (tps < 4) return { label: "Very slow", tone: "bad" };
  if (tps < 8) return { label: "About reading speed", tone: "meh" };
  if (tps < 20) return { label: "Faster than you read", tone: "good" };
  return { label: "Feels instant", tone: "great" };
}

export function waitVerdict(ms: number | null | undefined) {
  if (ms == null) return "–";
  if (ms < 500) return "Starts right away";
  if (ms < 2000) return "Short pause";
  if (ms < 8000) return "Noticeable wait";
  return "Long wait";
}

/** Energy for a typical 500-token answer, in a relatable unit. */
export function energyVerdict(tokPerJoule: number | null | undefined) {
  if (!tokPerJoule) return null;
  const wh = 500 / tokPerJoule / 3600;
  const phoneCharges = wh / 15; // ~15 Wh per phone charge
  return { wh, text: wh < 0.1 ? `${(wh * 1000).toFixed(0)} mWh per answer` : `${wh.toFixed(2)} Wh per answer`,
    compare: `≈ ${(phoneCharges * 100).toFixed(1)}% of a phone charge` };
}

export const fmtMB = (mb: number) => (mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`);

// ---------- Line chart ----------

export type Point = { x: number; y: number };

/**
 * Single-series line chart with crosshair + tooltip on hover.
 * `compact` = sparkline (no axes). One series per chart: never two y-scales.
 */
export function LineChart({ points, height = 120, color = "var(--series-1)", unit = "", compact = false,
  yMax, xLabel, fmtX, area = true }: {
  points: Point[]; height?: number; color?: string; unit?: string; compact?: boolean; yMax?: number;
  xLabel?: string; fmtX?: (x: number) => string; area?: boolean;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const W = 600;
  const H = height;
  const pad = compact ? { l: 0, r: 0, t: 4, b: 2 } : { l: 40, r: 8, t: 10, b: xLabel ? 26 : 18 };

  const { path, areaPath, ys, top, x0, x1 } = useMemo(() => {
    if (points.length === 0) return { path: "", areaPath: "", ys: [], top: 1, x0: 0, x1: 1 };
    const x0 = points[0].x, x1 = Math.max(points[points.length - 1].x, x0 + 1e-9);
    const max = yMax ?? Math.max(...points.map((p) => p.y), 1e-9);
    const top = niceCeil(max);
    const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
    const sy = (y: number) => pad.t + (1 - y / top) * (H - pad.t - pad.b);
    const d = points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
    const a = `${d}L${sx(points[points.length - 1].x).toFixed(1)},${H - pad.b}L${sx(x0).toFixed(1)},${H - pad.b}Z`;
    return { path: d, areaPath: a, ys: [0, top / 2, top], top, x0, x1 };
  }, [points, yMax, H, pad.l, pad.r, pad.t, pad.b]);

  const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - y / top) * (H - pad.t - pad.b);
  const gid = useId().replace(/:/g, "");

  const onMove = (e: React.MouseEvent) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box || points.length === 0) return;
    const x = ((e.clientX - box.left) / box.width) * W;
    let best = 0;
    for (let i = 1; i < points.length; i++) if (Math.abs(sx(points[i].x) - x) < Math.abs(sx(points[best].x) - x)) best = i;
    setHover(best);
  };

  const hp = hover != null ? points[hover] : null;
  return (
    <div className={`linechart ${compact ? "compact" : ""}`}>
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height: compact ? height : undefined }}
        onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img" aria-label={`${xLabel ?? "chart"}${unit ? ` in ${unit}` : ""}`}>
        <defs>
          <linearGradient id={gid} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.22" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        {!compact && ys.map((y) => (
          <g key={y}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(y)} y2={sy(y)} className="lc-grid" />
            <text x={pad.l - 6} y={sy(y) + 4} className="lc-tick" textAnchor="end">{fmtNum(y)}</text>
          </g>
        ))}
        {area && points.length > 1 && <path d={areaPath} fill={`url(#${gid})`} />}
        {points.length > 1 && <path d={path} fill="none" stroke={color} strokeWidth={2} vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />}
        {hp && (
          <>
            <line x1={sx(hp.x)} x2={sx(hp.x)} y1={pad.t} y2={H - pad.b} className="lc-cross" vectorEffect="non-scaling-stroke" />
          </>
        )}
        {!compact && xLabel && <text x={(pad.l + W - pad.r) / 2} y={H - 4} className="lc-tick" textAnchor="middle">{xLabel}</text>}
      </svg>
      {hp && (
        <div className="lc-dot" style={{ left: `${(sx(hp.x) / W) * 100}%`, top: `${(sy(hp.y) / H) * 100}%`, background: color }} />
      )}
      {hp && (
        <div className="lc-tip" style={{ left: `${(sx(hp.x) / W) * 100}%` }}>
          <b>{fmtNum(hp.y)} {unit}</b>{fmtX && <span>{fmtX(hp.x)}</span>}
        </div>
      )}
      {points.length < 2 && !compact && <div className="lc-empty">Waiting for data…</div>}
    </div>
  );
}

function niceCeil(v: number) {
  if (v <= 0) return 1;
  const mag = 10 ** Math.floor(Math.log10(v));
  const n = v / mag;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * mag;
}
const fmtNum = (v: number) => (v >= 1000 ? Math.round(v).toLocaleString() : v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));
