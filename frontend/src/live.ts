/**
 * Tiny live-generation store. Chat pushes events as tokens stream in; the
 * Inspector's live chart subscribes. Kept outside React state so streaming
 * doesn't re-render the whole app on every token.
 */
export type LiveState = {
  active: boolean;
  phase: "idle" | "waiting" | "thinking" | "writing";
  started: number;
  tokens: { t: number; kind: "r" | "c" }[]; // timestamps of streamed chunks (~1 token each)
};

let state: LiveState = { active: false, phase: "idle", started: 0, tokens: [] };
const subs = new Set<() => void>();
const emit = () => subs.forEach((f) => f());

export const live = {
  get: () => state,
  subscribe(f: () => void) { subs.add(f); return () => { subs.delete(f); }; },
  start() { state = { active: true, phase: "waiting", started: performance.now(), tokens: [] }; emit(); },
  token(kind: "r" | "c") {
    state.tokens.push({ t: performance.now(), kind });
    const phase = kind === "r" ? "thinking" : "writing";
    if (state.phase !== phase) { state = { ...state, phase }; emit(); }
  },
  stop() { state = { ...state, active: false, phase: "idle" }; emit(); },
};

/** Tokens/second over time, in 250 ms buckets, smoothed over ~1 s. */
export function rateSeries(s: LiveState, until = performance.now()) {
  const bucket = 250;
  const out: { x: number; y: number }[] = [];
  if (!s.started) return out;
  const n = Math.max(1, Math.ceil((until - s.started) / bucket));
  const counts = new Array(n).fill(0);
  for (const tok of s.tokens) {
    const i = Math.floor((tok.t - s.started) / bucket);
    if (i >= 0 && i < n) counts[i]++;
  }
  for (let i = 0; i < n; i++) {
    const win = counts.slice(Math.max(0, i - 3), i + 1);
    out.push({ x: ((i + 1) * bucket) / 1000, y: (win.reduce((a, b) => a + b, 0) / win.length) * (1000 / bucket) });
  }
  return out;
}
