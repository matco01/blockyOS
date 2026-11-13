import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Copy, GitBranch, GitCompare, Loader2, Maximize, Minus, Play, Plus, Repeat, Server, Square, StickyNote, Trash2, Wand2, X } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { api, CanvasNote, CanvasState, Device, Experiment, ExpRun, RemoteJob } from "./api";
import { allMetrics, colorOf, delta, hwLabel, METRICS, n1, runTitle } from "./expmetrics";

/**
 * Research canvas: the experiment as a tree you work in. Each device (this one, remote Blocky devices, imported results)
 * is a root card with its hardware; its runs grow from it: device → baseline → changes. Branch from any card with "+"
 * or right-click, profile setups on another device from its card, compare any two cards, annotate with notes.
 */

const CARD_W = 270;
const CARD_H = 186;
const DEV_W = 250;
const COL = CARD_W + 200;   // horizontal step per generation (room for the "what changed" label)
const ROW = CARD_H + 28;
const ROOT_X = DEV_W + 110; // first run column, right of the device cards

type View = { x: number; y: number; k: number };
type Menu = { x: number; y: number; wx: number; wy: number; run?: ExpRun; lane?: Lane } | null;
type Hw = { gpu?: string | null; vram_mb?: number | null; cpu?: string; cpu_threads?: number; ram_mb?: number; ram_bandwidth_gbps?: number | null;
  gpu_bandwidth_gbps?: number | null; device?: string };
type Lane = { key: string; name: string; kind: "local" | "remote" | "imported"; deviceId?: number; runs: ExpRun[]; hw?: Hw };

/** Which device a run ran on: a registered remote device, an imported file's device, or this one. */
function laneOf(r: ExpRun, currentGpu: string | null | undefined): Omit<Lane, "runs"> {
  const x = r.results;
  if (x.remote_device) return { key: `dev:${x.remote_device.id}`, name: x.remote_device.name, kind: "remote", deviceId: x.remote_device.id };
  if (x.imported_from) { const n = x.imported_from.device ?? x.imported_from.gpu; return { key: `imp:${n}`, name: n, kind: "imported" }; }
  const g = x.hardware?.gpu;
  if (g && currentGpu && g !== currentGpu) return { key: `local:${g}`, name: `This device (earlier, ${g.replace("NVIDIA GeForce ", "")})`, kind: "imported" };
  return { key: "local", name: "This device", kind: "local" };
}

/** Lanes stacked top to bottom; in each, a left-to-right tree whose roots hang from the device card. */
function layout(lanes: Lane[]) {
  const pos: Record<number, { x: number; y: number }> = {};
  const dev: Record<string, { x: number; y: number }> = {};
  let row = 0;
  for (const L of lanes) {
    const ids = new Set(L.runs.map((r) => r.id));
    const kids = new Map<number | null, ExpRun[]>();
    for (const r of L.runs) {
      const p = r.parent_id != null && ids.has(r.parent_id) ? r.parent_id : null;
      kids.set(p, [...(kids.get(p) ?? []), r]);
    }
    const start = row;
    const place = (r: ExpRun, depth: number): number => {
      const ch = kids.get(r.id) ?? [];
      const y = ch.length === 0 ? row++ * ROW : (() => { const ys = ch.map((c) => place(c, depth + 1)); return (ys[0] + ys[ys.length - 1]) / 2; })();
      pos[r.id] = { x: ROOT_X + depth * COL, y };
      return y;
    };
    const ys = (kids.get(null) ?? []).map((root) => place(root, 0));
    if (row === start) row++;
    dev[L.key] = { x: 0, y: ys.length ? (ys[0] + ys[ys.length - 1]) / 2 : start * ROW };
    row += 0.6;
  }
  return { pos, dev };
}

const SHOWN = ["tg", "mmlu", "kld", "vram_model"] as const;
const SHORT: Record<string, string> = { tg: "Decode", mmlu: "MMLU-Pro", kld: "KLD", vram_model: "VRAM" };
const gb = (mb?: number | null) => (mb ? `${Math.round(mb / 1024)} GB` : "–");

function Card({ run, idx, cmp, selected, onDown, onClick, onMenu, onBranch }: { run: ExpRun; idx: number; cmp: ExpRun | undefined; selected: number;
  onDown: (e: React.PointerEvent) => void; onClick: (e: React.MouseEvent) => void; onMenu: (e: React.MouseEvent) => void; onBranch: () => void }) {
  const r = run.results;
  return (
    <div className={`cv-card ${run.status} ${selected ? "sel" : ""}`} onPointerDown={onDown} onClick={onClick} onContextMenu={onMenu}
      style={{ "--c": colorOf(idx) } as React.CSSProperties}>
      <div className="cv-head">
        <span className="xdot" style={{ background: colorOf(idx) }} />
        <b title={run.label}>{runTitle(run)}</b>
        {selected ? <span className="cv-pick">{selected === 1 ? "A" : "B"}</span> : null}
      </div>
      <div className="cv-model" title={run.config.model}>{run.config.model.replace(".gguf", "")} · ctx {run.config.ctx / 1024}k · KV {run.config.kv}</div>
      {run.status === "done" ? (
        <>
          {SHOWN.map((key) => {
            const m = METRICS.find((x) => x.key === key)!;
            const v = m.get(r);
            const d = cmp && cmp.status === "done" && cmp.id !== run.id ? delta(m, { run, base: cmp }) : null;
            return (
              <div className="cv-metric" key={key}><span>{SHORT[key]}</span>
                <b>{v == null ? (key === "kld" && r.token_note ? "n/a" : "–") : (m.fmt ?? n1)(v)}<small>{m.unit}</small></b>
                {d && <em className={`d ${d.tone}`} title={d.title}>{d.text}</em>}</div>
            );
          })}
          {r.hw?.limiter && <div className="cv-lim">limited by {r.hw.limiter.what.toLowerCase()}</div>}
        </>
      ) : (
        <div className="cv-status">
          {run.status === "running" ? <><Loader2 className="spin" size={14} /> {run.phase ?? "starting"}…</> : run.status === "queued" ? "In the queue" : `${run.status}${run.error ? `: ${run.error.slice(0, 90)}` : ""}`}
        </div>
      )}
      <button className="cv-plus" title="Branch from this run: same setup + one change" onPointerDown={(e) => e.stopPropagation()} onClick={(e) => { e.stopPropagation(); onBranch(); }}><Plus size={15} /></button>
    </div>
  );
}

function DeviceCard({ lane, jobs, onDown, onMenu, onProfile }: { lane: Lane; jobs: RemoteJob[]; onDown: (e: React.PointerEvent) => void;
  onMenu: (e: React.MouseEvent) => void; onProfile?: () => void }) {
  const hw = lane.hw ?? {};
  const active = jobs.find((j) => j.status === "running" || j.status === "starting");
  const failed = !active ? jobs.find((j) => j.status === "error") : undefined;
  const tag = lane.kind === "local" ? "this device" : lane.kind === "remote" ? "remote" : "imported";
  return (
    <div className={`cv-card cv-dev ${lane.kind}`} onPointerDown={onDown} onContextMenu={onMenu}>
      <div className="cv-head"><Server size={15} /><b title={lane.name}>{lane.name}</b><span className="cv-tag">{tag}</span></div>
      <div className="cv-spec"><span>GPU</span><b>{hw.gpu ? `${hw.gpu.replace("NVIDIA GeForce ", "")} · ${gb(hw.vram_mb)}` : lane.hw ? "none" : "–"}</b></div>
      <div className="cv-spec"><span>CPU</span><b title={hw.cpu}>{hw.cpu ? `${hw.cpu}${hw.cpu_threads ? ` · ${hw.cpu_threads} thr` : ""}` : "–"}</b></div>
      <div className="cv-spec"><span>RAM</span><b>{gb(hw.ram_mb)}{hw.ram_bandwidth_gbps ? ` · ${Math.round(hw.ram_bandwidth_gbps)} GB/s` : ""}</b></div>
      <div className="cv-spec"><span>Runs</span><b>{lane.runs.length}</b></div>
      {active && <div className="cv-status"><Loader2 className="spin" size={14} /> {active.step ?? "starting"}…</div>}
      {failed?.error && <div className="cv-status err" title={failed.error}>{failed.error.slice(0, 80)}</div>}
      {onProfile && !active && <button className="btn small ghost cv-dev-btn" onPointerDown={(e) => e.stopPropagation()} onClick={(e) => { e.stopPropagation(); onProfile(); }}>
        <Play size={13} /> Profile setups here</button>}
    </div>
  );
}

function Compare({ a, b, onClose }: { a: ExpRun; b: ExpRun; onClose: () => void }) {
  const rows = allMetrics([a, b]).filter((m) => m.get(a.results) != null || m.get(b.results) != null);
  return (
    <div className="cv-side" onPointerDown={(e) => e.stopPropagation()} onWheel={(e) => e.stopPropagation()} onContextMenu={(e) => e.stopPropagation()}>
      <div className="cv-side-head"><GitCompare size={15} /><b title={runTitle(a)}>A: {runTitle(a)}</b><span>vs</span><b title={runTitle(b)}>B: {runTitle(b)}</b><button className="icon-btn" onClick={onClose}><X /></button></div>
      <div className="cv-side-body">
        {(a.results.hardware?.gpu !== b.results.hardware?.gpu || a.results.imported_from || b.results.imported_from) &&
          <div className="row-sub cv-drawer-note">Different devices: A on {hwLabel(a)}, B on {hwLabel(b)}.</div>}
        <table className="data">
          <thead><tr><th>Metric</th><th className="r">A</th><th className="r">B</th><th className="r">B vs A</th></tr></thead>
          <tbody>{rows.map((m) => {
            const va = m.get(a.results), vb = m.get(b.results);
            const d = delta(m, { run: b, base: a });
            return <tr key={m.key}><td>{m.label} <span className="u">{m.unit}</span></td><td className="r">{va == null ? "–" : (m.fmt ?? n1)(va)}</td>
              <td className="r">{vb == null ? "–" : (m.fmt ?? n1)(vb)}</td><td className="r">{d ? <span className={`d ${d.tone}`}>{d.text}</span> : "–"}</td></tr>;
          })}</tbody>
        </table>
        <div className="row-sub">Same noise rules as the results table. MMLU-Pro significance is computed against a run's baseline and its parent; for other pairs a 2% threshold is used.</div>
      </div>
    </div>
  );
}

/** Pick setups (from this device's runs) to profile on a remote device. */
function ProfileHere({ exp, lane, setups, onClose, onStarted }: { exp: Experiment; lane: Lane; setups: ExpRun[]; onClose: () => void; onStarted: () => void }) {
  const [pick, setPick] = useState<number[]>(() => (setups[0] ? [setups[0].id] : []));
  const [err, setErr] = useState<string | null>(null);
  const go = async () => {
    setErr(null);
    try { await api.remoteRun(exp.id, lane.deviceId!, pick); onStarted(); onClose(); } catch (e) { setErr((e as Error).message); }
  };
  return (
    <div className="cv-drawer" onPointerDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()} onContextMenu={(e) => e.stopPropagation()}>
      <div className="cv-side-head"><Server size={15} /><span>Profile on</span><b>{lane.name}</b><button className="icon-btn" onClick={onClose}><X /></button></div>
      <div className="cv-side-body">
        <div className="row-sub cv-drawer-note">Pick setups from this device. {lane.name} runs them with the same protocol (the first one becomes its baseline) and the results appear under its card. The same model files must be on that device.</div>
        <div className="cv-picks">
          {setups.map((r, i) => (
            <label key={r.id} className="ws-check"><input type="checkbox" checked={pick.includes(r.id)} onChange={() => setPick(pick.includes(r.id) ? pick.filter((x) => x !== r.id) : [...pick, r.id])} />
              <span><b>{runTitle(r)}</b><em>{r.config.model.replace(".gguf", "")} · ctx {r.config.ctx / 1024}k · KV {r.config.kv}{i === 0 ? " · baseline" : ""}</em></span></label>
          ))}
          {setups.length === 0 && <span className="row-sub">No setups yet: run a baseline on this device first.</span>}
        </div>
        <div className="xc-btns"><button className="btn" disabled={!pick.length} onClick={go}><Play /> Profile {pick.length} on {lane.name}</button></div>
        {err && <div className="error">{err}</div>}
      </div>
    </div>
  );
}

export default function ResearchCanvas({ exp, runs, branchPanel, onRepeat, onDelete, onCancel, onChanged }: {
  exp: Experiment; runs: ExpRun[]; branchPanel: (parent: ExpRun, close: () => void) => React.ReactNode; onChanged: () => void;
  onRepeat: (r: ExpRun) => void; onDelete: (r: ExpRun) => void; onCancel: (r: ExpRun) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const saved = exp.canvas;
  const [view, setView] = useState<View>(saved?.view ?? { x: 80, y: 80, k: 0.9 });
  // Canvases saved before device cards existed: move hand-placed cards right, out of the device column.
  const [pos, setPos] = useState<CanvasState["pos"]>(() => saved?.devices ? saved.pos
    : Object.fromEntries(Object.entries(saved?.pos ?? {}).map(([k, p]) => [k, { x: p.x + ROOT_X, y: p.y }])));
  const [notes, setNotes] = useState<CanvasNote[]>(saved?.notes ?? []);
  const [pinned, setPinned] = useState<number[]>(saved?.devices ?? []);
  const [sel, setSel] = useState<number[]>([]);
  const [menu, setMenu] = useState<Menu>(null);
  const [branchFrom, setBranchFrom] = useState<number | null>(() => { const m = window.location.hash.match(/branch=(\d+)/); return m ? Number(m[1]) : null; });
  const [profileOn, setProfileOn] = useState<string | null>(null);
  const [vs, setVs] = useState<"parent" | "baseline">("parent");
  const [devices, setDevices] = useState<Device[]>([]);
  const [specs, setSpecs] = useState<Record<number, Hw | null>>({});
  const [jobs, setJobs] = useState<RemoteJob[]>([]);
  const [addOpen, setAddOpen] = useState(false);
  const drag = useRef<{ kind: "pan" | "card" | "note" | "dev"; id?: string; sx: number; sy: number; ox: number; oy: number; moved: boolean } | null>(null);

  const shown = runs.filter((r) => r.status !== "cancelled");
  const done = shown.filter((r) => r.status === "done");
  const byId = useMemo(() => new Map(shown.map((r) => [r.id, r])), [shown]);
  const baseline = shown.find((r) => r.parent_id == null);

  // Registered devices, their specs (asked once), and remote jobs for this experiment.
  useEffect(() => { api.devices().then(setDevices).catch(() => {}); }, []);
  useEffect(() => {
    for (const id of pinned) if (!(id in specs)) {
      setSpecs((s) => ({ ...s, [id]: null }));
      api.deviceCheck(id).then((c) => setSpecs((s) => ({ ...s, [id]: c.ok ? (c.hardware as Hw) : null }))).catch(() => {});
    }
  }, [pinned]); // eslint-disable-line react-hooks/exhaustive-deps
  const activeJobs = useRef(false);
  useEffect(() => {
    let alive = true;
    const tick = () => api.remoteJobs(exp.id).then((j) => {
      if (!alive) return;
      const active = j.some((x) => x.status === "running" || x.status === "starting");
      if (activeJobs.current && !active) onChanged();  // results just came back
      activeJobs.current = active;
      setJobs(j);
    }).catch(() => {});
    tick();
    const t = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(t); };
  }, [exp.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const lanes = useMemo(() => {
    const m = new Map<string, Lane>();
    const cur = exp.limits?.gpu;
    for (const r of shown) {
      const l = laneOf(r, cur);
      if (!m.has(l.key)) m.set(l.key, { ...l, runs: [] });
      const L = m.get(l.key)!;
      L.runs.push(r);
      if (!L.hw && r.results.hardware) L.hw = r.results.hardware as Hw;
    }
    if (!m.has("local")) m.set("local", { key: "local", name: "This device", kind: "local", runs: [] });
    const local = m.get("local")!;
    local.hw = local.hw ?? (exp.limits as Hw | undefined);
    if (local.hw?.device) local.name = `${local.hw.device} (this device)`;
    for (const id of pinned) {
      const d = devices.find((x) => x.id === id);
      if (!m.has(`dev:${id}`) && d) m.set(`dev:${id}`, { key: `dev:${id}`, name: d.name, kind: "remote", deviceId: id, runs: [] });
    }
    for (const L of m.values()) if (L.kind === "remote" && L.deviceId != null && !L.hw && specs[L.deviceId]) L.hw = specs[L.deviceId]!;
    return [local, ...[...m.values()].filter((l) => l !== local)];
  }, [shown, exp.limits, pinned, devices, specs]);
  const laneById = useMemo(() => { const m = new Map<number, string>(); for (const L of lanes) for (const r of L.runs) m.set(r.id, L.key); return m; }, [lanes]);

  const auto = useMemo(() => layout(lanes), [lanes.map((L) => `${L.key}[${L.runs.map((r) => `${r.id}:${r.parent_id}`).join(",")}]`).join("|")]); // eslint-disable-line react-hooks/exhaustive-deps
  const P = (id: number) => pos[id] ?? auto.pos[id] ?? { x: 0, y: 0 };
  const D = (key: string) => pos[`d:${key}`] ?? auto.dev[key] ?? { x: 0, y: 0 };
  const cmpFor = (r: ExpRun) => (vs === "parent" ? byId.get(r.parent_id ?? -1) : baseline);

  // Persist (debounced) on the Blocky box, so the map survives reloads and other browsers see it.
  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    const t = setTimeout(() => { api.experimentCanvas(exp.id, { pos, notes, view, devices: pinned }).catch(() => {}); }, 700);
    return () => clearTimeout(t);
  }, [pos, notes, view, pinned, exp.id]);

  const fit = useCallback(() => {
    const el = box.current;
    if (!el) return;
    const all = [...shown.map((r) => P(r.id)), ...lanes.map((L) => D(L.key)), ...notes.map((n) => ({ x: n.x, y: n.y }))];
    if (!all.length) return;
    const x0 = Math.min(...all.map((p) => p.x)) - 40, y0 = Math.min(...all.map((p) => p.y)) - 20;
    const x1 = Math.max(...all.map((p) => p.x + CARD_W)) + 40, y1 = Math.max(...all.map((p) => p.y + CARD_H)) + 20;
    const k = Math.min(1.1, Math.max(0.25, Math.min((el.clientWidth - 80) / (x1 - x0), (el.clientHeight - 120) / (y1 - y0))));
    setView({ k, x: (el.clientWidth - (x1 - x0) * k) / 2 - x0 * k, y: (el.clientHeight - 60 - (y1 - y0) * k) / 2 - y0 * k });
  }, [shown, lanes, notes, pos, auto]); // eslint-disable-line react-hooks/exhaustive-deps
  // Open fitted unless the map was arranged by hand.
  useEffect(() => { if (!saved?.view || Object.keys(saved?.pos ?? {}).length === 0) fit(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const zoomAt = (factor: number, cx?: number, cy?: number) => setView((v) => {
    const el = box.current!;
    const px = cx ?? el.clientWidth / 2, py = cy ?? el.clientHeight / 2;
    const k = Math.min(2.5, Math.max(0.15, v.k * factor));
    return { k, x: px - ((px - v.x) * k) / v.k, y: py - ((py - v.y) * k) / v.k };
  });

  // Wheel pans; ctrl/⌘ + wheel (or a trackpad pinch) zooms around the cursor.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if ((e.target as HTMLElement).closest(".cv-side, .cv-drawer, .cv-menu")) return;
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const r = el.getBoundingClientRect();
        zoomAt(Math.exp(-e.deltaY * 0.01), e.clientX - r.left, e.clientY - r.top);
      } else setView((v) => ({ ...v, x: v.x - e.deltaX, y: v.y - e.deltaY }));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") { setMenu(null); setBranchFrom(null); setProfileOn(null); setSel([]); setAddOpen(false); } };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, []);

  const down = (e: React.PointerEvent, kind: "pan" | "card" | "note" | "dev", id?: string) => {
    if (e.button !== 0) return;
    if ((e.target as HTMLElement).closest("textarea, button, .cv-side, .cv-drawer, .cv-menu")) return;
    e.stopPropagation();
    setMenu(null);
    const o = kind === "pan" ? { x: view.x, y: view.y } : kind === "card" ? P(Number(id)) : kind === "dev" ? D(id!) : notes.find((n) => n.id === id)!;
    drag.current = { kind, id, sx: e.clientX, sy: e.clientY, ox: o.x, oy: o.y, moved: false };
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
  };
  const move = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.sx, dy = e.clientY - d.sy;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    if (!d.moved) return;
    if (d.kind === "pan") setView((v) => ({ ...v, x: d.ox + dx, y: d.oy + dy }));
    else if (d.kind === "card") setPos((p) => ({ ...p, [d.id!]: { x: d.ox + dx / view.k, y: d.oy + dy / view.k } }));
    else if (d.kind === "dev") setPos((p) => ({ ...p, [`d:${d.id}`]: { x: d.ox + dx / view.k, y: d.oy + dy / view.k } }));
    else setNotes((ns) => ns.map((n) => (n.id === d.id ? { ...n, x: d.ox + dx / view.k, y: d.oy + dy / view.k } : n)));
  };
  const up = () => { setTimeout(() => { drag.current = null; }, 0); };

  const world = (cx: number, cy: number) => {
    const r = box.current!.getBoundingClientRect();
    return { x: (cx - r.left - view.x) / view.k, y: (cy - r.top - view.y) / view.k, sx: cx - r.left, sy: cy - r.top };
  };
  const click = (e: React.MouseEvent, id: number) => {
    if (drag.current?.moved) return;
    setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : e.shiftKey ? [...s.slice(-1), id] : [id]));
  };
  const openMenu = (e: React.MouseEvent, run?: ExpRun, lane?: Lane) => {
    e.preventDefault(); e.stopPropagation();
    const w = world(e.clientX, e.clientY);
    setMenu({ x: w.sx, y: w.sy, wx: w.x, wy: w.y, run, lane });
  };
  const addNote = (x?: number, y?: number) => {
    const el = box.current!;
    const wx = x ?? (el.clientWidth / 2 - view.x) / view.k, wy = y ?? (el.clientHeight / 2 - view.y) / view.k;
    setNotes((ns) => [...ns, { id: Math.random().toString(36).slice(2, 9), x: wx - 100, y: wy - 40, text: "" }]);
  };
  const dbl = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest(".cv-card, .cv-note, .cv-side, .cv-drawer, .cv-menu, .cv-tools, .cv-add")) return;
    const w = world(e.clientX, e.clientY);
    addNote(w.x, w.y);
  };
  const copySettings = (r: ExpRun) => { try { navigator.clipboard.writeText(JSON.stringify(r.config, null, 1)); } catch { /* not allowed */ } };
  const pin = (id: number) => { setPinned((p) => (p.includes(id) ? p : [...p, id])); setAddOpen(false); setPos((p) => p); setTimeout(fit, 50); };

  const pair = sel.length === 2 ? [byId.get(sel[0]), byId.get(sel[1])] : null;
  const branchRun = branchFrom != null ? byId.get(branchFrom) : undefined;
  const profileLane = profileOn ? lanes.find((L) => L.key === profileOn) : undefined;
  const localSetups = (lanes[0]?.runs ?? []).filter((r) => r.status === "done");
  const unpinned = devices.filter((d) => !lanes.some((L) => L.deviceId === d.id));
  const grid = 24 * view.k;
  const sameLane = (r: ExpRun) => r.parent_id != null && byId.has(r.parent_id) && laneById.get(r.parent_id) === laneById.get(r.id);
  const crossLane = (r: ExpRun) => r.parent_id != null && byId.has(r.parent_id) && laneById.get(r.parent_id) !== laneById.get(r.id);
  const roots = (L: Lane) => { const ids = new Set(L.runs.map((r) => r.id)); return L.runs.filter((r) => r.parent_id == null || !ids.has(r.parent_id)); };
  const curve = (x1: number, y1: number, x2: number, y2: number) => { const mx = x1 + Math.max(40, (x2 - x1) * 0.45); return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`; };

  const Item = ({ icon: Icon, label, onClick, danger }: { icon: LucideIcon; label: string; onClick: () => void; danger?: boolean }) => (
    <button className={danger ? "danger" : ""} onClick={() => { setMenu(null); onClick(); }}><Icon size={14} />{label}</button>
  );

  return (
    <div className="cv" ref={box} onPointerDown={(e) => down(e, "pan")} onPointerMove={move} onPointerUp={up} onDoubleClick={dbl}
      onContextMenu={(e) => { if (!(e.target as HTMLElement).closest(".cv-side, .cv-drawer")) openMenu(e); }}
      style={{ backgroundSize: `${grid}px ${grid}px`, backgroundPosition: `${view.x}px ${view.y}px` }}>
      <div className="cv-world" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})` }}>
        <svg className="cv-edges" style={{ overflow: "visible" }}>
          {lanes.flatMap((L) => roots(L).map((r) => {
            const a = D(L.key), b = P(r.id);
            return <path key={`d${r.id}`} d={curve(a.x + DEV_W, a.y + CARD_H / 2, b.x, b.y + CARD_H / 2)} className={`cv-edge dev ${r.status}`} />;
          }))}
          {shown.filter(sameLane).map((r) => {
            const a = P(r.parent_id!), b = P(r.id);
            return <path key={r.id} d={curve(a.x + CARD_W, a.y + CARD_H / 2, b.x, b.y + CARD_H / 2)} className={`cv-edge ${r.status}`} />;
          })}
          {shown.filter(crossLane).map((r) => {
            const a = P(r.parent_id!), b = P(r.id);
            return <path key={`x${r.id}`} d={curve(a.x + CARD_W / 2, a.y + CARD_H, b.x + CARD_W / 2, b.y)} className="cv-edge cross" />;
          })}
        </svg>
        {shown.filter(sameLane).map((r) => {
          const b = P(r.id);
          return <div key={`l${r.id}`} className="cv-edge-label" style={{ left: b.x - 14, top: b.y + CARD_H / 2, transform: "translate(-100%, -50%)" }} title="What changed from the parent run">{r.label}</div>;
        })}
        {lanes.map((L) => (
          <div key={L.key} className="cv-node dev" style={{ left: D(L.key).x, top: D(L.key).y, width: DEV_W }}>
            <DeviceCard lane={L} jobs={jobs.filter((j) => L.deviceId != null && j.device_id === L.deviceId)}
              onDown={(e) => down(e, "dev", L.key)} onMenu={(e) => openMenu(e, undefined, L)}
              onProfile={L.kind === "remote" && L.deviceId != null ? () => setProfileOn(L.key) : undefined} />
          </div>
        ))}
        {shown.map((r) => (
          <div key={r.id} className="cv-node" style={{ left: P(r.id).x, top: P(r.id).y }}>
            <Card run={r} idx={done.indexOf(r)} cmp={cmpFor(r)} selected={sel.indexOf(r.id) + 1}
              onDown={(e) => down(e, "card", String(r.id))} onClick={(e) => click(e, r.id)} onMenu={(e) => openMenu(e, r)} onBranch={() => setBranchFrom(r.id)} />
          </div>
        ))}
        {notes.map((n) => (
          <div key={n.id} className="cv-note" style={{ left: n.x, top: n.y }} onPointerDown={(e) => down(e, "note", n.id)}>
            <button className="cv-note-x" onClick={() => setNotes((ns) => ns.filter((x) => x.id !== n.id))}><X size={12} /></button>
            <textarea value={n.text} placeholder="Note…" onChange={(e) => setNotes((ns) => ns.map((x) => (x.id === n.id ? { ...x, text: e.target.value } : x)))} />
          </div>
        ))}
      </div>

      <div className="cv-topbar" onPointerDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
        <span className="cv-hint">Each device is a root: its runs grow from it · right-click or <b>+</b> to branch · shift-click two cards to compare · ⌃/⌘ + scroll to zoom</span>
        <div className="segmented"><button className={vs === "parent" ? "on" : ""} onClick={() => setVs("parent")}>Δ vs parent</button>
          <button className={vs === "baseline" ? "on" : ""} onClick={() => setVs("baseline")}>Δ vs baseline</button></div>
      </div>

      <div className="cv-tools" onPointerDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
        <button className="icon-btn" title="Zoom out" onClick={() => zoomAt(1 / 1.2)}><Minus /></button>
        <span className="cv-zoom">{Math.round(view.k * 100)}%</span>
        <button className="icon-btn" title="Zoom in" onClick={() => zoomAt(1.2)}><Plus /></button>
        <span className="cv-sep" />
        <button className="icon-btn" title="Fit everything" onClick={fit}><Maximize /></button>
        <button className="icon-btn" title="Add a note" onClick={() => addNote()}><StickyNote /></button>
        <button className="btn small ghost" title="Put cards back in tree order" onClick={() => { setPos({}); setTimeout(fit, 0); }}><Wand2 size={14} /> Tidy up</button>
        <button className="btn small ghost" title="Put another device on the canvas and profile setups on it" onClick={() => setAddOpen(!addOpen)}><Server size={14} /> Add device</button>
        {baseline && <button className="btn small" onClick={() => setBranchFrom(baseline.id)}><GitBranch size={14} /> Branch from baseline</button>}
        {addOpen && (
          <div className="cv-menu cv-add" style={{ left: "50%", bottom: 46, top: "auto", transform: "translateX(-50%)" }}>
            <div className="cv-menu-title">Add a device</div>
            {unpinned.map((d) => <button key={d.id} onClick={() => pin(d.id)}><Server size={14} />{d.name} <span className="muted">{d.url.replace(/^https?:\/\//, "")}</span></button>)}
            {unpinned.length === 0 && <div className="row-sub" style={{ padding: "4px 10px", maxWidth: 260 }}>{devices.length ? "Every registered device is already on the canvas." : "No other devices yet."} Register Blocky devices in <a href="#/settings">Settings → Remote access & devices</a>.</div>}
          </div>
        )}
      </div>

      {menu && (
        <div className="cv-menu" style={{ left: menu.x, top: menu.y }} onPointerDown={(e) => e.stopPropagation()} onContextMenu={(e) => e.preventDefault()}>
          {menu.run ? <>
            <div className="cv-menu-title">{runTitle(menu.run)}</div>
            <Item icon={GitBranch} label="Branch from here…" onClick={() => setBranchFrom(menu.run!.id)} />
            <Item icon={Repeat} label="Repeat this run" onClick={() => onRepeat(menu.run!)} />
            {menu.run.parent_id != null && byId.get(menu.run.parent_id) && <Item icon={GitCompare} label="Compare with its parent" onClick={() => setSel([menu.run!.parent_id!, menu.run!.id])} />}
            {baseline && menu.run.id !== baseline.id && <Item icon={GitCompare} label="Compare with baseline" onClick={() => setSel([baseline.id, menu.run!.id])} />}
            {sel.length === 1 && sel[0] !== menu.run.id && <Item icon={GitCompare} label="Compare with selected card" onClick={() => setSel([sel[0], menu.run!.id])} />}
            {lanes.filter((L) => L.kind === "remote" && L.deviceId != null).map((L) => (
              <Item key={L.key} icon={Server} label={`Profile this setup on ${L.name}`} onClick={() => api.remoteRun(exp.id, L.deviceId!, [menu.run!.id]).then(() => api.remoteJobs(exp.id).then(setJobs)).catch((e) => alert((e as Error).message))} />
            ))}
            <Item icon={Copy} label="Copy settings (JSON)" onClick={() => copySettings(menu.run!)} />
            {(menu.run.status === "running" || menu.run.status === "queued") && <Item icon={Square} label="Stop" onClick={() => onCancel(menu.run!)} />}
            {menu.run.status !== "running" && menu.run.status !== "queued" && <Item icon={Trash2} label="Delete run" danger onClick={() => { if (confirm(`Delete "${menu.run!.label}"? Its branches stay and attach to its parent.`)) onDelete(menu.run!); }} />}
          </> : menu.lane ? <>
            <div className="cv-menu-title">{menu.lane.name}</div>
            {menu.lane.kind === "local" && baseline && <Item icon={GitBranch} label="Branch from baseline…" onClick={() => setBranchFrom(baseline.id)} />}
            {menu.lane.kind === "remote" && menu.lane.deviceId != null && <Item icon={Play} label="Profile setups here…" onClick={() => setProfileOn(menu.lane!.key)} />}
            {roots(menu.lane)[0] && baseline && roots(menu.lane)[0].id !== baseline.id && <Item icon={GitCompare} label="Compare its first run with the baseline" onClick={() => setSel([baseline.id, roots(menu.lane!)[0].id])} />}
            {menu.lane.kind === "remote" && menu.lane.runs.length === 0 && <Item icon={X} label="Remove from canvas" onClick={() => setPinned((p) => p.filter((x) => x !== menu.lane!.deviceId))} />}
          </> : <>
            {baseline && <Item icon={GitBranch} label="Branch from baseline…" onClick={() => setBranchFrom(baseline.id)} />}
            <Item icon={Server} label="Add device…" onClick={() => setAddOpen(true)} />
            <Item icon={StickyNote} label="Add note here" onClick={() => addNote(menu.wx + 100, menu.wy + 40)} />
            <Item icon={Maximize} label="Fit to screen" onClick={fit} />
            <Item icon={Wand2} label="Tidy up" onClick={() => { setPos({}); setTimeout(fit, 0); }} />
          </>}
        </div>
      )}

      {profileLane ? (
        <ProfileHere exp={exp} lane={profileLane} setups={localSetups} onClose={() => setProfileOn(null)} onStarted={() => api.remoteJobs(exp.id).then(setJobs).catch(() => {})} />
      ) : branchRun ? (
        <div className="cv-drawer" onPointerDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()} onContextMenu={(e) => e.stopPropagation()}>
          <div className="cv-side-head"><GitBranch size={15} /><span>Branch from</span><b>{runTitle(branchRun)}</b>
            <button className="icon-btn" onClick={() => setBranchFrom(null)}><X /></button></div>
          <div className="cv-side-body">
            <div className="row-sub cv-drawer-note">The new run starts from this card's exact setup ({branchRun.config.model.replace(".gguf", "")}, ctx {branchRun.config.ctx / 1024}k, KV {branchRun.config.kv}). Change one thing and it's queued as a child of this card{branchRun.results.imported_from ? ", on this device" : ""}.</div>
            {branchPanel(branchRun, () => setBranchFrom(null))}
          </div>
        </div>
      ) : pair && pair[0] && pair[1] ? <Compare a={pair[0]} b={pair[1]} onClose={() => setSel([])} /> : null}
    </div>
  );
}
