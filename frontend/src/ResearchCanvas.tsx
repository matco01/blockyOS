import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Maximize, Minus, Plus, StickyNote, X } from "lucide-react";
import { api, CanvasNote, CanvasState, Experiment, ExpConfig, ExpRun } from "./api";
import { colorOf, delta, METRICS, n1 } from "./expmetrics";

/**
 * Research canvas: an infinite, zoomable map of an experiment. Every run is a card; lines show what each variant
 * changed relative to the baseline. Drag cards, add notes, shift-click two cards to compare them head-to-head.
 */

const CARD_W = 270;
const CARD_H = 168;
const KEY_NAMES: Partial<Record<keyof ExpConfig, string>> = { model: "Weights", kv: "KV cache", ctx: "Context", vram_limit_mb: "VRAM limit", ngl: "GPU layers", fa: "Flash attention", threads: "Threads" };

type View = { x: number; y: number; k: number };

function changedKeys(base: ExpConfig, cfg: ExpConfig): (keyof ExpConfig)[] {
  return (Object.keys(KEY_NAMES) as (keyof ExpConfig)[]).filter((k) => String(cfg[k] ?? "") !== String(base[k] ?? ""));
}

/** Baseline on the left; variants fan out to one column on the right, grouped by the setting they changed. */
type Group = { label: string; x: number; y: number; ids: number[] };

function autoLayout(runs: ExpRun[]): { pos: Record<string, { x: number; y: number }>; groups: Group[] } {
  const base = runs.find((r) => r.status === "done") ?? runs[0];
  if (!base) return { pos: {}, groups: [] };
  const groups = new Map<string, ExpRun[]>();
  for (const r of runs) {
    if (r.id === base.id) continue;
    const ks = changedKeys(base.config, r.config);
    const g = ks.length === 0 ? "Repeats of the baseline" : ks.length === 1 ? `${KEY_NAMES[ks[0]]}` : "Several changes at once";
    groups.set(g, [...(groups.get(g) ?? []), r]);
  }
  // A tree: baseline on top, one branch per changed setting, that branch's runs stacked under it.
  const pos: Record<string, { x: number; y: number }> = {};
  const labels: { label: string; x: number; y: number; ids: number[] }[] = [];
  const GX = CARD_W + 70, HY = CARD_H + 110;
  Array.from(groups.entries()).forEach(([label, rs], i) => {
    labels.push({ label, x: i * GX, y: HY, ids: rs.map((r) => r.id) });
    rs.forEach((r, j) => { pos[r.id] = { x: i * GX, y: HY + 48 + j * (CARD_H + 22) }; });
  });
  const width = Math.max(1, groups.size) * GX - 70;
  pos[base.id] = { x: (width - CARD_W) / 2, y: 0 };
  return { pos, groups: labels };
}

function Card({ run, idx, base, selected, onDown, onClick }: { run: ExpRun; idx: number; base: ExpRun | undefined; selected: number;
  onDown: (e: React.PointerEvent) => void; onClick: (e: React.MouseEvent) => void }) {
  const r = run.results;
  const isBase = base?.id === run.id;
  const tg = METRICS.find((m) => m.key === "tg")!, mm = METRICS.find((m) => m.key === "mmlu")!, kl = METRICS.find((m) => m.key === "kld")!, vr = METRICS.find((m) => m.key === "vram_model")!;
  const row = (m: typeof tg, label: string) => {
    const v = m.get(r);
    const d = !isBase && base && run.status === "done" ? delta(m, { run, base }) : null;
    return (
      <div className="cv-metric"><span>{label}</span><b>{v == null ? "–" : (m.fmt ?? n1)(v)}<small>{m.unit}</small></b>
        {d && <em className={`d ${d.tone}`}>{d.text}</em>}</div>
    );
  };
  return (
    <div className={`cv-card ${run.status} ${selected ? "sel" : ""}`} onPointerDown={onDown} onClick={onClick} style={{ "--c": colorOf(idx) } as React.CSSProperties}>
      <div className="cv-head">
        <span className="xdot" style={{ background: colorOf(idx) }} />
        <b title={run.label}>{isBase ? "Baseline" : run.label}</b>
        {selected ? <span className="cv-pick">{selected === 1 ? "A" : "B"}</span> : null}
      </div>
      {run.status === "done" ? (
        <>
          {row(tg, "Decode")}
          {row(mm, "MMLU-Pro")}
          {row(kl, "KLD")}
          {row(vr, "VRAM")}
          {r.hw?.limiter && <div className="cv-lim">limited by {r.hw.limiter.what.toLowerCase()}</div>}
        </>
      ) : (
        <div className="cv-status">
          {run.status === "running" ? <><Loader2 className="spin" size={14} /> {run.phase ?? "starting"}…</> : run.status === "queued" ? "In the queue" : `${run.status}${run.error ? `: ${run.error.slice(0, 80)}` : ""}`}
        </div>
      )}
    </div>
  );
}

function Compare({ a, b, onClose }: { a: ExpRun; b: ExpRun; onClose: () => void }) {
  const rows = METRICS.filter((m) => m.get(a.results) != null || m.get(b.results) != null);
  return (
    <div className="cv-compare" onPointerDown={(e) => e.stopPropagation()} onWheel={(e) => e.stopPropagation()}>
      <div className="cv-compare-head"><b>A: {a.label}</b><span>vs</span><b>B: {b.label}</b><button className="icon-btn" onClick={onClose}><X /></button></div>
      <div className="cv-compare-body">
        <table className="data">
          <thead><tr><th>Metric</th><th className="r">A</th><th className="r">B</th><th className="r">B vs A</th></tr></thead>
          <tbody>{rows.map((m) => {
            const va = m.get(a.results), vb = m.get(b.results);
            const d = delta(m, { run: b, base: a });
            return <tr key={m.key}><td>{m.label} <span className="u">{m.unit}</span></td><td className="r">{va == null ? "–" : (m.fmt ?? n1)(va)}</td>
              <td className="r">{vb == null ? "–" : (m.fmt ?? n1)(vb)}</td><td className="r">{d ? <span className={`d ${d.tone}`}>{d.text}</span> : "–"}</td></tr>;
          })}</tbody>
        </table>
        <div className="row-sub">Colors use the same noise rules as the results table (MMLU-Pro significance is only computed against the baseline, so here it falls back to a 2% threshold).</div>
      </div>
    </div>
  );
}

export default function ResearchCanvas({ exp, runs }: { exp: Experiment; runs: ExpRun[] }) {
  const box = useRef<HTMLDivElement>(null);
  const saved = exp.canvas;
  const [view, setView] = useState<View>(saved?.view ?? { x: 80, y: 80, k: 0.9 });
  const [pos, setPos] = useState<CanvasState["pos"]>(saved?.pos ?? {});
  const [notes, setNotes] = useState<CanvasNote[]>(saved?.notes ?? []);
  const [sel, setSel] = useState<number[]>([]);
  const drag = useRef<{ kind: "pan" | "card" | "note"; id?: string; sx: number; sy: number; ox: number; oy: number; moved: boolean } | null>(null);
  const done = runs.filter((r) => r.status === "done");
  const base = done[0];
  const shown = runs.filter((r) => r.status !== "cancelled");

  // Cards without a saved position get an automatic one.
  const auto = useMemo(() => autoLayout(shown), [shown.map((r) => r.id).join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  const P = (id: number) => pos[id] ?? auto.pos[id] ?? { x: 0, y: 0 };
  const untouched = shown.every((r) => !pos[r.id]);

  // Persist (debounced) to the Blocky box, so the map survives reloads and other browsers see it.
  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    const t = setTimeout(() => { api.experimentCanvas(exp.id, { pos, notes, view }).catch(() => {}); }, 700);
    return () => clearTimeout(t);
  }, [pos, notes, view, exp.id]);

  const fit = useCallback(() => {
    const el = box.current;
    if (!el || shown.length === 0) return;
    const ps = shown.map((r) => P(r.id));
    const all = [...ps, ...notes.map((n) => ({ x: n.x, y: n.y }))];
    const x0 = Math.min(...all.map((p) => p.x)), y0 = Math.min(...all.map((p) => p.y));
    const x1 = Math.max(...all.map((p) => p.x + CARD_W)), y1 = Math.max(...all.map((p) => p.y + CARD_H));
    const k = Math.min(1.2, Math.max(0.25, Math.min((el.clientWidth - 120) / (x1 - x0), (el.clientHeight - 140) / (y1 - y0))));
    setView({ k, x: (el.clientWidth - (x1 - x0) * k) / 2 - x0 * k, y: (el.clientHeight - (y1 - y0) * k) / 2 - y0 * k });
  }, [shown, notes, pos, auto]); // eslint-disable-line react-hooks/exhaustive-deps
  // Open fitted, unless the researcher arranged the map by hand (then restore their exact viewport).
  useEffect(() => { if (!saved?.view || Object.keys(saved?.pos ?? {}).length === 0) fit(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const zoomAt = (factor: number, cx?: number, cy?: number) => setView((v) => {
    const el = box.current!;
    const px = cx ?? el.clientWidth / 2, py = cy ?? el.clientHeight / 2;
    const k = Math.min(2.5, Math.max(0.15, v.k * factor));
    return { k, x: px - ((px - v.x) * k) / v.k, y: py - ((py - v.y) * k) / v.k };
  });

  // Wheel: pan; ctrl/⌘ + wheel or trackpad pinch: zoom around the cursor. Native listener so we can preventDefault.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const r = el.getBoundingClientRect();
        zoomAt(Math.exp(-e.deltaY * 0.01), e.clientX - r.left, e.clientY - r.top);
      } else setView((v) => ({ ...v, x: v.x - e.deltaX, y: v.y - e.deltaY }));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const down = (e: React.PointerEvent, kind: "pan" | "card" | "note", id?: string) => {
    if (e.button !== 0) return;
    if ((e.target as HTMLElement).closest("textarea, button")) return;
    e.stopPropagation();
    const o = kind === "pan" ? { x: view.x, y: view.y } : kind === "card" ? P(Number(id)) : notes.find((n) => n.id === id)!;
    drag.current = { kind, id, sx: e.clientX, sy: e.clientY, ox: o.x, oy: o.y, moved: false };
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
  };
  const move = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.sx, dy = e.clientY - d.sy;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    if (d.kind === "pan") setView((v) => ({ ...v, x: d.ox + dx, y: d.oy + dy }));
    else if (d.kind === "card") setPos((p) => ({ ...p, [d.id!]: { x: d.ox + dx / view.k, y: d.oy + dy / view.k } }));
    else setNotes((ns) => ns.map((n) => (n.id === d.id ? { ...n, x: d.ox + dx / view.k, y: d.oy + dy / view.k } : n)));
  };
  const up = () => { setTimeout(() => { drag.current = null; }, 0); };

  const click = (e: React.MouseEvent, id: number) => {
    if (drag.current?.moved) return;
    setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : e.shiftKey ? [...s.slice(-1), id] : [id]));
  };
  const addNote = (x?: number, y?: number) => {
    const el = box.current!;
    const wx = x ?? (el.clientWidth / 2 - view.x) / view.k, wy = y ?? (el.clientHeight / 2 - view.y) / view.k;
    setNotes((ns) => [...ns, { id: Math.random().toString(36).slice(2, 9), x: wx - 100, y: wy - 40, text: "" }]);
  };
  const dbl = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest(".cv-card, .cv-note")) return;
    const r = box.current!.getBoundingClientRect();
    addNote((e.clientX - r.left - view.x) / view.k, (e.clientY - r.top - view.y) / view.k);
  };

  const pair = sel.length === 2 ? [runs.find((r) => r.id === sel[0]), runs.find((r) => r.id === sel[1])] : null;
  const grid = 24 * view.k;

  return (
    <div className="cv" ref={box} onPointerDown={(e) => down(e, "pan")} onPointerMove={move} onPointerUp={up} onDoubleClick={dbl}
      style={{ backgroundSize: `${grid}px ${grid}px`, backgroundPosition: `${view.x}px ${view.y}px` }}>
      <div className="cv-world" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})` }}>
        {untouched && base ? (
          /* Tree: baseline → one branch per changed setting → that branch's runs on a spine. */
          <>
            <svg className="cv-edges" style={{ overflow: "visible" }}>
              {auto.groups.map((g) => {
                const a = P(base.id);
                const x1 = a.x + CARD_W / 2, y1 = a.y + CARD_H, x2 = g.x + CARD_W / 2, y2 = g.y;
                const my = (y1 + y2) / 2;
                const last = P(g.ids[g.ids.length - 1]);
                const sx = g.x - 18;
                return (
                  <g key={g.label}>
                    <path d={`M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`} className="cv-edge" />
                    <path d={`M${x2},${y2 + 30} L${x2},${y2 + 40} L${sx},${y2 + 40} L${sx},${last.y + CARD_H / 2}`} className="cv-edge" />
                    {g.ids.map((id) => <path key={id} d={`M${sx},${P(id).y + CARD_H / 2} L${g.x},${P(id).y + CARD_H / 2}`} className={`cv-edge ${runs.find((r) => r.id === id)?.status ?? ""}`} />)}
                  </g>
                );
              })}
            </svg>
            {auto.groups.map((g) => (
              <div key={g.label} className="cv-group" style={{ left: g.x, top: g.y, width: CARD_W }}>
                <span>{g.label}</span><em>{g.ids.length} run{g.ids.length === 1 ? "" : "s"}</em>
              </div>
            ))}
          </>
        ) : (
          /* Hand-arranged: a line from the baseline to every run, labeled with what changed. */
          <>
            <svg className="cv-edges" style={{ overflow: "visible" }}>
              {base && shown.filter((r) => r.id !== base.id).map((r) => {
                const a = P(base.id), b = P(r.id);
                const x1 = a.x + CARD_W, y1 = a.y + CARD_H / 2, x2 = b.x, y2 = b.y + CARD_H / 2;
                const mx = (x1 + x2) / 2;
                return <path key={r.id} d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`} className={`cv-edge ${r.status}`} />;
              })}
            </svg>
            {base && shown.filter((r) => r.id !== base.id).map((r) => {
              const b = P(r.id);
              const ks = changedKeys(base.config, r.config);
              const text = ks.length === 0 ? "same setup (repeat)" : ks.map((k) => `${KEY_NAMES[k]} → ${k === "vram_limit_mb" ? (r.config.vram_limit_mb ? `${r.config.vram_limit_mb / 1024} GB` : "full") : k === "model" ? r.config.model.replace(".gguf", "").split("-").pop() : String(r.config[k])}`).join(", ");
              return <div key={`l${r.id}`} className="cv-edge-label" style={{ left: b.x - 12, top: b.y + CARD_H / 2, transform: "translate(-100%, -50%)" }}>{text}</div>;
            })}
          </>
        )}        {shown.map((r) => (
          <div key={r.id} className="cv-node" style={{ left: P(r.id).x, top: P(r.id).y }}>
            <Card run={r} idx={done.indexOf(r)} base={base} selected={sel.indexOf(r.id) + 1}
              onDown={(e) => down(e, "card", String(r.id))} onClick={(e) => click(e, r.id)} />
          </div>
        ))}
        {notes.map((n) => (
          <div key={n.id} className="cv-note" style={{ left: n.x, top: n.y }} onPointerDown={(e) => down(e, "note", n.id)}>
            <button className="cv-note-x" onClick={() => setNotes((ns) => ns.filter((x) => x.id !== n.id))}><X size={12} /></button>
            <textarea value={n.text} placeholder="Note…" onChange={(e) => setNotes((ns) => ns.map((x) => (x.id === n.id ? { ...x, text: e.target.value } : x)))} />
          </div>
        ))}
      </div>

      <div className="cv-hint">Drag to move · ⌃/⌘ + scroll or pinch to zoom · double-click for a note · shift-click two cards to compare</div>
      <div className="cv-tools" onPointerDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
        <button className="icon-btn" title="Zoom out" onClick={() => zoomAt(1 / 1.2)}><Minus /></button>
        <span className="cv-zoom">{Math.round(view.k * 100)}%</span>
        <button className="icon-btn" title="Zoom in" onClick={() => zoomAt(1.2)}><Plus /></button>
        <span className="cv-sep" />
        <button className="icon-btn" title="Fit everything" onClick={fit}><Maximize /></button>
        <button className="icon-btn" title="Add a note" onClick={() => addNote()}><StickyNote /></button>
        <button className="btn small ghost" title="Put cards back in their automatic places" onClick={() => { setPos({}); setTimeout(fit, 0); }}>Tidy up</button>
      </div>
      {pair && pair[0] && pair[1] && <Compare a={pair[0]} b={pair[1]} onClose={() => setSel([])} />}
    </div>
  );
}
