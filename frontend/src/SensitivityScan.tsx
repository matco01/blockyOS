import { useEffect, useState } from "react";
import { AlertCircle, Loader2, Radar, Square, Trash2 } from "lucide-react";
import { api, LayerScan } from "./api";

/**
 * Layer sensitivity: change ONE layer at a time (quantize it low, or remove it), measure the drift (KL divergence) vs
 * the original. Tall bars = sensitive layers to protect; short bars = layers you can squeeze or maybe prune.
 */
export default function SensitivityScan({ model, onProtect, onPrune }: { model: string; onProtect: (layers: number[]) => void; onPrune: (layers: number[]) => void }) {
  const [scans, setScans] = useState<LayerScan[]>([]);
  const [mode, setMode] = useState<"quantize" | "prune">("quantize");
  const [qtype, setQtype] = useState("q2_k");
  const [sel, setSel] = useState<number | null>(null);
  const [topK, setTopK] = useState(4);
  const [err, setErr] = useState<string | null>(null);
  const active = scans.some((s) => s.status === "running" || s.status === "queued");

  useEffect(() => {
    const tick = () => api.scans(model).then(setScans).catch(() => {});
    tick();
    const t = setInterval(tick, active ? 4000 : 20000);
    return () => clearInterval(t);
  }, [model, active]);

  const scan = scans.find((s) => s.id === sel) ?? scans[0];
  const L = scan?.results.layers ?? [];
  const max = Math.max(...L.map((x) => x.excess_kld ?? 0), 1e-6);
  const ranked = [...L].sort((a, b) => (b.excess_kld ?? 0) - (a.excess_kld ?? 0));
  const top = ranked.slice(0, topK).map((x) => x.layer);
  const bottom = ranked.slice(-topK).map((x) => x.layer).reverse();
  const n = scan?.results.n_layers ?? 0;

  const start = async () => {
    setErr(null);
    try { await api.scanStart(model, mode, qtype); setScans(await api.scans(model)); setSel(null); } catch (e) { setErr((e as Error).message); }
  };

  return (
    <div className="sc">
      <div className="ws-sub"><b>Find sensitive layers</b>
        <span className="row-sub">Changes one layer at a time and measures the drift from the original. Run it on the high-precision source. 9B: about 1–2 h; 0.8B: a few minutes.</span></div>
      <div className="sc-start">
        <div className="segmented"><button className={mode === "quantize" ? "on" : ""} onClick={() => setMode("quantize")}>Quantize one layer</button>
          <button className={mode === "prune" ? "on" : ""} onClick={() => setMode("prune")}>Remove one layer</button></div>
        {mode === "quantize" && <select className="field select small-select" value={qtype} onChange={(e) => setQtype(e.target.value)}>
          {["q2_k", "q3_k", "q4_k", "iq4_xs"].map((t) => <option key={t}>{t}</option>)}</select>}
        <button className="btn small" disabled={active} onClick={start}><Radar size={14} /> {active ? "A scan is running…" : `Scan ${model.replace(".gguf", "")}`}</button>
        {scans.length > 1 && <select className="field select small-select" value={scan?.id ?? ""} onChange={(e) => setSel(Number(e.target.value))}>
          {scans.map((s) => <option key={s.id} value={s.id}>#{s.id} · {s.mode === "prune" ? "remove" : s.qtype} · {s.status}</option>)}</select>}
      </div>
      {err && <div className="error"><AlertCircle /> {err}</div>}

      {scan && (
        <div className="sc-result">
          <div className="sc-head">
            {scan.status === "running" || scan.status === "queued" ? <><Loader2 className="spin" size={14} /> <span>{scan.step ?? "starting"}…</span>
              <button className="btn small ghost" onClick={() => api.scanCancel(scan.id)}><Square size={13} /> Stop</button></>
              : <span>{scan.mode === "prune" ? "Each layer removed" : `Each layer at ${scan.qtype}`}, the rest at {scan.results.source_type} · {scan.status}{scan.error ? `: ${scan.error}` : ""}</span>}
            {scan.results.control && <span className="muted">noise floor (control) KLD {scan.results.control.kld?.toFixed(4)}</span>}
            {scan.status !== "running" && scan.status !== "queued" && <button className="icon-btn danger" title="Delete scan" onClick={() => api.scanDelete(scan.id).then(() => api.scans(model).then(setScans))}><Trash2 /></button>}
          </div>
          {L.length > 0 && (
            <>
              <div className="sc-chart" role="img" aria-label="KL divergence caused by each layer">
                {Array.from({ length: n }, (_, i) => {
                  const x = L.find((l) => l.layer === i);
                  const v = x?.excess_kld ?? 0;
                  return (
                    <div key={i} className={`sc-bar ${top.includes(i) ? "hot" : ""} ${bottom.includes(i) ? "cold" : ""} ${x ? "" : "pending"}`}
                      title={x ? `Layer ${i}: KLD +${v.toFixed(4)} over the control · same top token ${x.same_top?.toFixed(1)}% · PPL ${x.ppl?.toFixed(3)}` : `Layer ${i}: not measured yet`}>
                      <div style={{ height: `${x ? Math.max(2, (Math.max(0, v) / max) * 100) : 0}%` }} />
                      <span>{i}</span>
                    </div>
                  );
                })}
              </div>
              <div className="xlegend"><span><i className="sq" style={{ background: "var(--series-2)" }} />most sensitive {topK}</span>
                <span><i className="sq" style={{ background: "var(--series-4)" }} />least sensitive {topK}</span>
                <span><i className="sq" style={{ background: "var(--series-1)" }} />other layers</span>
                <span className="muted">bar = extra KL divergence vs the original, caused by that layer alone</span></div>
              {scan.status === "done" && (
                <div className="sc-actions">
                  <label className="xrep">Layers <select className="field select small-select" value={topK} onChange={(e) => setTopK(Number(e.target.value))}>{[2, 4, 6, 8].map((k) => <option key={k}>{k}</option>)}</select></label>
                  <button className="btn small" onClick={() => onProtect(top)}>Protect most sensitive ({top.join(", ")}) at 8-bit</button>
                  {scan.mode === "prune" && <button className="btn small ghost" onClick={() => onPrune(bottom)}>Try removing least sensitive ({bottom.join(", ")})</button>}                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
