import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, Hammer, Loader2, Play, Square, Trash2 } from "lucide-react";
import { api, ModelFile, QuantRun, Status } from "./api";
import { Term } from "./ui";

const gb = (b: number) => `${(b / 2 ** 30).toFixed(2)} GB`;

/** One row per compressed version: what you save vs. what you lose, against the most precise one. */
function Results({ run }: { run: QuantRun }) {
  const ref = run.results.find((r) => r.reference);
  const rows = [...run.results].sort((a, b) => b.bits - a.bits);
  const maxSize = Math.max(...rows.map((r) => r.size_bytes), 1);
  const maxTg = Math.max(...rows.map((r) => r.tg_tps ?? 0), 1);
  // Sweet spot: the smallest version that still agrees with the reference on ≥ 95% of next-word picks.
  const sweet = [...rows].filter((r) => (r.same_top ?? 0) >= 95).sort((a, b) => a.size_bytes - b.size_bytes)[0];
  return (
    <div className="ql-results">
      {rows.map((r) => {
        const smaller = ref ? 1 - r.size_bytes / ref.size_bytes : 0;
        const faster = ref?.tg_tps && r.tg_tps ? r.tg_tps / ref.tg_tps - 1 : null;
        return (
          <div key={r.model} className={`ql-row ${sweet?.model === r.model && !r.reference ? "sweet" : ""}`}>
            <div className="ql-head">
              <b>{r.quant ?? r.model}</b>
              {r.reference && <span className="badge">reference (most precise)</span>}
              {sweet?.model === r.model && !r.reference && <span className="badge you">sweet spot</span>}
              {r.state !== "done" && <span className="row-sub"><Loader2 className="spin" size={12} /> {r.state}…</span>}
            </div>
            <div className="ql-metrics">
              <div><span>Size</span><div className="ql-track"><i style={{ width: `${(r.size_bytes / maxSize) * 100}%`, background: "var(--series-3)" }} /></div>
                <b>{gb(r.size_bytes)}</b>{!r.reference && <em>{Math.round(smaller * 100)}% smaller</em>}</div>
              <div><span><Term k="quant">Same top word</Term></span><div className="ql-track"><i style={{ width: `${r.same_top ?? 0}%`, background: "var(--series-1)" }} /></div>
                <b>{r.same_top != null ? `${r.same_top.toFixed(1)}%` : "…"}</b>{!r.reference && r.same_top != null && <em>{r.same_top >= 97 ? "almost identical" : r.same_top >= 92 ? "small drift" : "noticeably different"}</em>}</div>
              <div><span>Writing speed</span><div className="ql-track"><i style={{ width: `${((r.tg_tps ?? 0) / maxTg) * 100}%`, background: "var(--series-4)" }} /></div>
                <b>{r.tg_tps ? `${r.tg_tps.toFixed(0)} tok/s` : "…"}</b>{faster != null && !r.reference && <em>{faster >= 0 ? `${Math.round(faster * 100)}% faster` : `${Math.round(-faster * 100)}% slower`}</em>}</div>
            </div>
            <div className="ql-pro mono">
              perplexity {r.ppl?.toFixed(3) ?? "…"}{r.ppl_err ? ` ± ${r.ppl_err.toFixed(3)}` : ""}
              {!r.reference && <> · mean KLD {r.kld?.toFixed(4) ?? "…"}{r.kld_99 != null ? ` · 99% KLD ${r.kld_99.toFixed(3)}` : ""}</>}
              {r.pp_tps ? ` · prompt ${Math.round(r.pp_tps)} tok/s` : ""}
            </div>
          </div>
        );
      })}
      <div className="row-sub">
        <b>Same top word</b> = how often this version picks the same most-likely next word as the reference, on identical Wikipedia text.
        <b> Perplexity</b> = how surprised the model is by real text (lower is better). <b>KLD</b> = how far its word probabilities drift from the reference (0 = identical).
        All measured with llama.cpp's own tools on 4,096 tokens.
      </div>
    </div>
  );
}

export default function QuantLab({ models, status }: { models: ModelFile[]; status: Status | null }) {
  const [runs, setRuns] = useState<QuantRun[]>([]);
  const [jobs, setJobs] = useState<{ id: string; output: string; status: string; progress: number; error: string | null; note: string | null }[]>([]);
  const [types, setTypes] = useState<string[]>([]);
  const [picked, setPicked] = useState<string[]>([]);
  const [src, setSrc] = useState("");
  const [qtype, setQtype] = useState("Q3_K_M");
  const [view, setView] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => api.quantLab().then((d) => { setRuns(d.runs); setJobs(d.jobs); setTypes(d.types); setView((v) => v ?? d.runs[0]?.id ?? null); }).catch(() => {}), []);
  useEffect(() => { load(); }, [load]);
  const running = runs.find((r) => r.status === "running" || r.status === "queued");
  const busyJob = jobs.some((j) => j.status === "running");
  useEffect(() => {
    const t = setInterval(load, running || busyJob ? 1500 : 10000);
    return () => clearInterval(t);
  }, [running, busyJob, load]);

  // Versions of the same model share architecture + layer count.
  const families = useMemo(() => {
    const g: Record<string, ModelFile[]> = {};
    for (const m of models) (g[`${m.architecture}-${m.layers}-${m.size_label}`] ??= []).push(m);
    return Object.values(g);
  }, [models]);
  const shown = runs.find((r) => r.id === view) ?? running ?? null;

  const start = async () => {
    setErr(null);
    try { const { id } = await api.quantStart(picked); setView(id); load(); } catch (e) { setErr((e as Error).message); }
  };
  const make = async () => {
    setErr(null);
    try { await api.quantMake(src || models[0]?.file, qtype); load(); } catch (e) { setErr((e as Error).message); }
  };

  return (
    <div className="quantlab">
      <p className="page-sub" style={{ marginTop: 16 }}>
        <Term k="quant">Quantization</Term> shrinks a model by storing its numbers with fewer bits. Smaller files fit on smaller GPUs and run faster, but lose a little accuracy. This lab measures exactly how much, for each version.
      </p>

      <div className="group-label">Compare versions of one model</div>
      <div className="group pad">
        {families.map((fam) => (
          <div key={fam[0].file} className="ql-family">
            <div className="row-sub"><b>{fam[0].name ?? fam[0].file}</b> · {fam.length} version{fam.length > 1 ? "s" : ""} on this computer{fam.length < 2 ? " (get or make another to compare)" : ""}</div>
            <div className="chips">
              {fam.map((m) => (
                <button key={m.file} disabled={fam.length < 2} className={`chip ${picked.includes(m.file) ? "on" : ""}`}
                  onClick={() => setPicked((p) => {
                    const sameFam = p.filter((f) => fam.some((x) => x.file === f));
                    return sameFam.includes(m.file) ? sameFam.filter((f) => f !== m.file) : [...sameFam, m.file];
                  })}>{m.quant} · {gb(m.size_bytes)}</button>
              ))}
            </div>
          </div>
        ))}
        <div className="ql-actions">
          <span className="row-sub">Takes about 1–2 minutes per version. Chat pauses meanwhile.</span>
          {running ? <button className="btn ghost" onClick={() => api.quantCancel(running.id).then(load)}><Square /> Stop</button>
            : <button className="btn" disabled={picked.length < 2 || !!status?.busy} onClick={start}><Play /> Compare {picked.length || ""}</button>}
        </div>
        {err && <div className="error"><AlertCircle /> {err}</div>}
      </div>

      {shown && (
        <>
          <div className="group-label">Results · {shown.results[0]?.model.replace(/-(UD-)?(IQ|Q|BF|F)\d.*$/, "") ?? ""}{shown.status === "error" ? " · failed" : ""}</div>
          <div className="group pad">
            {shown.error && <div className="error"><AlertCircle /> {shown.error}</div>}
            {shown.results.length > 0 ? <Results run={shown} /> : <div className="row-sub"><Loader2 className="spin" size={13} /> Preparing (downloading the test text the first time)…</div>}
          </div>
        </>
      )}

      <div className="group-label">Make a smaller version yourself</div>
      <div className="group pad">
        <div className="ql-make">
          <select className="field select" value={src || models[0]?.file || ""} onChange={(e) => setSrc(e.target.value)}>
            {models.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file} · {m.quant}</option>)}
          </select>
          <span className="row-sub">→</span>
          <select className="field select" value={qtype} onChange={(e) => setQtype(e.target.value)}>
            {types.map((t) => <option key={t}>{t}</option>)}
          </select>
          <button className="btn" onClick={make}><Hammer /> Make it</button>
        </div>
        <div className="row-sub">Uses llama.cpp's quantizer. Best from a full-precision (F16/BF16) file; compressing an already-compressed file works but loses a bit more.</div>
        {jobs.map((j) => (
          <div key={j.id} className="ql-job">
            <span className="mono small">{j.output}</span>
            {j.status === "running" ? <div className="mem-bar"><div className="ok" style={{ width: `${j.progress * 100}%` }} /></div>
              : <span className={`status-tag ${j.status}`}>{j.status}</span>}
            {j.error && <span className="row-sub" style={{ color: "var(--danger)" }}>{j.error}</span>}
          </div>
        ))}
      </div>

      {runs.length > 1 && (
        <>
          <div className="group-label">Earlier comparisons</div>
          <div className="group">
            {runs.map((r) => (
              <div key={r.id} className={`row hist-row ${shown?.id === r.id ? "selected" : ""}`} onClick={() => setView(r.id)}>
                <div className="row-main"><div>{r.results.map((x) => x.quant).join(" vs ") || r.models.join(", ")} <span className={`status-tag ${r.status}`}>{r.status}</span></div>
                  <div className="row-sub">#{r.id} · {new Date(r.created_at * 1000).toLocaleString()}</div></div>
                <button className="icon-btn danger" onClick={(e) => { e.stopPropagation(); api.quantDelete(r.id).then(load); }}><Trash2 /></button>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
