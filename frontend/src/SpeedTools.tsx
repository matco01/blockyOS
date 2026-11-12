import { useCallback, useEffect, useState } from "react";
import { AlertCircle, Download, Loader2, Square, Users, Zap } from "lucide-react";
import { api, ModelFile, Status, ToolRun } from "./api";
import { LineChart } from "./ui";

function useRuns(kind: "throughput" | "speculative") {
  const [runs, setRuns] = useState<ToolRun[]>([]);
  const load = useCallback(() => api.toolRuns(kind).then(setRuns).catch(() => {}), [kind]);
  const running = runs.find((r) => r.status === "running" || r.status === "queued");
  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    const t = setInterval(load, running ? 1500 : 10000);
    return () => clearInterval(t);
  }, [running, load]);
  return { runs, load, running };
}

/** How many people can use the machine at once? */
export function Throughput({ models, active, status }: { models: ModelFile[]; active?: string | null; status: Status | null }) {
  const { runs, load, running } = useRuns("throughput");
  const [model, setModel] = useState(active ?? models[0]?.file ?? "");
  const [err, setErr] = useState<string | null>(null);
  const last = running ?? runs.find((r) => r.status === "done");
  const rows = (last?.results ?? []) as { users: number; total_gen_tps: number; per_user_tps: number; power_w?: number }[];
  const comfy = rows.filter((r) => r.per_user_tps >= 8).sort((a, b) => b.users - a.users)[0];

  const start = async () => {
    setErr(null);
    try { await api.toolStart("throughput", { model: model || active, users: [1, 2, 4, 8, 16] }); load(); } catch (e) { setErr((e as Error).message); }
  };

  return (
    <div>
      <p className="page-sub" style={{ marginTop: 16 }}>
        Several people (or apps) can use one model at the same time. Total output goes up, but each person's answer gets slower. This finds the sweet spot for your hardware.
      </p>
      <div className="group pad" style={{ marginTop: 16 }}>
        <div className="tool-row">
          <select className="field select" value={model || active || ""} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file} · {m.quant}</option>)}
          </select>
          <span className="row-sub">Tests 1, 2, 4, 8 and 16 people at once (512-word question, 128-word answer each).</span>
          {running ? <button className="btn ghost" onClick={() => api.toolCancel(running.id).then(load)}><Square /> Stop</button>
            : <button className="btn" disabled={!!status?.busy} onClick={start}><Users /> Run</button>}
        </div>
        {err && <div className="error"><AlertCircle /> {err}</div>}
        {last?.error && <div className="error"><AlertCircle /> {last.error}</div>}
      </div>

      {rows.length > 0 && (
        <>
          <div className="group-label">Results{running ? " (so far)" : ""} · {String(last?.params.model ?? "").replace(".gguf", "")}</div>
          <div className="group pad">
            {comfy && (
              <div className="tool-verdict">
                Up to <b>{comfy.users} people</b> can use it at once while each still gets <b>{comfy.per_user_tps.toFixed(0)} tok/s</b>, faster than reading speed.
              </div>
            )}
            <div className="tool-charts">
              <div><div className="mini-title">Total output, everyone combined <span className="muted">(tok/s)</span></div>
                <LineChart points={rows.map((r) => ({ x: r.users, y: r.total_gen_tps }))} height={140} color="var(--series-1)" unit="tok/s" xLabel="people at once" fmtX={(x) => `${x} people`} /></div>
              <div><div className="mini-title">Speed each person gets <span className="muted">(tok/s)</span></div>
                <LineChart points={rows.map((r) => ({ x: r.users, y: r.per_user_tps }))} height={140} color="var(--series-2)" unit="tok/s" xLabel="people at once" fmtX={(x) => `${x} people`} /></div>
            </div>
            <table className="data">
              <thead><tr><th>People</th><th className="r">Total tok/s</th><th className="r">Per person</th><th className="r">GPU power</th></tr></thead>
              <tbody>{rows.map((r) => (
                <tr key={r.users}><td>{r.users}</td><td className="r b">{r.total_gen_tps.toFixed(0)}</td><td className="r">{r.per_user_tps.toFixed(1)}</td><td className="r">{r.power_w ? `${r.power_w} W` : "–"}</td></tr>
              ))}</tbody>
            </table>
          </div>
        </>
      )}
      {running && rows.length === 0 && <div className="row-sub" style={{ marginTop: 12 }}><Loader2 className="spin" size={13} /> Loading the model and running…</div>}
    </div>
  );
}

// Tiny same-family models that work as drafts (they share the vocabulary).
const DRAFTS: Record<string, { repo: string; file: string; title: string }> = {
  qwen35: { repo: "unsloth/Qwen3.5-0.8B-GGUF", file: "Qwen3.5-0.8B-Q8_0.gguf", title: "Qwen3.5 0.8B (0.8 GB)" },
  qwen3: { repo: "unsloth/Qwen3-0.6B-GGUF", file: "Qwen3-0.6B-Q8_0.gguf", title: "Qwen3 0.6B (0.6 GB)" },
};

/** Speculative decoding: does a tiny draft model make the big one faster? */
export function Speculative({ models, active, status }: { models: ModelFile[]; active?: string | null; status: Status | null }) {
  const { runs, load, running } = useRuns("speculative");
  const [model, setModel] = useState(active ?? "");
  const [draft, setDraft] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const target = models.find((m) => m.file === (model || active));
  const candidates = models.filter((m) => target && m.architecture === target.architecture && m.size_bytes < target.size_bytes * 0.4);
  const rec = target?.architecture ? DRAFTS[target.architecture] : undefined;
  const last = running ?? runs.find((r) => r.status === "done") ?? runs[0];
  const res = (last?.results ?? []) as { mode: string; tps: number; acceptance?: number; speedup?: number }[];
  const withDraft = res.find((r) => r.mode === "with draft");

  const start = async () => {
    setErr(null);
    try { await api.toolStart("speculative", { model: target?.file, draft: draft || candidates[0]?.file }); load(); } catch (e) { setErr((e as Error).message); }
  };

  return (
    <div>
      <p className="page-sub" style={{ marginTop: 16 }}>
        A tiny model from the same family quickly guesses the next few words; the big model then checks them all in one go. When the guesses are right, the big model writes several words per step. Same answers, potentially much faster.
      </p>
      <div className="group pad" style={{ marginTop: 16 }}>
        <div className="tool-row">
          <select className="field select" value={model || active || ""} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file} · {m.quant}</option>)}
          </select>
          <span className="row-sub">draft:</span>
          {candidates.length > 0 ? (
            <select className="field select" value={draft || candidates[0].file} onChange={(e) => setDraft(e.target.value)}>
              {candidates.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file}</option>)}
            </select>
          ) : rec ? (
            <button className="btn small ghost" onClick={() => api.download(rec.repo, rec.file).then(() => setErr("Downloading the draft model; check Models → Get more. Come back when it's done."))}>
              <Download /> Get {rec.title}
            </button>
          ) : <span className="row-sub">No small model of the same family on this computer.</span>}
          {running ? <button className="btn ghost" onClick={() => api.toolCancel(running.id).then(load)}><Square /> Stop</button>
            : <button className="btn" disabled={!!status?.busy || candidates.length === 0} onClick={start}><Zap /> Measure</button>}
        </div>
        {err && <div className="row-sub" style={{ marginTop: 8 }}>{err}</div>}
        {last?.error && <div className="error"><AlertCircle /> {last.error}</div>}
      </div>

      {res.length > 0 && (
        <>
          <div className="group-label">Result · {String(last?.params.model ?? "").replace(".gguf", "")} + {String(last?.params.draft ?? "").replace(".gguf", "")}</div>
          <div className="group pad">
            {withDraft?.speedup && (
              <div className="tool-verdict">
                {withDraft.speedup >= 1.05 ? <>The draft makes it <b>{withDraft.speedup.toFixed(2)}× faster</b></> : <>The draft <b>doesn't help</b> here ({withDraft.speedup.toFixed(2)}×)</>}
                {withDraft.acceptance != null && <>: the big model accepted <b>{Math.round(withDraft.acceptance * 100)}%</b> of the tiny model's guesses.</>}
              </div>
            )}
            {res.map((r) => (
              <div className="memo-line" key={r.mode}><span>{r.mode === "with draft" ? "With draft" : "Normal"}</span>
                <div className="memo-track"><div style={{ width: `${(r.tps / Math.max(...res.map((x) => x.tps || 0), 1)) * 100}%`, background: r.mode === "with draft" ? "var(--series-1)" : "var(--series-3)" }} /></div>
                <b>{r.tps?.toFixed(0)}</b></div>
            ))}
            <div className="row-sub">Tokens per second over 4 fixed prompts with greedy decoding, so both runs write the same text.</div>
          </div>
        </>
      )}
      {running && res.length === 0 && <div className="row-sub" style={{ marginTop: 12 }}><Loader2 className="spin" size={13} /> Measuring without the draft first…</div>}
    </div>
  );
}
