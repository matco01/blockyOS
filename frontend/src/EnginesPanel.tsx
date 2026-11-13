import { useEffect, useState } from "react";
import { AlertCircle, Check, Cpu, Loader2, Square, Trash2 } from "lucide-react";
import { api, Engine } from "./api";

/** Add a llama.cpp build (any repo + branch/tag/commit) as an engine; it then becomes an experiment variable. */
export default function EnginesPanel({ onChange }: { onChange: () => void }) {
  const [list, setList] = useState<Engine[]>([]);
  const [repo, setRepo] = useState("https://github.com/ggml-org/llama.cpp");
  const [ref, setRef] = useState("master");
  const [name, setName] = useState("");
  const [backend, setBackend] = useState("auto");
  const [log, setLog] = useState<{ id: string; text: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const building = list.some((e) => e.status === "building" || e.status === "queued");

  useEffect(() => {
    let lastReady = "";
    const tick = () => api.engines().then((l) => {
      setList(l);
      const ready = l.filter((e) => e.status === "ready").map((e) => e.id).join(",");
      if (lastReady && ready !== lastReady) onChange();
      lastReady = ready;
    }).catch(() => {});
    tick();
    const t = setInterval(tick, building ? 3000 : 15000);
    return () => clearInterval(t);
  }, [building]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!log) return;
    const t = setInterval(() => api.engineLog(log.id).then((d) => setLog((l) => (l ? { ...l, text: d.log } : l))).catch(() => {}), 3000);
    return () => clearInterval(t);
  }, [log?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const add = async () => {
    setErr(null);
    try { await api.engineAdd(repo, ref, name, backend); setName(""); setList(await api.engines()); } catch (e) { setErr((e as Error).message); }
  };

  return (
    <div className="ws">
      <div className="row-sub">Build any version of llama.cpp (a branch, tag, commit, or your own fork with a new kernel or quantization type), then profile it against <b>stock</b> with identical measurements. Pick the compute backend: CUDA (NVIDIA, compiled for this GPU), Vulkan (AMD, Intel and most other GPUs, including integrated ones) or CPU only. CUDA builds take 10–20 minutes. <b>Building runs that repository's code on this machine</b>, so only add repositories you trust.</div>
      <div className="ws-grid">
        <label><span>Git repository (https)</span><input className="field" value={repo} onChange={(e) => setRepo(e.target.value)} /></label>
        <label><span>Branch, tag or commit</span><input className="field" value={ref} onChange={(e) => setRef(e.target.value)} placeholder="master, b6000, a1b2c3d…" /></label>
        <label><span>Name <span className="muted">(optional)</span></span><input className="field" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. my-kernel" /></label>
        <label><span>Backend</span><select className="field select" value={backend} onChange={(e) => setBackend(e.target.value)}>
          <option value="auto">Auto (CUDA if installed, else CPU)</option><option value="cuda">CUDA (NVIDIA)</option><option value="vulkan">Vulkan (any GPU)</option><option value="cpu">CPU only</option>
        </select></label>
      </div>
      <div className="xc-btns"><button className="btn" disabled={building} onClick={add}><Cpu /> {building ? "A build is running…" : "Build engine"}</button></div>
      {err && <div className="error"><AlertCircle /> {err}</div>}
      <div className="ws-jobs">
        {list.map((e) => (
          <div key={e.id} className={`ws-job ${e.status === "ready" ? "done" : e.status}`}>
            {e.status === "building" || e.status === "queued" ? <Loader2 className="spin" size={15} /> : e.status === "ready" ? <Check size={15} /> : <AlertCircle size={15} />}
            <div><b>{e.name}</b>
              <span>{e.repo.replace("https://", "")} @ {e.ref}{e.commit_sha ? ` (${e.commit_sha.slice(0, 9)})` : ""} · {e.status === "building" ? `${e.step ?? "building"}…` : e.status}{e.error ? `: ${e.error}` : ""}</span>
              {log?.id === e.id && <pre className="xcmds">{log.text.split("\n").slice(-30).join("\n") || "…"}</pre>}
            </div>
            {e.id !== "stock" && <button className="btn small ghost" onClick={() => (log?.id === e.id ? setLog(null) : api.engineLog(e.id).then((d) => setLog({ id: e.id, text: d.log })))}>{log?.id === e.id ? "Hide log" : "Log"}</button>}
            {(e.status === "building" || e.status === "queued") && <button className="btn small ghost" onClick={() => api.engineCancel(e.id)}><Square /> Stop</button>}
            {e.id !== "stock" && e.status !== "building" && e.status !== "queued" &&
              <button className="icon-btn danger" title="Remove engine" onClick={() => { if (confirm(`Remove engine ${e.name}?`)) api.engineDelete(e.id).then(() => api.engines().then(setList)).then(onChange); }}><Trash2 /></button>}
          </div>
        ))}
      </div>
    </div>
  );
}
