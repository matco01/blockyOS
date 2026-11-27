/* GPU safety: the banner shown everywhere while GPU work is paused, and its section in Settings. */
import { useEffect, useState } from "react";
import { Thermometer, TriangleAlert } from "lucide-react";
import { api, GpuSafety, Status } from "./api";

const WHY: Record<string, string> = {
  heat: "The GPU got too hot",
  lost: "The GPU stopped responding",
  restart: "Blocky restarted during GPU work",
  manual: "GPU work is paused",
};

/** Shown above every page while GPU work is paused. Nothing GPU-heavy starts until the owner resumes. */
export function GpuPauseBanner({ status, onResumed }: { status: Status | null; onResumed: () => void }) {
  const [busy, setBusy] = useState(false);
  const p = status?.gpu_pause;
  if (!p) return null;
  return (
    <div className="gpu-banner" role="alert">
      <TriangleAlert />
      <div><b>{WHY[p.kind] ?? WHY.manual}. GPU work is paused.</b><span>{p.reason}.</span></div>
      <button className="btn small" disabled={busy} onClick={() => { setBusy(true); api.gpuResume().then(onResumed).finally(() => setBusy(false)); }}>Resume GPU work</button>
    </div>
  );
}

/** Settings → GPU safety. */
export function GpuSafetySettings() {
  const [g, setG] = useState<GpuSafety | null>(null);
  const [limit, setLimit] = useState(""), [resume, setResume] = useState(""), [err, setErr] = useState("");
  const load = () => api.gpuSafety().then((x) => { setG(x); setErr(""); }).catch((e) => setErr((e as Error).message));
  useEffect(() => {
    api.settings().then((s) => { setLimit(s.gpu_max_temp_c ? String(s.gpu_max_temp_c) : ""); setResume(s.gpu_resume_temp_c ? String(s.gpu_resume_temp_c) : ""); }).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);
  if (!g) return null;
  const save = (body: Record<string, unknown>) => api.updateSettings(body).then(load).catch((e) => setErr((e as Error).message));
  return (
    <>
      <div className="group-label">GPU safety</div>
      <div className="group pad gpu-safety">
        <div className="gpu-safety-head">
          <span className="row-icon" style={{ background: "linear-gradient(180deg,#ff8a5c,#ff5c5c)" }}><Thermometer /></span>
          <div>
            <b>Protect the computer during long GPU work</b>
            <span className="row-sub">If the GPU stays above the limit, or stops responding, Blocky stops its GPU work and pauses until you resume. Queued runs wait until the GPU has cooled down.</span>
          </div>
          <input type="checkbox" className="switch" aria-label="GPU safety on" checked={g.enabled} onChange={(e) => save({ gpu_guard: e.target.checked })} />
        </div>
        {!g.gpu_seen && g.temp_c == null ? <p className="row-sub">No NVIDIA GPU detected yet: nothing to watch.</p> : <>
          <div className="gpu-safety-grid">
            <div><span className="row-sub">Now</span><b>{g.temp_c != null ? `${g.temp_c} °C` : "–"}</b></div>
            <label><span className="row-sub">Stop above (°C)</span>
              <input className="field" type="number" min={40} max={100} value={limit} placeholder={`${g.auto.limit_c} (this GPU)`}
                onChange={(e) => setLimit(e.target.value)} onBlur={() => save({ gpu_max_temp_c: limit === "" ? null : Number(limit) })} /></label>
            <label><span className="row-sub">Start again below (°C)</span>
              <input className="field" type="number" min={40} max={100} value={resume} placeholder={`${g.auto.resume_c} (default)`}
                onChange={(e) => setResume(e.target.value)} onBlur={() => save({ gpu_resume_temp_c: resume === "" ? null : Number(resume) })} /></label>
          </div>
          <p className="row-sub">
            In use: stop at {g.limit_c} °C, start again at {g.resume_c} °C.
            {g.auto.max_operating_c ? ` This GPU reports a maximum operating temperature of ${g.auto.max_operating_c} °C; the default stops 3 °C below it.` : ""}
            {" "}If your GPU has overheated before, set a lower limit.
          </p>
        </>}
        {g.waiting && <p className="pj-hint">Waiting for the GPU to cool: {g.waiting.temp_c} °C → {g.waiting.target_c} °C.</p>}
        {g.paused && <p className="error">{WHY[g.paused.kind] ?? WHY.manual}: {g.paused.reason}. <button className="link" onClick={() => api.gpuResume().then(load)}>Resume GPU work</button></p>}
        {err && <p className="error">{err}</p>}
        {g.events.length > 0 && <details><summary>Recent events</summary>
          <ul className="gpu-events">{g.events.map((e, i) => <li key={i}><span className="muted">{new Date(e.at * 1000).toLocaleString()}</span> {e.text}</li>)}</ul></details>}
      </div>
    </>
  );
}
