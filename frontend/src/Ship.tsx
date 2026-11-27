/* Ship: publish a measured version to Hugging Face with a model card written from its measurements. */
import { useEffect, useState } from "react";
import { Check, ExternalLink, Lock, Rocket } from "lucide-react";
import { api, PublishJob, PublishLicense } from "./api";
import { fmtBytes } from "./Upload";

const PERMISSIVE = new Set(["apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0", "unlicense"]);
const slug = (s: string) => s.toLowerCase().replace(/\.gguf$/, "").replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80);

export function PublishPanel({ experimentId, runId, projectName, onClose }: { experimentId: number; runId: number; projectName: string; onClose: () => void }) {
  const [account, setAccount] = useState<{ connected: boolean; user: string | null } | null>(null);
  const [lic, setLic] = useState<PublishLicense | null>(null);
  const [token, setToken] = useState(""), [repo, setRepo] = useState(""), [priv, setPriv] = useState(true);
  const [license, setLicense] = useState(""), [confirmed, setConfirmed] = useState(false);
  const [baseModel, setBaseModel] = useState(""), [bundle, setBundle] = useState(true);
  const [card, setCard] = useState(""), [job, setJob] = useState<PublishJob | null>(null), [err, setErr] = useState("");

  useEffect(() => {
    api.publishAccount().then(setAccount).catch((e) => setErr((e as Error).message));
    api.publishLicense(experimentId, runId).then((l) => { setLic(l); setLicense(l.detected ?? ""); }).catch((e) => setErr((e as Error).message));
  }, [experimentId, runId]);
  useEffect(() => {
    if (account?.user && lic && !repo) setRepo(`${account.user}/${slug(`${projectName}-${lic.file}`)}`);
  }, [account, lic]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const t = setTimeout(() => fetch(api.publishCardUrl(experimentId, runId, license, baseModel)).then((r) => r.text()).then(setCard).catch(() => {}), 300);
    return () => clearTimeout(t);
  }, [experimentId, runId, license, baseModel]);
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const t = setInterval(() => api.publishJobs().then((all) => { const j = all.find((x) => x.id === job.id); if (j) setJob(j); }).catch(() => {}), 1500);
    return () => clearInterval(t);
  }, [job]);

  const needsConfirm = !!license && !PERMISSIVE.has(license);
  const publish = async () => {
    setErr("");
    try {
      const { id } = await api.publishHf({ experiment_id: experimentId, run_id: runId, repo, private: priv, license, confirm_license: confirmed, base_model: baseModel, title: "", include_bundle: bundle });
      setJob({ id, repo, status: "running", step: "starting", error: null, url: null, private: priv, file: lic?.file ?? "", bytes: lic?.bytes ?? 0, started: Date.now() / 1000 });
    } catch (e) { setErr((e as Error).message); }
  };

  return (
    <div className="group pad ship-panel">
      <div className="pj-line-head"><b><Rocket size={16} /> Publish to Hugging Face</b><button className="btn small ghost" onClick={onClose}>Close</button></div>
      {err && <p className="error">{err}</p>}

      <div className="ship-step">
        <span className="ship-num">1</span>
        <div>
          <b>Your Hugging Face account</b>
          {account?.connected
            ? <p className="row-sub"><Check size={14} /> Connected as <b>{account.user}</b>. <button className="link" onClick={() => api.publishDisconnect().then(setAccount)}>Disconnect</button></p>
            : <>
              <p className="row-sub">Create an access token with <b>write</b> access at <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">huggingface.co → Settings → Access Tokens <ExternalLink size={12} /></a> and paste it here. It stays on this device and is never shown again.</p>
              <div className="xc-btns"><input className="field" type="password" value={token} placeholder="hf_…" onChange={(e) => setToken(e.target.value)} aria-label="Hugging Face token" />
                <button className="btn small" disabled={!token.trim()} onClick={() => api.publishConnect(token).then((a) => { setAccount(a); setToken(""); setErr(""); }).catch((e) => setErr((e as Error).message))}>Connect</button></div>
            </>}
        </div>
      </div>

      <div className="ship-step">
        <span className="ship-num">2</span>
        <div>
          <b>Where</b>
          <div className="ship-fields">
            <label>Repository<input className="field" value={repo} placeholder="your-name/model-name" onChange={(e) => setRepo(e.target.value)} /></label>
            <label className="flow-check"><input type="checkbox" checked={priv} onChange={(e) => setPriv(e.target.checked)} /> <Lock size={13} /> Private (only you; make it public later on Hugging Face)</label>
          </div>
        </div>
      </div>

      <div className="ship-step">
        <span className="ship-num">3</span>
        <div>
          <b>License</b>
          <p className="row-sub">{lic?.detected ? <>The model file says <code>{lic.detected}</code>.</> : <>The model file doesn't say which license it has: check the base model's page and choose it.</>} A model built from another model keeps that model's license.</p>
          <div className="ship-fields">
            <label>Shared under<select className="field select" value={license} onChange={(e) => { setLicense(e.target.value); setConfirmed(false); }}>
              <option value="">Choose…</option>
              {lic?.detected && !lic.options.some((o) => o.id === lic.detected) && <option value={lic.detected}>{lic.detected}</option>}
              {(lic?.options ?? []).map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
            </select></label>
            <label>Built on (Hugging Face id, optional)<input className="field" value={baseModel} placeholder="e.g. Qwen/Qwen3.5-0.8B" onChange={(e) => setBaseModel(e.target.value)} /></label>
          </div>
          {needsConfirm && <label className="flow-check ship-confirm"><input type="checkbox" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} /> This license has conditions. I've read them and I'm allowed to share this model under it.</label>}
        </div>
      </div>

      <div className="ship-step">
        <span className="ship-num">4</span>
        <div>
          <b>What gets uploaded</b>
          <ul className="ship-list">
            <li><code>README.md</code>: the model card below, written from the measurements</li>
            <li><code>{lic?.file}</code>{lic ? ` (${fmtBytes(lic.bytes)})` : ""}</li>
            <li><label className="flow-check"><input type="checkbox" checked={bundle} onChange={(e) => setBundle(e.target.checked)} /> <code>reproducibility-bundle.zip</code>: lets anyone re-run every measurement</label></li>
          </ul>
          <details><summary>Preview the model card</summary><pre className="rp-output ship-card">{card || "…"}</pre></details>
        </div>
      </div>

      {job ? (
        <div className={`ship-job ${job.status}`}>
          {job.status === "running" && <p role="status">Publishing to <b>{job.repo}</b>: {job.step}… Large files take a while; you can leave this page.</p>}
          {job.status === "done" && <p><Check size={14} /> Published. <a href={job.url ?? `https://huggingface.co/${job.repo}`} target="_blank" rel="noreferrer">Open {job.repo} <ExternalLink size={12} /></a></p>}
          {job.status === "error" && <p className="error">Publishing failed: {job.error}</p>}
        </div>
      ) : (
        <button className="btn" disabled={!account?.connected || !repo.includes("/") || !license || (needsConfirm && !confirmed)} onClick={publish}><Rocket /> Publish{priv ? " privately" : ""}</button>
      )}
    </div>
  );
}
