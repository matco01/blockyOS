/* Bringing files in from your computer: a button, an area you can drop files on, and progress while they upload. */
import { useEffect, useRef, useState } from "react";
import { FolderUp, Upload as UploadIcon } from "lucide-react";
import { api, EnvStatus } from "./api";

/** A button that opens the file picker. `folder` picks a whole folder (its files keep their relative paths). */
export function FilePick({ label, accept, multiple, folder, disabled, ghost, onFiles }: {
  label: string; accept?: string; multiple?: boolean; folder?: boolean; disabled?: boolean; ghost?: boolean; onFiles: (files: File[]) => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  const dir = folder ? { webkitdirectory: "", directory: "" } as Record<string, string> : {};
  return (
    <>
      <button type="button" className={`btn small ${ghost ? "ghost" : ""}`} disabled={disabled} onClick={() => ref.current?.click()}>
        {folder ? <FolderUp /> : <UploadIcon />} {label}
      </button>
      <input ref={ref} type="file" hidden accept={accept} multiple={multiple || folder} {...dir}
        onChange={(e) => { const f = Array.from(e.target.files ?? []); e.target.value = ""; if (f.length) onFiles(f); }} />
    </>
  );
}

/** An area that accepts dropped files. */
export function DropZone({ accept, onFiles, children }: { accept: string[]; onFiles: (files: File[]) => void; children: React.ReactNode }) {
  const [over, setOver] = useState(false);
  return (
    <div className={`drop-zone ${over ? "over" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault(); setOver(false);
        const files = Array.from(e.dataTransfer.files).filter((f) => accept.some((a) => f.name.toLowerCase().endsWith(a)));
        if (files.length) onFiles(files);
      }}>
      {children}
    </div>
  );
}

export function ProgressBar({ value, label }: { value: number; label: string }) {
  return (
    <div className="upload-progress" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
      <div className="upload-progress-bar" style={{ width: `${Math.max(2, value * 100)}%` }} />
      <span>{label} · {Math.round(value * 100)}%</span>
    </div>
  );
}

const SKIP = /(^|\/)(\.git|__pycache__|\.venv|venv|node_modules|\.ipynb_checkpoints)(\/|$)/;

/** Uploads your code into a code workspace: a chosen folder (file by file) or one .zip. Returns the files that could start it. */
export async function uploadCode(rid: number, files: File[], onProgress: (fraction: number, label: string) => void): Promise<string[]> {
  if (files.length === 1 && files[0].name.toLowerCase().endsWith(".zip")) {
    const r = await api.researchImportZip(rid, files[0], (f) => onProgress(f, `Uploading ${files[0].name}`));
    return r.entry_candidates;
  }
  // A folder: strip its own name, skip environments and caches, keep the structure.
  const items = files.map((f) => {
    const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
    const parts = rel.split("/");
    return { file: f, path: parts.length > 1 ? parts.slice(1).join("/") : rel };
  }).filter((x) => x.path && !SKIP.test(x.path));
  if (items.length === 0) throw new Error("That folder has no files to upload");
  if (items.length > 5000) throw new Error("That folder has more than 5,000 files. Upload a .zip instead, or leave out data and environments.");
  const total = items.reduce((n, x) => n + x.file.size, 0) || 1;
  let done = 0;
  for (const x of items) {
    onProgress(done / total, `Uploading ${x.path}`);
    await api.researchUpload(rid, x.path, x.file);
    done += x.file.size;
  }
  onProgress(1, `Uploaded ${items.length} file${items.length === 1 ? "" : "s"}`);
  return (await api.researchEntries(rid)).entries;
}

export const fmtBytes = (b: number) => b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GB` : b >= 2 ** 20 ? `${(b / 2 ** 20).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`;

/** Paste an https link to a .gguf file; downloads on the device itself (no browser upload), with progress. */
export function LinkDownload({ onDone }: { onDone: (file: string) => void }) {
  const [url, setUrl] = useState(""), [err, setErr] = useState(""), [prog, setProg] = useState<{ value: number; label: string } | null>(null);
  const start = async () => {
    setErr("");
    try {
      const { id } = await api.urlDownload(url);
      setProg({ value: 0, label: "Starting the download" });
      for (;;) {
        await new Promise((r) => setTimeout(r, 1000));
        const d = (await api.downloads()).find((x) => x.id === id);
        if (!d) throw new Error("The download disappeared (was Blocky restarted?)");
        if (d.status === "done") { setProg(null); setUrl(""); onDone(d.file); return; }
        if (d.status !== "downloading") throw new Error(d.error || `Download ${d.status}`);
        setProg({ value: d.total ? d.done / d.total : 0, label: `Downloading ${d.file} · ${fmtBytes(d.done)}${d.total ? ` of ${fmtBytes(d.total)}` : ""}` });
      }
    } catch (e) { setErr((e as Error).message); setProg(null); }
  };
  return (
    <div className="link-download">
      <div className="xc-btns">
        <input className="field" value={url} placeholder="https://…/model.gguf" onChange={(e) => setUrl(e.target.value)} aria-label="Link to a .gguf file" />
        <button className="btn small" disabled={!url.trim() || !!prog} onClick={start}>Download</button>
      </div>
      {prog && <ProgressBar value={prog.value} label={prog.label} />}
      {err && <p className="error">{err}</p>}
    </div>
  );
}

/** A model folder in Hugging Face format (config.json + .safetensors) from your computer, for fine-tuning and conversion. */
export function CheckpointUpload({ onDone }: { onDone: (repo: string) => void }) {
  const [files, setFiles] = useState<File[]>([]), [name, setName] = useState("");
  const [err, setErr] = useState(""), [prog, setProg] = useState<{ value: number; label: string } | null>(null);
  const pick = (f: File[]) => {
    setFiles(f); setErr("");
    const top = ((f[0] as File & { webkitRelativePath?: string }).webkitRelativePath || "").split("/")[0];
    setName((n) => n || top || "my-model");
  };
  const send = async () => {
    setErr("");
    try {
      const total = files.reduce((n, f) => n + f.size, 0) || 1;
      let done = 0;
      for (const f of files) {
        const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
        if (/(^|\/)\./.test(rel)) continue; // hidden files (.git, .cache)
        setProg({ value: done / total, label: `Uploading ${rel}` });
        await api.checkpointUpload(name, rel, f);
        done += f.size;
      }
      setProg({ value: 1, label: "Checking the model" });
      const r = await api.checkpointFinish(name);
      setProg(null); setFiles([]); setName(""); onDone(r.repo);
    } catch (e) { setErr((e as Error).message); setProg(null); }
  };
  return (
    <div className="ckpt-upload">
      <div className="xc-btns">
        <FilePick label={files.length ? `${files.length} files chosen` : "Choose the model folder"} folder ghost={files.length > 0} disabled={!!prog} onFiles={pick} />
        {files.length > 0 && <>
          <input className="field" value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" aria-label="Model name" />
          <button className="btn small" disabled={!name.trim() || !!prog} onClick={send}>Upload</button>
        </>}
      </div>
      <p className="row-sub">The folder must contain <code>config.json</code> and the weights (<code>.safetensors</code>), plus the tokenizer files.</p>
      {prog && <ProgressBar value={prog.value} label={prog.label} />}
      {err && <p className="error">{err}</p>}
    </div>
  );
}

/** The packages a code workspace needs, installed into its own environment with one button. */
export function EnvPanel({ rid, python, setPython }: { rid: number; python: string; setPython: (p: string) => void }) {
  const [s, setS] = useState<EnvStatus | null>(null), [err, setErr] = useState("");
  const [reuse, setReuse] = useState(true);
  useEffect(() => {
    let active = true;
    const poll = () => api.researchEnv(rid).then((x) => { if (active) setS(x); }).catch((e) => active && setErr((e as Error).message));
    poll();
    const t = window.setInterval(poll, 2000);
    return () => { active = false; window.clearInterval(t); };
  }, [rid]);
  // Once the workspace has its environment, runs use it.
  useEffect(() => { if (s?.status === "ready" && s.python && python !== s.python) setPython(s.python); }, [s?.status, s?.python]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!s) return null;
  const req = s.requirements;
  const install = () => { setErr(""); api.researchEnvInstall(rid, reuse).then(setS).catch((e) => setErr((e as Error).message)); };
  return (
    <div className="group pad env-panel">
      <div className="env-head">
        <b>Packages</b>
        <span className={`env-state ${s.status}`}>
          {s.status === "installing" ? "Installing…" : s.status === "ready" ? (s.up_to_date ? "Installed" : "Out of date") : s.status === "error" ? "Install failed" : "Not installed"}
        </span>
      </div>
      <p className="row-sub">
        {req.file === "requirements.txt" ? <>Your code lists {req.packages.length} package{req.packages.length === 1 ? "" : "s"} in <code>requirements.txt</code>{req.packages.length > 0 && <>: {req.packages.slice(0, 6).join(", ")}{req.packages.length > 6 ? ", …" : ""}</>}.</>
          : req.file ? <>Your code is a Python package (<code>{req.file}</code>); installing it also installs what it depends on.</>
            : <>No <code>requirements.txt</code> found. If your code imports packages, add one listing them (one per line).</>}
        {" "}They are installed into this code's own environment, so other projects are not affected.
      </p>
      {s.ml_available && <label className="flow-check"><input type="checkbox" checked={reuse} onChange={(e) => setReuse(e.target.checked)} /> Reuse the PyTorch and Transformers already on this device (saves several GB)</label>}
      <div className="flow-toolbar">
        <button className="btn small" disabled={s.status === "installing"} onClick={install}>{s.status === "ready" ? (s.up_to_date ? "Reinstall packages" : "Update packages") : "Install packages"}</button>
        {s.python && s.status !== "installing" && <button className="btn small ghost" onClick={() => { if (confirm("Delete this code's environment? The code itself is kept.")) api.researchEnvDelete(rid).then(() => api.researchEnv(rid)).then(setS); }}>Delete environment</button>}
      </div>
      {s.status === "installing" && <p className="row-sub">This downloads packages from the internet and can take several minutes. You can leave this page.</p>}
      {(s.error || err) && <p className="error">{s.error || err}</p>}
      {s.log && <details open={s.status === "error" || undefined}><summary>Install log</summary><pre className="rp-output">{s.log}</pre></details>}
    </div>
  );
}
