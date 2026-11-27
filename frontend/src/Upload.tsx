/* Bringing files in from your computer: a button, an area you can drop files on, and progress while they upload. */
import { useRef, useState } from "react";
import { FolderUp, Upload as UploadIcon } from "lucide-react";
import { api } from "./api";

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
