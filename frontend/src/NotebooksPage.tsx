import { useEffect, useState } from "react";
import { BookOpen, Download, ExternalLink, FileCode2 } from "lucide-react";
import { api } from "./api";

const SNIPPET = `import blocky
b = blocky.Client()                            # this box
exp = b.experiment("quantization study")
df = exp.results()                             # pandas: one row per run, every metric
exp.branch(exp.run("Q4_K_M"), kv="q4_0")       # queue: that run's setup + one change
exp.sweep({"ctx": [4096, 16384, 32768]})       # queue a grid
exp.wait()

b.variant("Qwen3.5-9B-Q8_0.gguf", "Q3_K_M", imatrix=True).wait()
b.scan("Qwen3.5-0.8B-Q8_0.gguf", mode="prune").wait()
b.build_engine("https://github.com/you/llama.cpp", "my-branch").wait()`;

export default function NotebooksPage() {
  const [s, setS] = useState<{ installed: boolean; running: boolean; url: string | null; folder: string } | null>(null);
  useEffect(() => { api.notebooks().then(setS).catch(() => {}); }, []);
  return (
    <div className="page">
      <div className="page-inner">
        <h1 className="page-title">Notebooks</h1>
        <p className="page-sub">Script everything the lab does from Python: queue runs and sweeps, build weights and engines, pull every result into pandas. JupyterLab runs on this box, local-only.</p>

        <div className="group pad nb-open">
          <div><b><BookOpen size={16} /> JupyterLab</b>
            <span className="row-sub">{!s ? "Checking…" : !s.installed ? "Not installed. On the box run: sudo bash scripts/setup-notebooks.sh"
              : s.running ? `Running · notebooks are saved in ${s.folder}` : "Installed but not running: sudo systemctl start blocky-notebook"}</span></div>
          {s?.running && s.url && <a className="btn" href={s.url} target="_blank" rel="noreferrer"><ExternalLink /> Open JupyterLab</a>}
        </div>

        <div className="group-label">The Python client</div>
        <div className="group pad">
          <div className="row-sub">Already importable in every notebook here. To use it from your own machine, download one file; it needs only the Python standard library (pandas optional).</div>
          <pre className="xcmds nb-code">{SNIPPET}</pre>
          <div className="xc-btns">
            <a className="btn small ghost" href="/api/client.py" download><Download /> blocky.py</a>
            <a className="btn small ghost" href="/docs" target="_blank" rel="noreferrer"><FileCode2 /> REST API reference</a>
          </div>
        </div>
      </div>
    </div>
  );
}
