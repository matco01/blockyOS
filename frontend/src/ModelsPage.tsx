import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, Check, Cpu, Download as DownloadIcon, HardDrive, Loader2, MemoryStick, RotateCw, Search, Trash2, X } from "lucide-react";
import { api, CatalogItem, Download, Fit, Hardware, HfFile, ModelFile, Runtime, Status } from "./api";
import { BLOCK_COLORS } from "./settings";
import Anatomy from "./Anatomy";
import QuantLab from "./QuantLab";
import { Pro, Tabs, Term, useMode } from "./ui";

const GB = 2 ** 30;
const fmtGB = (bytes: number) => `${(bytes / GB).toFixed(bytes >= 10 * GB ? 0 : 1)} GB`;
const fmtCtx = (n?: number | null) => (n ? (n >= 1024 ? `${Math.round(n / 1024)}k` : `${n}`) : "–");
const KV_BYTES: Record<string, number> = { f16: 2, q8_0: 1.0625, q4_0: 0.5625 };
const CTX_PRESETS = [4096, 8192, 16384, 32768, 65536, 131072, 262144];

export function FitBadge({ fit }: { fit: Fit }) {
  return <span className={`fit fit-${fit.verdict}`} title={`≈ ${(fit.need_mb / 1024).toFixed(1)} GB needed (incl. ${(fit.kv_mb / 1024).toFixed(1)} GB context, ${fit.measured ? "measured" : "estimated"})`}>{fit.label}</span>;
}

function HardwareCard({ hw }: { hw: Hardware }) {
  const gpu = hw.gpus[0];
  return (
    <div className="hw-grid">
      <div className="hw-item" style={{ "--c": "#7c5cff" } as React.CSSProperties}>
        <span className="hw-icon"><MemoryStick /></span>
        <div><div className="hw-label">GPU</div><div className="hw-value">{gpu ? gpu.name.replace("NVIDIA GeForce ", "") : "None"}</div>
          <div className="hw-sub">{gpu ? `${(gpu.vram_mb / 1024).toFixed(0)} GB VRAM · driver ${gpu.driver}${gpu.pcie ? ` · ${gpu.pcie}` : ""}` : "CPU inference only"}</div></div>
      </div>
      <div className="hw-item" style={{ "--c": "#22d3ee" } as React.CSSProperties}>
        <span className="hw-icon"><Cpu /></span>
        <div><div className="hw-label">CPU</div><div className="hw-value">{hw.cpu.replace(/ with .*$/, "")}</div>
          <div className="hw-sub">{hw.cpu_cores} cores · {hw.cpu_threads} threads</div></div>
      </div>
      <div className="hw-item" style={{ "--c": "#34d399" } as React.CSSProperties}>
        <span className="hw-icon"><HardDrive /></span>
        <div><div className="hw-label">Memory</div><div className="hw-value">{(hw.ram_mb / 1024).toFixed(1)} GB RAM</div>
          <div className="hw-sub">{hw.os}{hw.wsl ? " · WSL2" : ""} · llama.cpp {hw.llama_cpp ?? "?"}</div></div>
      </div>
    </div>
  );
}

export default function ModelsPage({ status }: { status: Status | null }) {
  const [data, setData] = useState<{ models: ModelFile[]; runtime: Runtime; catalog: CatalogItem[]; hardware: Hardware; kv_types: string[] } | null>(null);
  const [draft, setDraft] = useState<Runtime | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloads, setDownloads] = useState<Download[]>([]);
  const [repo, setRepo] = useState("");
  const [hf, setHf] = useState<HfFile[] | null>(null);
  const [hfState, setHfState] = useState<"idle" | "loading" | string>("idle");
  const switching = status?.busy === "switching";
  const mode = useMode();
  const [tab, setTab] = useState<"mine" | "get" | "inside" | "compress">(() => (new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("tab") as "mine" | "get" | "inside" | "compress") || "mine");

  const load = useCallback(() => {
    api.models().then((d) => { setData(d); setDraft((cur) => cur ?? d.runtime); }).catch((e) => setError(e.message));
  }, []);

  useEffect(load, [load]);

  // Reload when a switch finishes, so "active" and fit badges update.
  useEffect(() => {
    if (!switching) { load(); setDraft(null); }
  }, [switching, load]);

  // Download progress
  useEffect(() => {
    let alive = true;
    let prevActive = 0;
    const poll = async () => {
      try {
        const d = await api.downloads();
        if (!alive) return;
        setDownloads(d);
        const active = d.filter((x) => x.status === "downloading").length;
        if (active < prevActive) load(); // something finished
        prevActive = active;
      } catch { /* ignore */ }
    };
    poll();
    const t = setInterval(poll, 1000);
    return () => { alive = false; clearInterval(t); };
  }, [load]);

  const rt = draft ?? data?.runtime ?? null;
  const vramMb = data?.hardware.gpus[0]?.vram_mb ?? 0;

  // Live memory estimate for the settings being edited.
  const estimate = useMemo(() => {
    if (!data || !rt) return null;
    const m = data.models.find((x) => x.file === rt.model);
    if (!m) return null;
    // Prefer the measured per-token cost (running model), else scale the estimate.
    const fixed = m.fixed_state_mb ?? 0;
    const perTok = m.kv_mb_per_token
      ? m.kv_mb_per_token / (KV_BYTES[data.runtime.kv] ?? 2)
      : m.fit.kv_mb / data.runtime.ctx / (KV_BYTES[data.runtime.kv] ?? 2);
    const kv = perTok * rt.ctx * (KV_BYTES[rt.kv] ?? 2) + fixed;
    const need = m.size_bytes / 2 ** 20 + kv + 700;
    return { need, kv, fitsGpu: vramMb > 0 && need <= vramMb - 600, model: m };
  }, [data, rt, vramMb]);

  const dirty = !!(data && draft && JSON.stringify(draft) !== JSON.stringify(data.runtime));

  const apply = async (cfg: Runtime) => {
    setError(null);
    try { await api.setRuntime(cfg); } catch (e) { setError((e as Error).message); }
  };

  const searchHf = async (r = repo) => {
    if (!r.trim()) return;
    setHfState("loading"); setHf(null);
    try { setHf(await api.hfFiles(r.trim())); setHfState("idle"); } catch (e) { setHfState((e as Error).message); }
  };

  const download = async (repoName: string, path: string) => {
    setError(null);
    try { await api.download(repoName, path); setDownloads(await api.downloads()); } catch (e) { setError((e as Error).message); }
  };

  const remove = async (m: ModelFile) => {
    if (!confirm(`Delete ${m.file} (${fmtGB(m.size_bytes)})? You can download it again later.`)) return;
    try { await api.deleteModel(m.file); load(); } catch (e) { setError((e as Error).message); }
  };

  const activeDownloads = downloads.filter((d) => d.status === "downloading" || d.status === "error");

  return (
    <div className="page">
      <div className="page-inner wide">
        <h1 className="page-title">Models</h1>
        <p className="page-sub">The AI models on this computer, how they run, and which others would fit.</p>
        <Tabs value={tab} onChange={setTab} tabs={[
          { id: "mine", label: "My models", badge: data?.models.length },
          { id: "get", label: "Get more", badge: activeDownloads.length || null },
          { id: "inside", label: "Inside the model" },
          { id: "compress", label: "Compression lab" },
        ]} />
        {(error || status?.switch_error) && (
          <div className="error" style={{ marginTop: 16 }}><AlertCircle /> {error ?? status?.switch_error}</div>
        )}

        {tab === "mine" && (<>
        {/* Local models */}
        <div className="group-label">On this computer</div>
        <div className="group">
          {data?.models.length === 0 && <div className="row"><div className="row-main muted">No models yet. Open "Get more" to download one.</div></div>}
          {data?.models.map((m, i) => (
            <div className="row model-row" key={m.file}>
              <span className="model-tile" style={{ "--c": BLOCK_COLORS[i % BLOCK_COLORS.length] } as React.CSSProperties}>{m.size_label ?? "?"}</span>
              <div className="row-main">
                <div className="model-name">{m.name ?? m.file} {m.experts ? <span className="badge"><Term k="moe">MoE</Term> {m.experts_used}/{m.experts}</span> : null}</div>
                {mode === "pro" ? (
                  <div className="row-sub mono">
                    {[m.architecture, m.quant, fmtGB(m.size_bytes), m.layers && `${m.layers} layers`, m.context_length && `${fmtCtx(m.context_length)} trained ctx`, m.license].filter(Boolean).join(" · ")}
                  </div>
                ) : (
                  <div className="row-sub">
                    {m.size_label ? `${m.size_label.replace("B", " billion")} parameters` : "Model"} · {fmtGB(m.size_bytes)} on disk · <Term k="quant">{m.quant ?? "?"}</Term> compression
                  </div>
                )}
              </div>
              <FitBadge fit={m.fit} />
              {m.active ? (
                <span className="badge you"><Check size={12} /> Running</span>
              ) : (
                <button className="btn small" disabled={switching || !!status?.busy} onClick={() => data && apply({ ...data.runtime, model: m.file })}>Load</button>
              )}
              <button className="icon-btn danger" title="Delete" disabled={m.active} onClick={() => remove(m)}><Trash2 /></button>
            </div>
          ))}
        </div>

        {/* Runtime */}
        <div className="group-label">How the running model is set up</div>
        {data && rt && (
          <div className="group">
            <div className="row">
              <div className="row-main"><div>Model</div><div className="row-sub">The model the chat uses.</div></div>
              <select className="field select" value={rt.model ?? ""} disabled={switching}
                onChange={(e) => setDraft({ ...rt, model: e.target.value })}>
                {data.models.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file} · {m.quant}</option>)}
              </select>
            </div>
            <div className="row">
              <div className="row-main"><div><Term k="context">Context window</Term></div>
                <div className="row-sub">How much text the model can see at once. Trained for {fmtCtx(estimate?.model.context_length)}. Bigger costs memory.</div></div>
              <select className="field select" value={rt.ctx} disabled={switching} onChange={(e) => setDraft({ ...rt, ctx: Number(e.target.value) })}>
                {CTX_PRESETS.filter((c) => !estimate?.model.context_length || c <= estimate.model.context_length).map((c) =>
                  <option key={c} value={c}>{fmtCtx(c)} tokens</option>)}
              </select>
            </div>
            <Pro>
              <div className="row">
                <div className="row-main"><div><Term k="kv">KV cache</Term> precision</div><div className="row-sub">q8_0 halves context memory with almost no quality loss. q4_0 saves more, costs some accuracy.</div></div>
                <div className="segmented">
                  {data.kv_types.map((k) => <button key={k} disabled={switching} className={rt.kv === k ? "on" : ""} onClick={() => setDraft({ ...rt, kv: k })}>{k}</button>)}
                </div>
              </div>
              <div className="row">
                <div className="row-main"><div><Term k="layers">GPU layers</Term></div><div className="row-sub">Auto lets llama.cpp fit as much as possible on the GPU (MoE-aware). Set a number to force a split.</div></div>
                <input className="field num" placeholder="auto" disabled={switching} value={rt.ngl === "auto" ? "" : rt.ngl}
                  onChange={(e) => setDraft({ ...rt, ngl: e.target.value.replace(/\D/g, "") || "auto" })} />
              </div>
              <div className="row">
                <div className="row-main"><div>CPU threads</div><div className="row-sub">Matters when part of the model runs on the CPU. Auto = physical cores.</div></div>
                <input className="field num" placeholder="auto" disabled={switching} value={rt.threads === "auto" ? "" : rt.threads}
                  onChange={(e) => setDraft({ ...rt, threads: e.target.value.replace(/\D/g, "") || "auto" })} />
              </div>
            </Pro>
            <div className="row apply-row">
              <div className="row-main">
                {estimate && (
                  <>
                    <div className="mem-line">
                      ≈ <b>{(estimate.need / 1024).toFixed(1)} GB</b> needed
                      <span className="muted"> · weights {fmtGB(estimate.model.size_bytes)} + context {(estimate.kv / 1024).toFixed(1)} GB + overhead</span>
                    </div>
                    {vramMb > 0 && (
                      <div className="mem-bar" title={`${(vramMb / 1024).toFixed(0)} GB VRAM`}>
                        <div style={{ width: `${Math.min(100, (estimate.need / vramMb) * 100)}%` }} className={estimate.fitsGpu ? "ok" : "over"} />
                      </div>
                    )}
                    <div className="row-sub">{estimate.fitsGpu ? "Fits entirely in GPU memory." : "Won't fit in VRAM: part runs from system RAM (slower)."}</div>
                  </>
                )}
              </div>
              {switching ? (
                <span className="btn ghost"><Loader2 className="spin" /> Restarting model…</span>
              ) : (
                <>
                  {dirty && <button className="btn ghost" onClick={() => setDraft(data.runtime)}>Reset</button>}
                  <button className="btn" disabled={!dirty} onClick={() => draft && apply(draft)}><RotateCw /> Apply & restart</button>
                </>
              )}
            </div>
          </div>
        )}

        <div className="group-label">This computer</div>
        {data && <HardwareCard hw={data.hardware} />}
        </>)}

        {tab === "inside" && data && (<>
        <div className="group-label">The model's real structure</div>
        <div className="group pad"><Anatomy models={data.models} active={data.runtime.model} /></div>
        </>)}

        {tab === "compress" && data && <QuantLab models={data.models} status={status} />}

        {tab === "get" && (<>

        {/* Downloads */}
        {activeDownloads.length > 0 && (
          <>
            <div className="group-label">Downloading</div>
            <div className="group">
              {activeDownloads.map((d) => (
                <div className="row" key={d.id}>
                  <div className="row-main">
                    <div className="mono small">{d.file}</div>
                    {d.status === "error" ? <div className="row-sub" style={{ color: "var(--danger)" }}>{d.error}</div> : (
                      <>
                        <div className="mem-bar"><div className="ok" style={{ width: `${d.total ? (d.done / d.total) * 100 : 0}%` }} /></div>
                        <div className="row-sub mono">{fmtGB(d.done)} / {d.total ? fmtGB(d.total) : "?"} · {(d.speed / 2 ** 20).toFixed(1)} MB/s
                          {d.speed > 0 && d.total ? ` · ${Math.ceil((d.total - d.done) / d.speed / 60)} min left` : ""}</div>
                      </>
                    )}
                  </div>
                  {d.status === "downloading" && <button className="icon-btn" title="Cancel" onClick={() => api.cancelDownload(d.id)}><X /></button>}
                </div>
              ))}
            </div>
          </>
        )}

        {/* Catalog */}
        <div className="group-label">Recommended</div>
        <div className="catalog">
          {data?.catalog.map((c, i) => {
            const dl = downloads.find((d) => d.file === c.file && d.status === "downloading");
            return (
              <div className="catalog-card" key={c.file} style={{ "--c": BLOCK_COLORS[i % BLOCK_COLORS.length] } as React.CSSProperties}>
                <div className="catalog-top"><span className="catalog-dot" /><b>{c.title}</b><span className="muted small">{fmtGB(c.size)}</span></div>
                <div className="small muted catalog-note">{c.note}</div>
                <div className="catalog-bottom">
                  <FitBadge fit={c.fit} />
                  {c.downloaded ? <span className="badge"><Check size={12} /> Downloaded</span>
                    : dl ? <span className="badge">{Math.round((dl.done / (dl.total || 1)) * 100)}%</span>
                    : <button className="btn small" disabled={c.fit.verdict === "too_big"} onClick={() => download(c.repo, c.file)}><DownloadIcon /> Get</button>}
                </div>
              </div>
            );
          })}
        </div>

        {/* Hugging Face */}
        <div className="group-label">Any model from Hugging Face</div>
        <div className="group">
          <form className="row" onSubmit={(e) => { e.preventDefault(); searchHf(); }}>
            <input className="field" placeholder="owner/repo, e.g. unsloth/Qwen3.5-9B-GGUF" value={repo} onChange={(e) => setRepo(e.target.value)} />
            <button className="btn" disabled={!repo.trim() || hfState === "loading"}>{hfState === "loading" ? <Loader2 className="spin" /> : <Search />} Browse</button>
          </form>
          {hfState !== "idle" && hfState !== "loading" && <div className="row"><div className="row-main" style={{ color: "var(--danger)" }}>{hfState}</div></div>}
          {hf?.length === 0 && <div className="row"><div className="row-main muted">No GGUF files in this repository.</div></div>}
          {hf?.map((f) => {
            const name = f.path.split("/").pop()!;
            const have = data?.models.some((m) => m.file === name);
            return (
              <div className="row" key={f.path}>
                <div className="row-main"><div className="mono small">{f.path}</div><div className="row-sub">{fmtGB(f.size)}</div></div>
                <FitBadge fit={f.fit} />
                {have ? <span className="badge"><Check size={12} /> Have it</span>
                  : f.split ? <span className="badge" title="Multi-part files aren't supported yet">Split file</span>
                  : <button className="btn small" disabled={f.fit.verdict === "too_big"} onClick={() => download(repo.trim(), f.path)}><DownloadIcon /> Get</button>}
              </div>
            );
          })}
          <div className="row"><div className="row-main row-sub">Downloads come straight from huggingface.co. That's the only time Blocky uses the internet.</div></div>
        </div>
        </>)}
      </div>
    </div>
  );
}
