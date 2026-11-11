import { useState } from "react";
import { MemoryMapData } from "./api";
import { fmtMB, Pro, Term } from "./ui";

type Seg = { key: string; label: string; term?: string; mb: number; color: string };

function Bar({ segs, total, title }: { segs: Seg[]; total: number; title: string }) {
  const [hover, setHover] = useState<string | null>(null);
  const shown = segs.filter((s) => s.mb > 0);
  return (
    <div className="mm-bar-wrap">
      <div className="mm-bar" role="img" aria-label={`${title}: ${shown.map((s) => `${s.label} ${fmtMB(s.mb)}`).join(", ")}`}>
        {shown.map((s) => (
          <div key={s.key} className={`mm-seg ${s.key === "free" ? "free" : ""} ${hover && hover !== s.key ? "dim" : ""}`}
            style={{ flexGrow: s.mb, background: s.key === "free" ? undefined : s.color }}
            onMouseEnter={() => setHover(s.key)} onMouseLeave={() => setHover(null)}>
            {s.mb / total > 0.12 && <span className="mm-seg-label">{fmtMB(s.mb)}</span>}
            {hover === s.key && <div className="mm-tip"><b>{s.label}</b> {fmtMB(s.mb)} · {((s.mb / total) * 100).toFixed(0)}%</div>}
          </div>
        ))}
      </div>
      <div className="mm-legend">
        {shown.map((s) => (
          <span key={s.key} className={hover && hover !== s.key ? "dim" : ""} onMouseEnter={() => setHover(s.key)} onMouseLeave={() => setHover(null)}>
            <i className={s.key === "free" ? "free" : ""} style={{ background: s.key === "free" ? undefined : s.color }} />
            {s.term ? <Term k={s.term}>{s.label}</Term> : s.label}
            <b>{fmtMB(s.mb)}</b>
          </span>
        ))}
      </div>
    </div>
  );
}

export default function MemoryMap({ mm }: { mm: MemoryMapData }) {
  const g = mm.gpu;
  const layers = mm.layers;
  const allOnGpu = layers && layers.gpu === layers.total;

  const headline = !mm.parsed
    ? "Waiting for the model to load…"
    : g && allOnGpu
      ? `The whole model is on your GPU, so it runs at full speed. ${fmtMB(g.free)} of VRAM is still free.`
      : layers && layers.gpu > 0
        ? `${layers.gpu} of ${layers.total} layers are on the GPU; the rest run on the CPU, which is slower.`
        : "The model is running on the CPU.";

  return (
    <div className="memory-map">
      <p className="mm-headline">{headline}</p>

      {g && (
        <div className="mm-block">
          <div className="mm-title"><span><Term k="vram">GPU memory</Term> · {g.name.replace("NVIDIA GeForce ", "")}</span><span className="muted">{fmtMB(g.used)} of {fmtMB(g.total)} used</span></div>
          <Bar title="GPU memory" total={g.total} segs={[
            { key: "model", label: "Model weights", mb: g.model, color: "var(--series-1)" },
            { key: "context", label: "Context memory", term: "kv", mb: g.context, color: "var(--series-2)" },
            { key: "compute", label: "Working buffers", term: "compute", mb: g.compute, color: "var(--series-3)" },
            { key: "other", label: "Other apps", term: "other", mb: g.other, color: "var(--neutral-seg)" },
            { key: "free", label: "Free", mb: g.free, color: "" },
          ]} />
        </div>
      )}

      <div className="mm-block">
        <div className="mm-title"><span>System memory (RAM)</span><span className="muted">{fmtMB(mm.ram.used)} of {fmtMB(mm.ram.total)} used</span></div>
        <Bar title="System memory" total={mm.ram.total} segs={[
          { key: "model", label: "Model weights", mb: mm.ram.model, color: "var(--series-1)" },
          { key: "context", label: "Context memory", term: "kv", mb: mm.ram.context, color: "var(--series-2)" },
          { key: "compute", label: "Working buffers", term: "compute", mb: mm.ram.compute, color: "var(--series-3)" },
          { key: "other", label: "Everything else", mb: mm.ram.other, color: "var(--neutral-seg)" },
          { key: "free", label: "Free", mb: mm.ram.free, color: "" },
        ]} />
      </div>

      {layers && (
        <div className="mm-block">
          <div className="mm-title"><span><Term k="layers">Layers</Term></span><span className="muted">{layers.gpu} of {layers.total} on GPU</span></div>
          <div className="layer-strip" role="img" aria-label={`${layers.gpu} of ${layers.total} layers on the GPU`}>
            {Array.from({ length: layers.total }, (_, i) => (
              <span key={i} className={i < layers.gpu ? "gpu" : "cpu"} title={`Layer ${i + 1}: ${i < layers.gpu ? "GPU" : "CPU"}`} />
            ))}
          </div>
          <div className="mm-legend">
            <span><i style={{ background: "var(--series-1)" }} />On GPU (fast)</span>
            <span><i style={{ background: "var(--series-3)" }} />On CPU (slow)</span>
          </div>
        </div>
      )}

      <Pro>
        <div className="mm-pro mono small">
          {mm.kv && <>KV cache {mm.kv.mib} MiB · {mm.kv.cells.toLocaleString()} cells · {mm.kv.attention_layers} attention layers · {mm.runtime.kv}<br /></>}
          {mm.fixed_state_mb > 0 && <>Recurrent state {mm.fixed_state_mb} MiB (fixed size, hybrid architecture)<br /></>}
          ctx {mm.runtime.ctx.toLocaleString()} · ngl {mm.runtime.ngl} · threads {mm.runtime.threads} · source: llama.cpp load log + nvidia-smi
        </div>
      </Pro>
    </div>
  );
}
