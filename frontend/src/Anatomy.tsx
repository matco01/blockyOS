import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Square } from "lucide-react";
import { api, ModelFile } from "./api";
import { fmtMB } from "./ui";

type Part = { bytes: number; params: number; types: string[]; tensors: number };
type Layer = { index: number; mixer: "attention" | "linear"; parts: Record<string, Part>; bytes: number; params: number };
type AnatomyData = {
  file: string; name: string | null; architecture: string | null; layers: Layer[];
  embedding: { bytes: number; params: number; types: string[] }; output: { bytes: number; params: number; types: string[] };
  total_params: number; total_bytes: number; experts: number | null; experts_used: number | null;
  heads: number | null; kv_heads: number | null; embedding_length: number | null; tensor_count: number; tied_output: boolean;
};

const PARTS: { key: string; label: string; color: string; what: string }[] = [
  { key: "attention", label: "Attention", color: "var(--series-1)",
    what: "Looks back at every earlier word to decide which ones matter right now. Its memory (the context cache) grows with the conversation." },
  { key: "linear", label: "Linear attention", color: "var(--series-4)",
    what: "A newer, cheaper way to look back: keeps a fixed-size running summary instead of remembering every word. That's why hybrid models need much less context memory." },
  { key: "ffn", label: "Feed-forward", color: "var(--series-2)",
    what: "Where most of the model's knowledge lives. Usually the biggest part of each layer." },
  { key: "experts", label: "Experts (MoE)", color: "var(--series-3)",
    what: "Many small feed-forward blocks; a router picks only a few per word. Big knowledge, small compute." },
  { key: "other", label: "Normalization", color: "var(--neutral-seg)",
    what: "Tiny scaling steps that keep the numbers in a healthy range between parts." },
];

const fmtParams = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)} billion` : `${(n / 1e6).toFixed(0)} million`);

export default function Anatomy({ models, active }: { models: ModelFile[]; active?: string | null }) {
  const [file, setFile] = useState(active ?? models[0]?.file ?? "");
  const [data, setData] = useState<AnatomyData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [pulse, setPulse] = useState<number | null>(null); // -1 = embedding, layers.length = output
  const [msPerToken, setMsPerToken] = useState<number | null>(null);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => { if (!file && models[0]) setFile(active ?? models[0].file); }, [models, active, file]);
  useEffect(() => {
    if (!file) return;
    setData(null); setErr(null);
    api.modelAnatomy(file).then(setData).catch((e) => setErr((e as Error).message));
  }, [file]);
  useEffect(() => {
    api.recentStats(10).then((s) => {
      const v = s.map((x) => x.predicted_per_second).filter((x): x is number => !!x);
      if (v.length) setMsPerToken(1000 / (v.reduce((a, b) => a + b, 0) / v.length));
    }).catch(() => {});
    return () => window.clearInterval(timer.current);
  }, []);

  const maxLayer = useMemo(() => Math.max(1, ...(data?.layers.map((l) => l.bytes) ?? [1])), [data]);
  const present = useMemo(() => PARTS.filter((p) => data?.layers.some((l) => l.parts[p.key])), [data]);

  const play = () => {
    if (!data) return;
    window.clearInterval(timer.current);
    let i = -1;
    setPulse(-1);
    timer.current = window.setInterval(() => {
      i += 1;
      if (i > data.layers.length + 1) { window.clearInterval(timer.current); setPulse(null); return; }
      setPulse(i);
    }, 70);
  };
  const stop = () => { window.clearInterval(timer.current); setPulse(null); };

  if (err) return <div className="error">{err}</div>;
  const sel = data && hover != null ? data.layers[hover] : null;
  const att = data?.layers.filter((l) => l.mixer === "attention").length ?? 0;
  const lin = (data?.layers.length ?? 0) - att;
  const slow = msPerToken && data ? Math.round((70 * (data.layers.length + 2)) / msPerToken) : null;

  return (
    <div className="anatomy">
      <div className="anat-top">
        <select className="field select" value={file} onChange={(e) => setFile(e.target.value)}>
          {models.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file} · {m.quant}</option>)}
        </select>
        {pulse == null
          ? <button className="btn small" disabled={!data} onClick={play}><Play /> Watch a word go through</button>
          : <button className="btn small ghost" onClick={stop}><Square /> Stop</button>}
      </div>

      {data && (
        <div className="anat-chips">
          <span><b>{fmtParams(data.total_params)}</b> numbers (parameters)</span>
          <span><b>{data.layers.length}</b> layers{lin > 0 ? <> · {att} attention + {lin} linear</> : null}</span>
          <span><b>{fmtMB(data.total_bytes / 2 ** 20)}</b> on disk</span>
          {data.embedding_length && <span>each word becomes <b>{data.embedding_length.toLocaleString()}</b> numbers</span>}
          {data.experts && <span><b>{data.experts_used}</b> of <b>{data.experts}</b> experts per word</span>}
        </div>
      )}

      <div className="anat-body">
        <div className="anat-stack" onMouseLeave={() => setHover(null)}>
          {data && (
            <>
              <div className={`anat-end ${pulse === data.layers.length + 1 ? "lit" : ""}`}>
                <span>Output</span><small>{data.tied_output ? "reuses the embedding table" : fmtMB(data.output.bytes / 2 ** 20)} · scores every possible next word</small>
              </div>
              {[...data.layers].reverse().map((l) => (
                <div key={l.index} className={`anat-row ${hover === l.index ? "on" : ""} ${pulse === l.index + 1 ? "lit" : ""}`}
                  onMouseEnter={() => setHover(l.index)}>
                  <span className="anat-idx">{l.index + 1}</span>
                  <div className="anat-bar" style={{ width: `${(l.bytes / maxLayer) * 100}%` }}>
                    {PARTS.filter((p) => l.parts[p.key]).map((p) => (
                      <i key={p.key} style={{ flexGrow: l.parts[p.key].bytes, background: p.color }} />
                    ))}
                  </div>
                </div>
              ))}
              <div className={`anat-end ${pulse === 0 || pulse === -1 ? "lit" : ""}`}>
                <span>Embedding</span><small>{fmtMB(data.embedding.bytes / 2 ** 20)} · turns each word-piece into numbers</small>
              </div>
            </>
          )}
          {!data && <div className="row-sub">Reading the model file…</div>}
        </div>

        <div className="anat-side">
          {sel ? (
            <div className="anat-detail">
              <b>Layer {sel.index + 1} of {data!.layers.length}</b>
              <span className="row-sub">{sel.mixer === "linear" ? "Linear-attention layer" : "Classic attention layer"} · {fmtParams(sel.params)} numbers · {fmtMB(sel.bytes / 2 ** 20)}</span>
              {PARTS.filter((p) => sel.parts[p.key]).map((p) => (
                <div key={p.key} className="anat-part">
                  <i style={{ background: p.color }} />
                  <span>{p.label}</span>
                  <b>{fmtMB(sel.parts[p.key].bytes / 2 ** 20)}</b>
                  <em>{sel.parts[p.key].types.join(", ")}</em>
                </div>
              ))}
            </div>
          ) : (
            <div className="anat-detail">
              <b>How a word travels</b>
              <span className="row-sub">Your text is split into word-pieces. Each one enters at the bottom, gets refined by every layer in turn, and at the top the model scores every possible next word. Hover a layer for details.</span>
            </div>
          )}
          <div className="anat-legend">
            {present.map((p) => (
              <div key={p.key}><i style={{ background: p.color }} /><div><b>{p.label}</b><span>{p.what}</span></div></div>
            ))}
          </div>
          {pulse != null && slow && (
            <div className="row-sub">Slowed down about {slow}× so you can follow it: on your GPU one word goes through all layers in about {msPerToken!.toFixed(0)} ms.</div>
          )}
          <div className="row-sub">Bar width = how much memory each layer's weights use. Read straight from the model file, nothing estimated.</div>
        </div>
      </div>
    </div>
  );
}
