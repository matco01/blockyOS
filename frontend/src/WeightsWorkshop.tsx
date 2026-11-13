import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Check, Hammer, Loader2, Square } from "lucide-react";
import { api, TensorGroup, VariantSpec, WeightFile, WeightJob } from "./api";
import { Term } from "./ui";
import SensitivityScan from "./SensitivityScan";

/**
 * Weights workshop: build a custom version of the model's weights (quantization type, importance matrix,
 * per-tensor precision, layer pruning) with llama.cpp's own tools. The result shows up as a "Weights" option
 * in the experiment, ready to profile against the baseline.
 */

const PART_LABEL: Record<string, string> = { attention: "attention", linear: "linear attention", ffn: "feed-forward", experts: "experts", other: "other" };
const gbs = (b: number) => `${(b / 2 ** 30).toFixed(2)} GB`;

export default function WeightsWorkshop({ model, onBuilt }: { model: string; onBuilt: () => void }) {
  const [files, setFiles] = useState<WeightFile[]>([]);
  const [types, setTypes] = useState<{ base: string[]; tensor: string[]; needs: string[] }>({ base: [], tensor: [], needs: [] });
  const [source, setSource] = useState("");
  const [groups, setGroups] = useState<TensorGroup[]>([]);
  const [spec, setSpec] = useState<Omit<VariantSpec, "source">>({ qtype: "Q4_K_M", imatrix: true, overrides: {}, output_type: null, embedding_type: null, prune: [] });
  const [pruneText, setPruneText] = useState("");
  const [jobs, setJobs] = useState<WeightJob[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.weightsFamily(model).then((d) => {
      setFiles(d.files);
      setTypes({ base: d.base_types, tensor: d.tensor_types, needs: d.needs_imatrix });
      setSource((s) => s || d.files[0]?.file || model);
    }).catch((e) => setErr((e as Error).message));
  }, [model]);
  useEffect(() => { if (source) api.weightsTensors(source).then(setGroups).catch(() => setGroups([])); }, [source]);

  // Poll jobs while any is active; tell the parent when one finishes so the new file appears as an option.
  const active = jobs.some((j) => j.status === "running" || j.status === "queued");
  useEffect(() => {
    let last = "";
    const tick = () => api.weightsJobs().then((js) => {
      setJobs(js);
      const sig = js.filter((j) => j.status === "done").map((j) => j.id).join(",");
      if (last && sig !== last) onBuilt();
      last = sig;
    }).catch(() => {});
    tick();
    const t = setInterval(tick, active ? 2000 : 10000);
    return () => clearInterval(t);
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  const src = files.find((f) => f.file === source);
  const lowSource = src && src.bits < 7.5;
  const needsImatrix = types.needs.includes(spec.qtype);
  const set = (p: Partial<typeof spec>) => setSpec((s) => ({ ...s, ...p }));
  const setOverride = (pat: string, t: string) => set({ overrides: Object.fromEntries(Object.entries({ ...spec.overrides, [pat]: t }).filter(([, v]) => v)) });
  const layers = src?.layers ?? 0;
  const prune = useMemo(() => pruneText.split(/[,\s]+/).filter(Boolean).map(Number).filter((n) => Number.isInteger(n)), [pruneText]);
  const pruneBad = prune.some((n) => n < 0 || n >= layers);

  // Rough size estimate: each overridden group at its new type; the rest scaled to the base type.
  const BITS: Record<string, number> = { q2_k: 2.6, q3_k: 3.4, q4_0: 4.5, q4_k: 4.5, q5_k: 5.5, q6_k: 6.6, q8_0: 8.5, iq4_xs: 4.25, f16: 16, bf16: 16 };
  const BASEBITS: Record<string, number> = { Q2_K: 3.0, Q3_K_S: 3.5, Q3_K_M: 3.9, Q4_0: 4.5, Q4_K_S: 4.6, Q4_K_M: 4.9, Q5_K_S: 5.5, Q5_K_M: 5.7, Q6_K: 6.6, Q8_0: 8.5,
    IQ4_XS: 4.3, IQ4_NL: 4.5, IQ3_M: 3.7, IQ3_XXS: 3.1, IQ2_M: 2.7, IQ2_XS: 2.4 };
  const estimate = useMemo(() => {
    if (!src || !groups.length) return null;
    const params = groups.reduce((a, g) => a + g.bytes, 0) * 8 / src.bits; // ≈ parameter count
    let bits = 0;
    for (const g of groups) {
      const p = (g.bytes * 8) / src.bits;
      const share = layers && g.count === layers ? 1 - prune.length / layers : 1;
      const o = spec.overrides[g.pattern] ?? (g.pattern === "output" ? spec.output_type : g.pattern === "token_embd" ? spec.embedding_type : null);
      bits += p * share * (o ? BITS[o] ?? BASEBITS[spec.qtype] : g.types.every((t) => t === "F32") ? 32 : BASEBITS[spec.qtype] ?? 5);
    }
    return { bytes: bits / 8, bpw: bits / params };
  }, [src, groups, spec, prune, layers]); // eslint-disable-line react-hooks/exhaustive-deps

  const build = async () => {
    setErr(null);
    try {
      await api.weightsCreate({ source, ...spec, imatrix: spec.imatrix || needsImatrix, prune });
      setJobs(await api.weightsJobs());
    } catch (e) { setErr((e as Error).message); }
  };
  const preset = (kind: string) => {
    if (kind === "clear") return set({ overrides: {}, output_type: null, embedding_type: null });
    if (kind === "attn8") return set({ overrides: { ...spec.overrides, ...Object.fromEntries(groups.filter((g) => g.part === "attention" || g.part === "linear").filter((g) => !g.types.every((t) => t === "F32")).map((g) => [g.pattern, "q8_0"])) } });
    if (kind === "edges8") return set({ output_type: "q8_0", embedding_type: "q8_0" });
    if (kind === "down6") return set({ overrides: { ...spec.overrides, ffn_down: "q6_k" } });
  };
  const editable = groups.filter((g) => !g.types.every((t) => t === "F32") && g.pattern !== "output" && g.pattern !== "token_embd");

  return (
    <div className="ws">
      <div className="ws-grid">
        <label><span>Start from</span>
          <select className="field select" value={source} onChange={(e) => setSource(e.target.value)}>
            {files.map((f) => <option key={f.file} value={f.file}>{f.file.replace(".gguf", "")} · {f.quant} · {gbs(f.size_bytes)}</option>)}
          </select>
          {lowSource && <em className="ws-warn"><AlertCircle size={13} /> This source is already compressed to ~{src!.bits.toFixed(1)} bits: re-compressing it stacks errors. Get the Q8_0 or BF16 file of this model in Models → Get models for clean results.</em>}
        </label>
        <label><span>Base <Term k="quant">quantization</Term></span>
          <select className="field select" value={spec.qtype} onChange={(e) => set({ qtype: e.target.value })}>
            {types.base.map((t) => <option key={t}>{t}</option>)}
          </select>
          <em>Applies to every tensor you don't override below.</em>
        </label>
        <label className="ws-check"><input type="checkbox" checked={spec.imatrix || needsImatrix} disabled={needsImatrix} onChange={(e) => set({ imatrix: e.target.checked })} />
          <span><b>Importance matrix</b><em>Measures which weights matter most on calibration text (WikiText-2 train, separate from the test text), so compression spends its bits there. Usually a clear quality win at 4 bits and below. {needsImatrix && "Required for this type."} Computed once per source, about 3–5 min.</em></span>
        </label>
      </div>

      <SensitivityScan model={source}
        onProtect={(layers) => set({ overrides: { [`blk\\.(${layers.join("|")})\\.`]: "q8_0", ...Object.fromEntries(Object.entries(spec.overrides).filter(([k]) => !k.startsWith("blk\\."))) } })}
        onPrune={(layers) => setPruneText(layers.join(", "))} />

      {Object.keys(spec.overrides).some((k) => k.startsWith("blk\\.")) && (
        <div className="sc-rules">{Object.entries(spec.overrides).filter(([k]) => k.startsWith("blk\\.")).map(([k, t]) => (
          <span key={k} className="xchip">layers {k.replace(/^blk\\\.\(?/, "").replace(/\)?\\\.$/, "").split("|").join(", ")} → {t}
            <button onClick={() => set({ overrides: Object.fromEntries(Object.entries(spec.overrides).filter(([x]) => x !== k)) })}>×</button></span>
        ))}<span className="row-sub">Layer rules are applied before the per-tensor choices below (first match wins in llama-quantize).</span></div>
      )}

      <div className="ws-sub"><b>Per-tensor precision</b><span className="row-sub">Keep sensitive parts at higher precision. Sizes are in the source file.</span>
        <div className="ws-presets">
          <button className="btn small ghost" onClick={() => preset("attn8")}>Attention at 8-bit</button>
          <button className="btn small ghost" onClick={() => preset("edges8")}>Embeddings + output at 8-bit</button>
          <button className="btn small ghost" onClick={() => preset("down6")}>ffn_down at 6-bit</button>
          <button className="btn small ghost" onClick={() => preset("clear")}>Clear</button>
        </div>
      </div>
      <div className="table-wrap ws-table"><table className="data">
        <thead><tr><th>Tensor</th><th>Part</th><th className="r">Count</th><th className="r">Size</th><th>Now</th><th>New type</th></tr></thead>
        <tbody>
          {[{ pattern: "output", label: "output (head)", key: "output_type" as const }, { pattern: "token_embd", label: "token_embd (embeddings)", key: "embedding_type" as const }].map((row) => {
            const g = groups.find((x) => x.pattern === row.pattern);
            if (!g) return null;
            return <tr key={row.pattern}><td className="mono">{row.label}</td><td>{PART_LABEL[g.part]}</td><td className="r">{g.count}</td><td className="r">{gbs(g.bytes)}</td><td>{g.types.join(", ")}</td>
              <td><select className="field select small-select" value={spec[row.key] ?? ""} onChange={(e) => set({ [row.key]: e.target.value || null })}>
                <option value="">default</option>{types.tensor.map((t) => <option key={t}>{t}</option>)}</select></td></tr>;
          })}
          {editable.map((g) => (
            <tr key={g.pattern} className={spec.overrides[g.pattern] ? "ws-on" : ""}><td className="mono">{g.pattern}</td><td>{PART_LABEL[g.part]}</td><td className="r">{g.count}</td><td className="r">{gbs(g.bytes)}</td><td>{g.types.join(", ")}</td>
              <td><select className="field select small-select" value={spec.overrides[g.pattern] ?? ""} onChange={(e) => setOverride(g.pattern, e.target.value)}>
                <option value="">base type</option>{types.tensor.map((t) => <option key={t}>{t}</option>)}</select></td></tr>
          ))}
        </tbody>
      </table></div>

      <div className="ws-grid">
        <label><span>Remove layers <span className="muted">(pruning, optional)</span></span>
          <input className="field" placeholder={layers ? `e.g. ${layers - 3}, ${layers - 2}  (0–${layers - 1})` : "layer numbers"} value={pruneText} onChange={(e) => setPruneText(e.target.value)} />
          <em className={pruneBad ? "ws-warn" : ""}>{pruneBad ? `Layer numbers go from 0 to ${layers - 1}.` : "Deletes whole layers: smaller and faster, usually at a real quality cost. KL divergence will show exactly how much."}</em>
        </label>
        <div className="ws-est">
          {estimate && <><b>≈ {gbs(estimate.bytes)}</b><span>{estimate.bpw.toFixed(2)} bits per weight (estimate)</span></>}
          <button className="btn" disabled={!source || pruneBad} onClick={build}><Hammer /> Build variant</button>
        </div>
      </div>
      {err && <div className="error"><AlertCircle /> {err}</div>}

      {jobs.length > 0 && (
        <div className="ws-jobs">
          {jobs.slice(0, 5).map((j) => (
            <div key={j.id} className={`ws-job ${j.status}`}>
              {j.status === "running" ? <Loader2 className="spin" size={15} /> : j.status === "done" ? <Check size={15} /> : j.status === "queued" ? <Loader2 size={15} /> : <AlertCircle size={15} />}
              <div><b>{j.out.replace(".gguf", "")}</b>
                <span>{j.status === "running" ? `${j.step ?? "starting"}…` : j.status === "done" ? `ready · ${gbs(j.size_bytes ?? 0)} · now selectable under Weights` : j.status}{j.error ? `: ${j.error}` : ""}</span>
                {j.status === "running" && j.log_tail.length > 0 && <code>{j.log_tail[j.log_tail.length - 1].slice(0, 140)}</code>}
              </div>
              {j.status === "running" && <button className="btn small ghost" onClick={() => api.weightsCancel(j.id)}><Square /> Stop</button>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
