import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Brain, Download, Loader2, Play, Square, Wand2 } from "lucide-react";
import { api, TorchJob, TorchStatus } from "./api";
import { LineChart, Tabs } from "./ui";

/** The PyTorch side: original Hugging Face checkpoints, activations inside the network, LoRA fine-tuning back to GGUF. */

function useJobs(kind: string) {
  const [jobs, setJobs] = useState<TorchJob[]>([]);
  const active = jobs.some((j) => j.status === "running" || j.status === "queued");
  useEffect(() => {
    const tick = () => api.torchJobs(kind).then(setJobs).catch(() => {});
    tick();
    const t = setInterval(tick, active ? 3000 : 15000);
    return () => clearInterval(t);
  }, [kind, active]);
  return [jobs, setJobs] as const;
}

function JobLine({ j }: { j: TorchJob }) {
  return (
    <div className={`ws-job ${j.status}`}>
      {j.status === "running" || j.status === "queued" ? <Loader2 className="spin" size={15} /> : j.status === "done" ? null : <AlertCircle size={15} />}
      <div><b>#{j.id} · {j.kind === "download" ? j.params.repo : j.kind === "lora" ? `LoRA on ${j.params.checkpoint} · ${j.params.data}` : `${j.params.checkpoint}`}</b>
        <span>{j.status === "running" ? `${j.step ?? "starting"}…` : j.status}{j.error ? `: ${j.error}` : ""}</span></div>
      {(j.status === "running" || j.status === "queued") && <button className="btn small ghost" onClick={() => api.torchCancel(j.id)}><Square /> Stop</button>}
    </div>
  );
}

/* ---------- Heatmap (layers × tokens) ---------- */

function Heat({ title, sub, rows, tokens, log = false }: { title: string; sub: string; rows: (number[] | null)[]; tokens: string[]; log?: boolean }) {
  const [hover, setHover] = useState<string | null>(null);
  const vals = rows.flatMap((r) => r ?? []).filter((v) => Number.isFinite(v));
  if (!vals.length) return null;
  const f = (v: number) => (log ? Math.log10(Math.max(v, 1e-6)) : v);
  const lo = Math.min(...vals.map(f)), hi = Math.max(...vals.map(f));
  return (
    <div className="hm">
      <div className="mini-title">{title} <span className="muted">{sub}</span></div>
      <div className="hm-grid" style={{ gridTemplateColumns: `36px repeat(${tokens.length}, minmax(10px, 1fr))` }} onMouseLeave={() => setHover(null)}>
        {rows.map((r, li) => r && (
          <div key={li} className="hm-row" style={{ display: "contents" }}>
            <span className="hm-l">{li === 0 && rows.length > 1 && r.length ? "emb" : li}</span>
            {r.map((v, ti) => (
              <div key={ti} className="hm-c" style={{ background: `color-mix(in srgb, var(--series-1) ${Math.round(8 + 92 * ((f(v) - lo) / (hi - lo || 1)))}%, var(--surface-2))` }}
                onMouseEnter={() => setHover(`layer ${li} · token “${tokens[ti]}” · ${v.toFixed(2)}`)} />
            ))}
          </div>
        ))}
        <span />{tokens.map((t, i) => <span key={i} className="hm-t" title={t}>{t.replace(/\n/g, "⏎").trim() || "·"}</span>)}
      </div>
      <div className="row-sub">{hover ?? `Darker = larger${log ? " (log scale)" : ""}. Hover a cell.`}</div>
    </div>
  );
}

function Activations({ status }: { status: TorchStatus | null }) {
  const [jobs, setJobs] = useJobs("activations");
  const [ckpt, setCkpt] = useState("");
  const [prompt, setPrompt] = useState("The capital of France is Paris. The capital of Japan is");
  const [sel, setSel] = useState<number | null>(null);
  const [detail, setDetail] = useState<TorchJob | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { if (!ckpt && status?.checkpoints[0]) setCkpt(status.checkpoints[0].id); }, [status]); // eslint-disable-line react-hooks/exhaustive-deps
  const shown = jobs.find((j) => j.id === sel) ?? jobs.find((j) => j.status === "done");
  useEffect(() => { if (shown?.status === "done") api.torchJob(shown.id).then(setDetail).catch(() => {}); }, [shown?.id, shown?.status]);
  const r = detail?.id === shown?.id ? detail?.results as ActResult | undefined : undefined;
  const run = async () => {
    setErr(null);
    try { const { id } = await api.torchActivations(ckpt, prompt); setSel(id); setJobs(await api.torchJobs("activations")); } catch (e) { setErr((e as Error).message); }
  };
  return (
    <div className="ws">
      <div className="row-sub">Run one prompt through the original (unquantized) model and look inside: how big the signal in each layer is, how much each layer changes it, and what the model “would answer” if it stopped at each layer (the logit lens).</div>
      <div className="ws-grid">
        <label><span>Checkpoint</span><select className="field select" value={ckpt} onChange={(e) => setCkpt(e.target.value)}>
          {(status?.checkpoints ?? []).map((c) => <option key={c.id} value={c.id}>{c.repo} · {c.layers} layers</option>)}</select></label>
        <label style={{ gridColumn: "span 2" }}><span>Prompt</span><textarea className="field" rows={2} value={prompt} onChange={(e) => setPrompt(e.target.value)} /></label>
      </div>
      <div className="xc-btns"><button className="btn" disabled={!ckpt} onClick={run}><Play /> Look inside</button>
        {jobs.length > 1 && <select className="field select small-select" value={shown?.id ?? ""} onChange={(e) => setSel(Number(e.target.value))}>
          {jobs.map((j) => <option key={j.id} value={j.id}>#{j.id} · {j.params.prompt?.slice(0, 40)}</option>)}</select>}</div>
      {err && <div className="error"><AlertCircle /> {err}</div>}
      {shown && shown.status !== "done" && <JobLine j={shown} />}
      {r && (
        <>
          <div className="lens">
            <div className="mini-title">Logit lens <span className="muted">(the next-word prediction read out after each layer, for the last token)</span></div>
            <div className="lens-rows">{r.logit_lens.map((top, li) => {
              const final = r.logit_lens[r.logit_lens.length - 1][0]?.token;
              return (
                <div key={li} className="lens-row"><span className="hm-l">{li === 0 ? "emb" : li}</span>
                  {top.slice(0, 3).map((t, k) => <span key={k} className={`lens-tok ${t.token === final ? "final" : ""}`} style={{ opacity: 0.35 + 0.65 * t.p }} title={`${(t.p * 100).toFixed(1)}%`}>{JSON.stringify(t.token)} <em>{(t.p * 100).toFixed(0)}%</em></span>)}</div>
              );
            })}</div>
            <div className="row-sub">Highlighted: the model's final answer, wherever it already shows up in earlier layers. Early layers predict noise; the answer usually forms in the last third.</div>
          </div>
          <Heat title="Residual stream size" sub="(L2 norm of each layer's output, per token)" rows={r.hidden_norm} tokens={r.tokens} log />
          <Heat title="How much each layer changes it" sub="(‖layer output − layer input‖ per token)" rows={r.delta_norm} tokens={r.tokens} log />
          {r.attn_entropy && r.attn_entropy.some(Boolean) && <Heat title="Attention spread" sub="(entropy of each token's attention, mean over heads; high = looks everywhere, low = focused)" rows={[null, ...r.attn_entropy]} tokens={r.tokens} />}
          {!r.attn_entropy && <div className="row-sub">This architecture doesn't expose attention maps (e.g. linear-attention layers), so only residual-stream views are shown.</div>}
          {r.knockout && <Knockout k={r.knockout} />}
          {r.timing && <LayerTiming t={r.timing} />}
          <div className="mini-title">Largest residual dimensions <span className="muted">(mean |activation| over the prompt; a few “massive activation” dimensions often dominate every layer)</span></div>
          <div className="td-grid">{r.top_dims.slice(1).map((d, i) => (
            <div key={i}><span className="hm-l">{i + 1}</span>{d.slice(0, 3).map((x) => <span key={x.dim} className="lens-tok">#{x.dim} <em>{x.mean_abs.toFixed(2)}</em></span>)}</div>
          ))}</div>
        </>
      )}
    </div>
  );
}

type LayerTime = { layer: number; total: number; attention: number; ffn: number; other: number };
type ActResult = { tokens: string[]; hidden_norm: number[][]; delta_norm: (number[] | null)[]; logit_lens: { token: string; p: number }[][];
  attn_entropy: (number[] | null)[] | null; top_dims: { dim: number; mean_abs: number }[][];
  timing?: { prefill: LayerTime[]; decode: LayerTime[]; prompt_tokens: number; note: string };
  knockout?: { answer: string; p_answer: number; layers: { layer: number; kl: number; p_answer: number; new_top: string; changed: boolean }[] } };

function LayerTiming({ t }: { t: NonNullable<ActResult["timing"]> }) {
  const [which, setWhich] = useState<"decode" | "prefill">("decode");
  const rows = t[which];
  const max = Math.max(...rows.map((r) => r.total), 1e-6);
  const total = rows.reduce((a, r) => a + r.total, 0);
  const sum = (k: "attention" | "ffn" | "other") => rows.reduce((a, r) => a + r[k], 0);
  const P = [{ k: "attention" as const, label: "Attention / mixer", c: "var(--series-2)" }, { k: "ffn" as const, label: "Feed-forward", c: "var(--series-1)" }, { k: "other" as const, label: "Rest (norms, residual)", c: "var(--split-other)" }];
  return (
    <div className="lt">
      <div className="lt-head"><div className="mini-title">GPU time per layer <span className="muted">({which === "decode" ? "one next token, with the KV cache" : `the ${t.prompt_tokens}-token prompt`}; total {total.toFixed(2)} ms)</span></div>
        <div className="segmented"><button className={which === "decode" ? "on" : ""} onClick={() => setWhich("decode")}>Decode</button><button className={which === "prefill" ? "on" : ""} onClick={() => setWhich("prefill")}>Prefill</button></div></div>
      <div className="xlegend top">{P.map((p) => <span key={p.k}><i className="sq" style={{ background: p.c }} />{p.label} {(sum(p.k) / total * 100).toFixed(0)}%</span>)}</div>
      <div className="lt-bars">{rows.map((r) => (
        <div key={r.layer} className="lt-row" title={`Layer ${r.layer}: ${r.total.toFixed(3)} ms (attention ${r.attention.toFixed(3)}, feed-forward ${r.ffn.toFixed(3)}, rest ${r.other.toFixed(3)})`}>
          <span className="hm-l">{r.layer}</span>
          <div className="lt-track" style={{ width: `${(r.total / max) * 100}%` }}>{P.map((p) => r[p.k] > 0 && <div key={p.k} style={{ flex: r[p.k], background: p.c }} />)}</div>
        </div>))}</div>
      <div className="row-sub">{t.note}</div>
    </div>
  );
}

function Knockout({ k }: { k: NonNullable<ActResult["knockout"]> }) {
  const max = Math.max(...k.layers.map((l) => l.kl), 1e-6);
  return (
    <div className="lt">
      <div className="mini-title">Layer knockout <span className="muted">(skip one layer, keep everything else: how much does the prediction move? Original answer {JSON.stringify(k.answer)} at {(k.p_answer * 100).toFixed(0)}%)</span></div>
      <div className="lt-bars">{k.layers.map((l) => (
        <div key={l.layer} className="lt-row" title={`Without layer ${l.layer}: KL ${l.kl.toFixed(4)} · p(${k.answer}) ${(l.p_answer * 100).toFixed(1)}%${l.changed ? ` · answer becomes ${JSON.stringify(l.new_top)}` : ""}`}>
          <span className="hm-l">{l.layer}</span>
          <div className="lt-track" style={{ width: `${Math.max(1, (l.kl / max) * 100)}%` }}><div style={{ flex: 1, background: l.changed ? "var(--series-2)" : "var(--series-4)" }} /></div>
          <em className="lt-note">{l.changed ? `→ ${JSON.stringify(l.new_top)}` : `${(l.p_answer * 100).toFixed(0)}%`}</em>
        </div>))}</div>
      <div className="xlegend"><span><i className="sq" style={{ background: "var(--series-2)" }} />answer changes</span><span><i className="sq" style={{ background: "var(--series-4)" }} />answer kept (bar = KL divergence; right: probability of the original answer)</span></div>
    </div>
  );
}

function FineTune({ status }: { status: TorchStatus | null }) {
  const [jobs, setJobs] = useJobs("lora");
  const [f, setF] = useState({ checkpoint: "", data: "yahma/alpaca-cleaned", split: "train", rows: 2000, steps: 200, lr: 0.0002, rank: 16, batch: 4, max_len: 512, outtype: "q8_0", name: "", qlora: false });
  const [sel, setSel] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { if (!f.checkpoint && status?.checkpoints[0]) setF((x) => ({ ...x, checkpoint: status.checkpoints[0].id })); }, [status]); // eslint-disable-line react-hooks/exhaustive-deps
  const shown = jobs.find((j) => j.id === sel) ?? jobs[0];
  const curve = useMemo(() => ((shown?.results?.curve as { step: number; loss: number }[]) ?? []).map((p) => ({ x: p.step, y: p.loss })), [shown]);
  const set = (k: string, v: string | number | boolean) => setF((x) => ({ ...x, [k]: v }));
  const start = async () => {
    setErr(null);
    try { const { id } = await api.torchLora(f); setSel(id); setJobs(await api.torchJobs("lora")); } catch (e) { setErr((e as Error).message); }
  };
  return (
    <div className="ws">
      <div className="row-sub">Train a LoRA adapter on a dataset, merge it into the model and convert it to GGUF with llama.cpp's own converter. The result appears under <b>Weights</b> / <b>A different model</b> in Experiments, so you can profile exactly what fine-tuning changed: quality, drift, speed. Full-precision fine-tuning needs a lot of GPU memory: as a rule of thumb, an 8 GB GPU handles models up to about 1–2B parameters.</div>
      <div className="ws-grid">
        <label><span>Checkpoint</span><select className="field select" value={f.checkpoint} onChange={(e) => set("checkpoint", e.target.value)}>
          {(status?.checkpoints ?? []).map((c) => <option key={c.id} value={c.id}>{c.repo}</option>)}</select></label>
        <label><span>Dataset (Hugging Face id or id:config)</span><input className="field" value={f.data} onChange={(e) => set("data", e.target.value)} />
          <em>Columns recognised: instruction/output, prompt/response, question/answer or chat messages.</em></label>
        <label><span>Rows · steps · batch</span><div className="ft-row">
          <input className="field" type="number" value={f.rows} onChange={(e) => set("rows", Number(e.target.value))} />
          <input className="field" type="number" value={f.steps} onChange={(e) => set("steps", Number(e.target.value))} />
          <input className="field" type="number" value={f.batch} onChange={(e) => set("batch", Number(e.target.value))} /></div></label>
        <label><span>LoRA rank · learning rate · max length</span><div className="ft-row">
          <input className="field" type="number" value={f.rank} onChange={(e) => set("rank", Number(e.target.value))} />
          <input className="field" type="number" step="0.00005" value={f.lr} onChange={(e) => set("lr", Number(e.target.value))} />
          <input className="field" type="number" value={f.max_len} onChange={(e) => set("max_len", Number(e.target.value))} /></div></label>
        <label className="ws-check"><input type="checkbox" checked={f.qlora} onChange={(e) => setF((x) => ({ ...x, qlora: e.target.checked }))} />
          <span><b>QLoRA (4-bit base)</b><em>Loads the frozen base model in 4-bit while training, so roughly 4× bigger models fit in the same GPU memory. Merging afterwards reloads the base in 16-bit on the CPU, so it needs about 2 bytes of system RAM per parameter.</em></span></label>
        <label><span>GGUF type · name</span><div className="ft-row">
          <select className="field select" value={f.outtype} onChange={(e) => set("outtype", e.target.value)}><option>q8_0</option><option>f16</option><option>bf16</option></select>
          <input className="field" placeholder="optional" value={f.name} onChange={(e) => set("name", e.target.value)} /></div></label>
      </div>
      <div className="xc-btns"><button className="btn" disabled={!f.checkpoint} onClick={start}><Wand2 /> Fine-tune</button>
        {jobs.length > 1 && <select className="field select small-select" value={shown?.id ?? ""} onChange={(e) => setSel(Number(e.target.value))}>
          {jobs.map((j) => <option key={j.id} value={j.id}>#{j.id} · {j.params.data} · {j.status}</option>)}</select>}</div>
      {err && <div className="error"><AlertCircle /> {err}</div>}
      {shown && <JobLine j={shown} />}
      {curve.length > 1 && <div><div className="mini-title">Training loss <span className="muted">(per step, answer tokens only)</span></div>
        <LineChart points={curve} height={140} color="var(--series-1)" unit="" fmtX={(x) => `step ${x}`} /></div>}
      {shown?.status === "done" && !!shown.results.gguf && <div className="row-sub">Saved <b>{String(shown.results.gguf)}</b> ({((Number(shown.results.size_bytes) || 0) / 2 ** 30).toFixed(2)} GB) with a recipe. Loss {String(shown.results.first_loss)} → {String(shown.results.final_loss)}. Open Experiments and add it as a run.</div>}
    </div>
  );
}

function Checkpoints({ status, reload }: { status: TorchStatus | null; reload: () => void }) {
  const [jobs, setJobs] = useJobs("download");
  const [repo, setRepo] = useState("Qwen/Qwen3.5-0.8B");
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { if (jobs.some((j) => j.status === "done")) reload(); }, [jobs.filter((j) => j.status === "done").length]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="ws">
      <div className="row-sub">PyTorch work needs the original Hugging Face weights (safetensors), not GGUF. Downloads go to /opt/blocky/hf.</div>
      <div className="xc-btns"><input className="field" value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="owner/name" />
        <button className="btn" onClick={async () => { setErr(null); try { await api.torchDownload(repo); setJobs(await api.torchJobs("download")); } catch (e) { setErr((e as Error).message); } }}><Download /> Download</button></div>
      {err && <div className="error"><AlertCircle /> {err}</div>}
      {jobs.filter((j) => j.status !== "done").slice(0, 3).map((j) => <JobLine key={j.id} j={j} />)}
      <div className="table-wrap"><table className="data"><thead><tr><th>Checkpoint</th><th>Architecture</th><th className="r">Layers</th><th className="r">Size</th></tr></thead>
        <tbody>{(status?.checkpoints ?? []).map((c) => <tr key={c.id}><td>{c.repo}</td><td>{c.architecture}</td><td className="r">{c.layers}</td><td className="r">{(c.size_bytes / 2 ** 30).toFixed(2)} GB</td></tr>)}</tbody></table></div>
    </div>
  );
}

export default function InternalsPage() {
  const [status, setStatus] = useState<TorchStatus | null>(null);
  const [tab, setTab] = useState("act");
  const load = () => api.torchStatus().then(setStatus).catch(() => {});
  useEffect(() => { load(); }, []);
  return (
    <div className="page">
      <div className="page-inner wide">
        <h1 className="page-title">Internals</h1>
        <p className="page-sub">The PyTorch side of the lab: look inside a model's layers, and fine-tune it, then bring the result back to GGUF to measure it like any other weights.</p>
        {status && !status.installed && <div className="group pad"><b>The PyTorch environment isn't installed.</b> <span className="row-sub">On the box: bash scripts/setup-ml.sh (a few GB).</span></div>}
        {status?.installed && <div className="row-sub ft-env"><Brain size={14} /> PyTorch {status.torch} · CUDA {status.cuda ? "on" : "off"} · transformers {status.transformers} · peft {status.peft}</div>}
        <Tabs value={tab} onChange={setTab} tabs={[{ id: "act", label: "Activations" }, { id: "ft", label: "Fine-tune (LoRA)" }, { id: "ckpt", label: "Checkpoints" }]} />
        <div className="group pad">
          {tab === "act" && <Activations status={status} />}
          {tab === "ft" && <FineTune status={status} />}
          {tab === "ckpt" && <Checkpoints status={status} reload={load} />}
        </div>
      </div>
    </div>
  );
}
