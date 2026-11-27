import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Brain, Download, FlaskRound, Loader2, Play, Square, Wand2 } from "lucide-react";
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
      <div><b>#{j.id} · {j.kind === "download" ? j.params.repo : j.kind === "lora" ? `LoRA on ${j.params.checkpoint} · ${j.params.data}` : j.kind === "train" ? `${j.params.mode === "pretrain" ? "Pretraining" : "Full fine-tune"} on ${j.params.checkpoint} · ${j.params.data}` : j.kind === "quantsim" ? `${qsName(j.params)} on ${j.params.checkpoint}` : `${j.params.checkpoint}`}</b>
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

/* ---------- Full training ---------- */

type TrainEval = { step: number; eval_loss: number; eval_ppl: number };

function FullTraining({ status }: { status: TorchStatus | null }) {
  const [jobs, setJobs] = useJobs("train");
  const [f, setF] = useState({ checkpoint: "", mode: "finetune", data: "wikitext:wikitext-2-raw-v1", split: "train", rows: 5000, steps: 500, lr: 0, batch: 4,
    accum: 1, max_len: 1024, eval_frac: 0.02, eval_every: 100, layers: 0, gpus: 1, nnodes: 1, master_addr: "", to_gguf: false, outtype: "q8_0", name: "" });
  const [sel, setSel] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { if (!f.checkpoint && status?.checkpoints[0]) setF((x) => ({ ...x, checkpoint: status.checkpoints[0].id })); }, [status]); // eslint-disable-line react-hooks/exhaustive-deps
  const shown = jobs.find((j) => j.id === sel) ?? jobs[0];
  const r = (shown?.results ?? {}) as { curve?: { step: number; loss: number }[]; evals?: TrainEval[]; checkpoint?: string; gguf?: string; gguf_error?: string;
    first_eval?: TrainEval; final_eval?: TrainEval; mfu_estimate?: number | null; mfu_note?: string; other_nodes_command?: string | null;
    end?: { seconds: number; tokens: number; tokens_per_second: number } | null; info?: { params: number; world_size: number; weights_dtype: string; train_examples: number; eval_examples: number } | null };
  const curve = useMemo(() => (r.curve ?? []).map((p) => ({ x: p.step, y: p.loss })), [r.curve]);
  const evalCurve = useMemo(() => (r.evals ?? []).map((p) => ({ x: p.step, y: p.eval_ppl })), [r.evals]);
  const set = (k: string, v: string | number | boolean) => setF((x) => ({ ...x, [k]: v }));
  const start = async () => {
    setErr(null);
    try {
      const body = { ...f, lr: f.lr || undefined, master_addr: f.nnodes > 1 ? f.master_addr : undefined };
      const { id } = await api.torchTrain(body); setSel(id); setJobs(await api.torchJobs("train"));
    } catch (e) { setErr((e as Error).message); }
  };
  return (
    <div className="ws">
      <div className="row-sub">Train <b>every weight</b>: fine-tune a checkpoint, or pretrain a model from random weights with its architecture and tokenizer. A held-out share of the data gives loss and perplexity before, during and after training, so you see generalization, not just the training loss. Several GPUs are sharded with FSDP; several machines run through torchrun. The result becomes a new checkpoint for Experiments (vLLM, SGLang, TensorRT-LLM), and optionally a GGUF for llama.cpp. Memory: full fine-tuning needs roughly 8–16 bytes per parameter on one GPU, so an 8 GB GPU fits models up to a few hundred million parameters; pretraining a smaller version (fewer layers) fits more.</div>
      <div className="ws-grid">
        <label><span>Checkpoint</span><select className="field select" value={f.checkpoint} onChange={(e) => set("checkpoint", e.target.value)}>
          {(status?.checkpoints ?? []).map((c) => <option key={c.id} value={c.id}>{c.repo}</option>)}</select></label>
        <label><span>Mode</span><select className="field select" value={f.mode} onChange={(e) => set("mode", e.target.value)}>
          <option value="finetune">Fine-tune all weights</option><option value="pretrain">Pretrain from scratch (same architecture)</option></select>
          {f.mode === "pretrain" && <em>Random weights; the checkpoint provides only the architecture and tokenizer.</em>}</label>
        <label><span>Data (Hugging Face id or id:config, or an absolute .txt / .jsonl path)</span><input className="field" value={f.data} onChange={(e) => set("data", e.target.value)} />
          <em>Text rows are packed into max-length blocks; chat-style rows (instruction/output, messages…) train on answers only.</em></label>
        <label><span>Rows · steps · max length</span><div className="ft-row">
          <input className="field" type="number" value={f.rows} onChange={(e) => set("rows", Number(e.target.value))} />
          <input className="field" type="number" value={f.steps} onChange={(e) => set("steps", Number(e.target.value))} />
          <input className="field" type="number" value={f.max_len} onChange={(e) => set("max_len", Number(e.target.value))} /></div></label>
        <label><span>Batch per GPU · accumulation · learning rate (0 = default)</span><div className="ft-row">
          <input className="field" type="number" value={f.batch} onChange={(e) => set("batch", Number(e.target.value))} />
          <input className="field" type="number" value={f.accum} onChange={(e) => set("accum", Number(e.target.value))} />
          <input className="field" type="number" step="0.00001" value={f.lr} onChange={(e) => set("lr", Number(e.target.value))} /></div></label>
        <label><span>Held-out share · evaluate every N steps{f.mode === "pretrain" ? " · layers (0 = same)" : ""}</span><div className="ft-row">
          <input className="field" type="number" step="0.01" value={f.eval_frac} onChange={(e) => set("eval_frac", Number(e.target.value))} />
          <input className="field" type="number" value={f.eval_every} onChange={(e) => set("eval_every", Number(e.target.value))} />
          {f.mode === "pretrain" && <input className="field" type="number" value={f.layers} onChange={(e) => set("layers", Number(e.target.value))} />}</div></label>
        <label><span>GPUs on this machine · machines</span><div className="ft-row">
          <input className="field" type="number" min={1} value={f.gpus} onChange={(e) => set("gpus", Number(e.target.value))} />
          <input className="field" type="number" min={1} value={f.nnodes} onChange={(e) => set("nnodes", Number(e.target.value))} /></div>
          {f.nnodes > 1 && <><input className="field" placeholder="This machine's network address, e.g. 192.168.1.40" value={f.master_addr} onChange={(e) => set("master_addr", e.target.value)} />
            <em>torchrun opens a port on that address while the job runs. Every machine needs the same checkpoint and data paths and the ML environment; the job shows the command to start on each of them.</em></>}</label>
        <label className="ws-check"><input type="checkbox" checked={f.to_gguf} onChange={(e) => set("to_gguf", e.target.checked)} />
          <span><b>Also convert to GGUF</b><em>For llama.cpp experiments (needs the llama.cpp source tree's converter).</em></span></label>
        <label><span>{f.to_gguf ? "GGUF type · name" : "Name"}</span><div className="ft-row">
          {f.to_gguf && <select className="field select" value={f.outtype} onChange={(e) => set("outtype", e.target.value)}><option>q8_0</option><option>f16</option><option>bf16</option></select>}
          <input className="field" placeholder="optional" value={f.name} onChange={(e) => set("name", e.target.value)} /></div></label>
      </div>
      <div className="xc-btns"><button className="btn" disabled={!f.checkpoint} onClick={start}><Wand2 /> Train</button>
        {jobs.length > 1 && <select className="field select small-select" value={shown?.id ?? ""} onChange={(e) => setSel(Number(e.target.value))}>
          {jobs.map((j) => <option key={j.id} value={j.id}>#{j.id} · {j.params.mode} · {j.params.data} · {j.status}</option>)}</select>}</div>
      {err && <div className="error"><AlertCircle /> {err}</div>}
      {shown && <JobLine j={shown} />}
      {r.other_nodes_command && shown?.status === "running" && <div className="row-sub">Start this on each other machine: <code style={{ userSelect: "all" }}>{r.other_nodes_command}</code></div>}
      {r.info && <div className="row-sub">{(r.info.params / 1e6).toFixed(0)}M parameters · {r.info.world_size} process{r.info.world_size > 1 ? "es (FSDP)" : ""} · weights {r.info.weights_dtype} · {r.info.train_examples} training / {r.info.eval_examples} held-out examples</div>}
      {curve.length > 1 && <div><div className="mini-title">Training loss <span className="muted">(per step)</span></div>
        <LineChart points={curve} height={140} color="var(--series-1)" unit="" fmtX={(x) => `step ${x}`} /></div>}
      {evalCurve.length > 1 && <div><div className="mini-title">Held-out perplexity <span className="muted">(data the model never trained on)</span></div>
        <LineChart points={evalCurve} height={120} color="var(--series-2)" unit="" fmtX={(x) => `step ${x}`} /></div>}
      {shown?.status === "done" && r.checkpoint && <div className="row-sub">Saved checkpoint <b>hf:{r.checkpoint}</b>{r.gguf ? <> and <b>{r.gguf}</b></> : null}{r.gguf_error ? ` (${r.gguf_error})` : ""}.
        {r.first_eval && r.final_eval ? <> Held-out perplexity {r.first_eval.eval_ppl.toFixed(2)} → {r.final_eval.eval_ppl.toFixed(2)}.</> : null}
        {r.end ? <> {Math.round(r.end.tokens_per_second).toLocaleString()} tokens/s over {(r.end.seconds / 60).toFixed(1)} min.</> : null}
        {r.mfu_estimate != null ? <> Estimated utilization {(r.mfu_estimate * 100).toFixed(0)}% <span title={r.mfu_note}>(spec-sheet peak; see note)</span>.</> : null}
        {" "}Open Experiments to profile it.</div>}
    </div>
  );
}

/* ---------- Technique playground ---------- */

const CUSTOM_EXAMPLE = `import torch

def quantize(w: torch.Tensor, name: str) -> torch.Tensor:
    """Return the weight as your method would reconstruct it (same shape)."""
    # Example: 3-bit symmetric round-to-nearest per output row, but keep attention values at 8 bit.
    bits = 8 if "v_proj" in name else 3
    qmax = 2 ** (bits - 1) - 1
    s = w.float().abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
    return ((w.float() / s).round().clamp(-qmax - 1, qmax) * s).to(w.dtype)
`;

const qsName = (p: Record<string, unknown>) => String(p.label || "") || [
  ["rtn", "gptq", "awq"].includes(String(p.method)) ? `${p.method === "rtn" ? "RTN" : p.method === "gptq" ? "GPTQ" : "AWQ-style"} ${p.grid === "q4_0" ? "· Q4_0 grid" : `${p.bits}-bit ${p.sym ? "sym" : "asym"}`}`
    : p.method === "none" ? "fp16 weights" : String(p.method).toUpperCase(),
  p.method !== "custom" && p.method !== "none" && p.grid !== "q4_0" ? `g${p.group}` : "", p.kv_bits ? `KV ${p.kv_bits}-bit` : "", p.include_head ? "+head" : ""].filter(Boolean).join(" · ");

type QsResult = { bits_per_weight: number | null; kld: number; kld_sd: number; kld_p99: number; same_top: number; ppl: number; ppl_base: number;
  weight_rel_error: number | null; output_rel_error?: number | null; per_layer: { layer: number | null; rel_error: number | null; out_error?: number | null }[];
  per_kind: Record<string, number | null>; per_kind_out?: Record<string, number | null>; tokens: number; seconds: number; note: string;
  calibration?: { chunks: number; tokens: number } | null };

function Playground({ status }: { status: TorchStatus | null }) {
  const [jobs, setJobs] = useJobs("quantsim");
  const [f, setF] = useState({ checkpoint: "", method: "gptq", grid: "int", bits: 4, group: 128, sym: true, include_head: false, kv_bits: 0, chunks: 16, code: CUSTOM_EXAMPLE, label: "" });
  const [exports, setExports] = useJobs("qexport");
  const [xType, setXType] = useState("Q4_0");
  const doExport = async (id: number) => {
    setErr(null);
    try { await api.torchQuantsimExport(id, xType); setExports(await api.torchJobs("qexport")); } catch (e) { setErr((e as Error).message); }
  };
  const [sel, setSel] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { if (!f.checkpoint && status?.checkpoints[0]) setF((x) => ({ ...x, checkpoint: status.checkpoints[0].id })); }, [status]); // eslint-disable-line react-hooks/exhaustive-deps
  const set = (k: string, v: string | number | boolean) => setF((x) => ({ ...x, [k]: v }));
  const start = async () => {
    setErr(null);
    try { const { id } = await api.torchQuantsim(f); setSel(id); setJobs(await api.torchJobs("quantsim")); } catch (e) { setErr((e as Error).message); }
  };
  const done = jobs.filter((j) => j.status === "done" && j.params.checkpoint === f.checkpoint);
  const shown = jobs.find((j) => j.id === sel) ?? jobs[0];
  const sr = shown?.status === "done" ? (shown.results as unknown as QsResult) : null;
  const useOut = !!sr?.per_layer.some((l) => l.out_error);
  const errOf = (l: QsResult["per_layer"][number]) => (useOut ? l.out_error : l.rel_error) ?? 0;
  const maxErr = Math.max(1e-9, ...(sr?.per_layer ?? []).map(errOf));
  const quantMethod = ["rtn", "gptq", "awq"].includes(f.method);
  return (
    <div className="ws">
      <div className="row-sub">Prototype a quantization idea in Python and see what it does to the model before anyone writes a kernel for it. Weights are quantized and dequantized back (“fake quantization”), so the model computes with exactly the values your method would store: the quality effect is real, speed and memory aren't measured. Every method is scored the same way against the unmodified checkpoint: KL divergence per token, same top token, perplexity, and weight and output error per layer. The model is processed one layer at a time, straight from the checkpoint files, so 7–14B models work on an 8 GB GPU; like the GPTQ paper, each layer is calibrated on the outputs of the already-quantized layers.</div>
      <div className="ws-grid">
        <label><span>Checkpoint</span><select className="field select" value={f.checkpoint} onChange={(e) => set("checkpoint", e.target.value)}>
          {(status?.checkpoints ?? []).map((c) => <option key={c.id} value={c.id}>{c.repo}</option>)}</select></label>
        <label><span>Weight method</span><select className="field select" value={f.method} onChange={(e) => set("method", e.target.value)}>
          <option value="gptq">GPTQ (error compensation, calibrated)</option><option value="awq">AWQ-style (activation-aware scales, calibrated)</option>
          <option value="rtn">Round-to-nearest (RTN)</option><option value="nf4">NF4 (NormalFloat 4-bit)</option><option value="ternary">Ternary (1.58-bit)</option>
          <option value="custom">Custom (your Python)</option><option value="none">None (KV only)</option></select></label>
        {quantMethod && <label><span>Grid</span><select className="field select" value={f.grid} onChange={(e) => set("grid", e.target.value)}>
          <option value="int">Integer: choose bits and group size</option><option value="q4_0">llama.cpp Q4_0 (4-bit, groups of 32): exports near-exactly</option></select>
          <em>On the Q4_0 grid, the exported GGUF stores the values simulated here (up to fp16 rounding of the scales), so a win here is a win in llama.cpp: measured KLD matches the simulation within a few percent.</em></label>}
        {quantMethod && f.grid === "int" && <label><span>Bits · symmetric</span><div className="ft-row">
          <select className="field select" value={f.bits} onChange={(e) => set("bits", Number(e.target.value))}>{[2, 3, 4, 5, 6, 8].map((b) => <option key={b} value={b}>{b}-bit</option>)}</select>
          <select className="field select" value={f.sym ? "sym" : "asym"} onChange={(e) => set("sym", e.target.value === "sym")}><option value="sym">symmetric</option><option value="asym">asymmetric (zero point)</option></select></div></label>}
        {f.method !== "custom" && f.method !== "none" && !(quantMethod && f.grid === "q4_0") && <label><span>Group size (weights per scale)</span><select className="field select" value={f.group} onChange={(e) => set("group", Number(e.target.value))}>
          {[32, 64, 128, 256].map((g) => <option key={g} value={g}>{g}</option>)}</select></label>}
        <label><span>KV cache simulation</span><select className="field select" value={f.kv_bits} onChange={(e) => set("kv_bits", Number(e.target.value))}>
          <option value={0}>off (fp16)</option>{[8, 6, 4, 3, 2].map((b) => <option key={b} value={b}>{b}-bit keys/values</option>)}</select>
          <em>Quantizes key/value projections per token before rotary embeddings: an approximation of a quantized KV cache.</em></label>
        <label><span>Text chunks (512 tokens each) · name</span><div className="ft-row">
          <input className="field" type="number" value={f.chunks} onChange={(e) => set("chunks", Number(e.target.value))} />
          <input className="field" placeholder="optional label" value={f.label} onChange={(e) => set("label", e.target.value)} /></div></label>
        <label className="ws-check"><input type="checkbox" checked={f.include_head} onChange={(e) => set("include_head", e.target.checked)} />
          <span><b>Also quantize the output head</b><em>Usually kept at high precision; it's often the most sensitive matrix.</em></span></label>
      </div>
      {f.method === "custom" && <label className="qs-code"><span className="row-sub">Your method: <code>quantize(w, name)</code> gets each decoder weight matrix (fp16, on the GPU) and its name, and returns the reconstructed weight. Add a third parameter, <code>quantize(w, name, h)</code>, to also get <code>h</code> = mean xᵀx of the layer's inputs on calibration text (what GPTQ and AWQ use), so the output error is tr(ΔW h ΔWᵀ). It runs on this machine with full Python access.</span>
        <textarea className="field" spellCheck={false} rows={12} value={f.code} onChange={(e) => set("code", e.target.value)} style={{ fontFamily: "var(--mono)", fontSize: 12, width: "100%" }} /></label>}
      <div className="xc-btns"><button className="btn" disabled={!f.checkpoint} onClick={start}><FlaskRound /> Simulate</button></div>
      {err && <div className="error"><AlertCircle /> {err}</div>}
      {shown && shown.status !== "done" && <JobLine j={shown} />}
      {done.length > 0 && (
        <div className="table-wrap"><table className="data">
          <thead><tr><th>Method</th><th>Bits / weight</th><th>KL divergence</th><th>KLD p99</th><th>Same top token</th><th>Perplexity</th><th title="||W − Q(W)|| / ||W||">Weight error</th>
            <th title="How much each layer's outputs change on real inputs: tr(ΔW H ΔWᵀ) / tr(W H Wᵀ). The error that matters.">Output error</th><th /></tr></thead>
          <tbody>{done.map((j) => { const r = j.results as unknown as QsResult; return (
            <tr key={j.id} className={shown?.id === j.id ? "sel" : ""} onClick={() => setSel(j.id)} style={{ cursor: "pointer" }}>
              <td>{qsName(j.params)}</td><td>{r.bits_per_weight ?? "–"}</td><td>{r.kld.toFixed(4)} <span className="muted">± {r.kld_sd.toFixed(4)}</span></td>
              <td>{r.kld_p99.toFixed(3)}</td><td>{r.same_top.toFixed(1)}%</td><td>{r.ppl.toFixed(3)} <span className="muted">(fp16 {r.ppl_base.toFixed(3)})</span></td>
              <td>{r.weight_rel_error != null ? `${(r.weight_rel_error * 100).toFixed(2)}%` : "–"}</td>
              <td>{r.output_rel_error ? `${(r.output_rel_error * 100).toFixed(2)}%` : "–"}</td>
              <td onClick={(e) => e.stopPropagation()}>{j.params.method !== "none" && <button className="btn small ghost" title={`Build a real ${xType} GGUF with this method and profile it against the original`} onClick={() => doExport(j.id)}>Export &amp; profile</button>}</td></tr>); })}</tbody>
        </table></div>
      )}
      {done.length > 0 && (
        <div className="row-sub qs-export">Export &amp; profile builds a real GGUF with the method (as <select className="field select small-select" value={xType} onChange={(e) => setXType(e.target.value)}>
          {["Q4_0", "Q4_1", "Q5_0", "Q8_0", "F16"].map((t) => <option key={t}>{t}</option>)}</select>), converts the original checkpoint to an F16 GGUF as the reference, and queues both in an experiment “… · playground”: real speed, memory, MMLU-Pro and KL divergence. Use the Q4_0 grid + Q4_0 for a near-exact export; other combinations re-quantize, which adds some error.</div>
      )}
      {exports.filter((j) => j.params.checkpoint === f.checkpoint).slice(0, 4).map((j) => (
        <div key={j.id} className={`ws-job ${j.status}`}>
          {j.status === "running" || j.status === "queued" ? <Loader2 className="spin" size={15} /> : j.status === "done" ? null : <AlertCircle size={15} />}
          <div><b>Export · {qsName(j.params)} → {String(j.params.type)}</b>
            <span>{j.status === "running" ? `${j.step ?? "starting"}…` : j.status}{j.error ? `: ${j.error}` : ""}
              {j.status === "done" && <> · {String(j.results.gguf)} ({((Number(j.results.size_bytes) || 0) / 2 ** 30).toFixed(2)} GB){j.results.exact ? " · same grid as simulated" : ""}
                {j.results.experiment_id ? <> · <a href={`#/experiments?exp=${j.results.experiment_id}`}>profiling in Experiments</a></> : null}</>}</span></div>
          {(j.status === "running" || j.status === "queued") && <button className="btn small ghost" onClick={() => api.torchCancel(j.id)}><Square /> Stop</button>}
        </div>
      ))}
      {sr && sr.per_layer.length > 0 && (
        <div>
          <div className="mini-title">{useOut ? "Output error" : "Relative weight error"} per layer · {qsName(shown!.params)} <span className="muted">({useOut ? "tr(ΔW H ΔWᵀ) / tr(W H Wᵀ) on calibration text" : "||W − Q(W)|| / ||W||"}; {Object.entries((useOut ? sr.per_kind_out : sr.per_kind) ?? {}).map(([k, v]) => `${k} ${v != null ? (v * 100).toFixed(1) : "–"}%`).join(" · ")})</span></div>
          <div className="qs-bars">{sr.per_layer.filter((l) => l.layer != null).map((l) => (
            <div key={l.layer} title={`layer ${l.layer}: ${(errOf(l) * 100).toFixed(2)}%`} style={{ height: `${(errOf(l) / maxErr) * 100}%` }} />))}</div>
          <div className="row-sub">{sr.tokens} tokens of held-out text (WikiText-2 test){sr.calibration ? ` · calibrated on ${sr.calibration.tokens.toLocaleString()} tokens of WikiText-2 train` : ""} · {sr.seconds} s · {sr.note}</div>
        </div>
      )}
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

/** The PyTorch side. Inside a project (`embedded`), it opens on the given tool and leaves the title to the project. */
export default function InternalsPage({ initialTab, embedded }: { initialTab?: string; embedded?: boolean } = {}) {
  const [status, setStatus] = useState<TorchStatus | null>(null);
  const [tab, setTab] = useState(initialTab ?? "act");
  useEffect(() => { if (initialTab) setTab(initialTab); }, [initialTab]);
  const load = () => api.torchStatus().then(setStatus).catch(() => {});
  useEffect(() => { load(); }, []);
  return (
    <div className={embedded ? "embedded" : "page"}>
      <div className={embedded ? "" : "page-inner wide"}>
        {!embedded && <>
          <h1 className="page-title">Internals</h1>
          <p className="page-sub">The PyTorch side of the lab: look inside a model's layers, fine-tune it (LoRA or every weight) or pretrain one, then measure the result like any other weights.</p>
        </>}
        {status && !status.installed && <div className="group pad"><b>The PyTorch environment isn't installed.</b> <span className="row-sub">On the box: bash scripts/setup-ml.sh (a few GB).</span></div>}
        {status?.installed && <div className="row-sub ft-env"><Brain size={14} /> PyTorch {status.torch} · CUDA {status.cuda ? "on" : "off"} · transformers {status.transformers} · peft {status.peft}</div>}
        {!embedded && <Tabs value={tab} onChange={setTab} tabs={[{ id: "act", label: "Activations" }, { id: "ft", label: "Fine-tune (LoRA)" }, { id: "train", label: "Full training" }, { id: "play", label: "Technique playground" }, { id: "ckpt", label: "Checkpoints" }]} />}
        <div className="group pad">
          {tab === "act" && <Activations status={status} />}
          {tab === "ft" && <FineTune status={status} />}
          {tab === "train" && <FullTraining status={status} />}
          {tab === "play" && <Playground status={status} />}
          {tab === "ckpt" && <Checkpoints status={status} reload={load} />}
        </div>
      </div>
    </div>
  );
}
