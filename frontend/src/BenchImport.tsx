import { useState } from "react";
import { AlertCircle, Check, Copy, Loader2, Search, Terminal } from "lucide-react";
import { api, HfPreview } from "./api";

/** Import any Hugging Face dataset as a benchmark by mapping its columns. */
export function ImportBenchmark({ onDone, onClose }: { onDone: () => void; onClose: () => void }) {
  const [repo, setRepo] = useState("");
  const [prev, setPrev] = useState<HfPreview | null>(null);
  const [state, setState] = useState<"idle" | "loading" | string>("idle");
  const [form, setForm] = useState({ name: "", split: "", question: "", answer: "", options: "", kind: "auto", limit: 1000 });

  const look = async () => {
    setState("loading"); setPrev(null);
    try {
      const p = await api.hfDataset(repo.trim());
      setPrev(p);
      const guess = (names: string[]) => p.columns.find((c) => names.includes(c.toLowerCase())) ?? "";
      setForm((f) => ({
        ...f, name: repo.split("/").pop() ?? "", split: `${p.splits[0].config}/${p.splits[0].split}`,
        question: guess(["question", "problem", "prompt", "input", "query", "text"]),
        answer: guess(["answer", "answerkey", "label", "solution", "output", "target"]),
        options: guess(["options", "choices"]),
      }));
      setState("idle");
    } catch (e) { setState((e as Error).message); }
  };

  const save = async () => {
    const [config, split] = form.split.split("/");
    try {
      await api.benchmarkImport({ repo: repo.trim(), config, split, name: form.name, question: form.question, answer: form.answer,
        options: form.options || null, kind: form.kind, limit: form.limit });
      onDone();
    } catch (e) { setState((e as Error).message); }
  };

  const col = (key: "question" | "answer" | "options", label: string, optional = false) => (
    <label><span>{label}</span>
      <select className="field select" value={form[key]} onChange={(e) => setForm({ ...form, [key]: e.target.value })}>
        {optional && <option value="">(none)</option>}
        {prev?.columns.map((c) => <option key={c}>{c}</option>)}
      </select>
    </label>
  );

  return (
    <div className="group pad custom-editor">
      <div className="row-sub">Any public dataset on Hugging Face with a question column and an answer column. Blocky checks answers as numbers, text, multiple choice or LaTeX math.</div>
      <form className="import-row" onSubmit={(e) => { e.preventDefault(); look(); }}>
        <input className="field" placeholder="owner/dataset, e.g. openai/gsm8k" value={repo} onChange={(e) => setRepo(e.target.value)} />
        <button className="btn" disabled={!repo.trim() || state === "loading"}>{state === "loading" ? <Loader2 className="spin" /> : <Search />} Look up</button>
      </form>
      {state !== "idle" && state !== "loading" && <div className="error"><AlertCircle /> {state}</div>}
      {prev && (
        <>
          <div className="form-grid">
            <label><span>Name</span><input className="field" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></label>
            <label><span>Split</span>
              <select className="field select" value={form.split} onChange={(e) => setForm({ ...form, split: e.target.value })}>
                {prev.splits.map((s) => <option key={`${s.config}/${s.split}`}>{`${s.config}/${s.split}`}</option>)}
              </select></label>
            <label><span>Answer type</span>
              <select className="field select" value={form.kind} onChange={(e) => setForm({ ...form, kind: e.target.value })}>
                <option value="auto">Detect automatically</option><option value="number">Number</option><option value="text">Text</option>
                <option value="choice">Multiple choice</option><option value="math">LaTeX math</option>
              </select></label>
            {col("question", "Question column")}
            {col("answer", "Answer column")}
            {col("options", "Options column (multiple choice)", true)}
            <label><span>Max questions</span><input className="field mono" value={form.limit} onChange={(e) => setForm({ ...form, limit: Number(e.target.value.replace(/\D/g, "")) || 100 })} /></label>
          </div>
          {prev.sample[0] && (
            <div className="sample">
              <div className="row-sub">First row, so you can check the mapping:</div>
              <div><span>Question</span><b>{prev.sample[0][form.question] ?? "–"}</b></div>
              <div><span>Answer</span><b>{prev.sample[0][form.answer] ?? "–"}</b></div>
              {form.options && <div><span>Options</span><b>{prev.sample[0][form.options]}</b></div>}
            </div>
          )}
          <div className="custom-actions">
            <span className="row-sub">Gated (login-only) datasets aren't supported.</span>
            <button className="btn ghost" onClick={onClose}>Cancel</button>
            <button className="btn" disabled={!form.question || !form.answer} onClick={save}><Check /> Import</button>
          </div>
        </>
      )}
      {!prev && <div className="custom-actions"><span /><button className="btn ghost" onClick={onClose}>Cancel</button></div>}
    </div>
  );
}

function CopyLine({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <div className="cmd">
      <code>{text}</code>
      <button className="icon-btn" title="Copy" onClick={async () => { try { await navigator.clipboard.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); } catch { /* blocked */ } }}>
        {done ? <Check /> : <Copy />}
      </button>
    </div>
  );
}

/** For benchmarks Blocky doesn't run itself (SWE-bench, lm-eval tasks…): point the standard tools at Blocky. */
export function OutsideTools() {
  const base = `${window.location.origin}/v1`;
  return (
    <div className="group pad outside">
      <div className="outside-head"><Terminal /> <b>Use the research community's own tools</b></div>
      <div className="row-sub">
        Blocky speaks the OpenAI API at <code>{base}</code>. Any tool that can talk to OpenAI can test your local model, including
        benchmarks Blocky can't run itself, like SWE-bench (needs Docker and much bigger models to score well).
        Chat pauses while an outside tool is running a test.
      </div>
      <div className="tool">
        <b>lm-evaluation-harness</b> <span className="muted">· hundreds of standard tasks (MMLU, HellaSwag, IFEval…)</span>
        <CopyLine text={`lm_eval --model local-chat-completions --model_args model=blocky,base_url=${base}/chat/completions --tasks gsm8k --apply_chat_template`} />
      </div>
      <div className="tool">
        <b>EvalPlus</b> <span className="muted">· the official HumanEval+/MBPP+ runner</span>
        <CopyLine text={`evalplus.evaluate --model blocky --dataset humaneval --backend openai --base-url ${base} --greedy`} />
      </div>
      <div className="tool">
        <b>SWE-bench agents</b> <span className="muted">· (mini-)SWE-agent and similar: set these, then follow their docs</span>
        <CopyLine text={`OPENAI_BASE_URL=${base} OPENAI_API_KEY=local`} />
      </div>
      <div className="row-sub">Exact flags change between tool versions; check each tool's docs if a command fails. Any API key works; Blocky doesn't check it yet.</div>
    </div>
  );
}
