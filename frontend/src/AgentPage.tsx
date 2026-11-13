import { useEffect, useRef, useState } from "react";
import { AlertCircle, ArrowUp, Bot, Check, ChevronRight, Cloud, Loader2, Lock, Plus, Square, Trash2, Wrench, X } from "lucide-react";
import { AgentItem, AgentSession, AgentStatus, api, streamAgent } from "./api";
import { Markdown } from "./Chat";

/** The research agent: a chat that can use the whole lab (tools + Python), with every step visible and risky ones approved. */

const SUGGESTIONS = [
  "Summarize my experiments: what's the best speed / quality trade-off so far, and is it significant?",
  "Find which layers of Qwen3.5-0.8B are most sensitive to quantization, then propose a mixed-precision recipe.",
  "Invent a 3-bit quantization method that beats RTN at the same bits per weight, and test it in the playground.",
  "How fast would my models run on a mini PC without a GPU? Use the device presets.",
];

const TOOL_LABEL: Record<string, string> = {
  get_status: "Looked at the device", get_experiment: "Read an experiment", create_experiment: "Created an experiment", queue_run: "Queued a run",
  queue_sweep: "Queued a sweep", cancel_run: "Cancelled a run", set_experiment_tests: "Changed an experiment's tests", list_tensor_groups: "Listed tensor groups",
  build_weight_variant: "Built a weight variant", start_layer_scan: "Started a layer scan", get_layer_scans: "Read layer scans", build_engine: "Build an engine",
  get_jobs: "Checked background jobs", simulate_quantization: "Simulate a quantization method", get_quantsim_results: "Read playground results", run_python: "Run Python",
};

function ToolCard({ it, onDecide }: { it: AgentItem & { role: "tool" }; onDecide: (allow: boolean, always?: boolean) => void }) {
  const [open, setOpen] = useState(it.status === "approval");
  const code = it.name === "run_python" ? String(it.input.code ?? "") : it.name === "simulate_quantization" && it.input.code ? String(it.input.code) : null;
  const args = Object.fromEntries(Object.entries(it.input).filter(([k]) => !(code && k === "code")));
  return (
    <div className={`ag-tool ${it.status}`}>
      <button className="ag-tool-head" onClick={() => setOpen(!open)}>
        {it.status === "running" ? <Loader2 className="spin" size={14} /> : it.status === "approval" ? <AlertCircle size={14} /> : it.status === "done" ? <Check size={14} /> : <X size={14} />}
        <b>{TOOL_LABEL[it.name] ?? it.name}</b>
        <span className="muted">{Object.keys(args).length ? JSON.stringify(args).slice(0, 90) : ""}</span>
        <ChevronRight size={14} className={open ? "rot" : ""} />
      </button>
      {open && (
        <div className="ag-tool-body">
          {code && <pre className="ag-code">{code}</pre>}
          {Object.keys(args).length > 0 && <pre className="ag-args">{JSON.stringify(args, null, 1)}</pre>}
          {it.result && <pre className="ag-result">{it.result}</pre>}
        </div>
      )}
      {it.status === "approval" && (
        <div className="ag-approve">
          <span>{it.name === "run_python" ? "The agent wants to run this code on this device." : it.name === "build_engine" ? "Building an engine downloads and runs that repository's code." : "This runs the agent's own code on this device."}</span>
          <button className="btn small" onClick={() => onDecide(true)}>Allow</button>
          <button className="btn small ghost" onClick={() => onDecide(true, true)}>Allow for this chat</button>
          <button className="btn small ghost" onClick={() => onDecide(false)}>Deny</button>
        </div>
      )}
    </div>
  );
}

export default function AgentPage() {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [sessions, setSessions] = useState<{ id: number; title: string; provider: string; updated_at: number }[]>([]);
  const [sid, setSid] = useState<number | null>(null);
  const [sess, setSess] = useState<AgentSession | null>(null);
  const [items, setItems] = useState<AgentItem[]>([]);
  const [thinking, setThinking] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [input, setInput] = useState("");
  const [provider, setProvider] = useState<"local" | "claude">("local");
  const [err, setErr] = useState<string | null>(null);
  const end = useRef<HTMLDivElement>(null);

  const loadList = () => api.agentSessions().then(setSessions).catch(() => {});
  useEffect(() => { api.agentStatus().then((s) => { setStatus(s); if (s.claude) setProvider("claude"); }).catch(() => {}); loadList(); }, []);
  const loadSession = (id: number) => api.agentSession(id).then((s) => { setSess(s); setItems(s.display); }).catch(() => {});
  useEffect(() => { if (sid != null) loadSession(sid); else { setSess(null); setItems([]); } }, [sid]);
  // Re-attached to a chat the agent is still working on (e.g. after a reload): follow it.
  useEffect(() => {
    if (!sess?.running || streaming || sid == null) return;
    const t = setInterval(() => loadSession(sid), 2000);
    return () => clearInterval(t);
  }, [sess?.running, streaming, sid]);
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [items, thinking]);

  const send = async (text: string) => {
    if (!text.trim() || streaming) return;
    setErr(null);
    let id = sid;
    try {
      if (id == null) { id = (await api.agentNew(provider)).id; setSid(id); loadList(); }
    } catch (e) { setErr((e as Error).message); return; }
    setInput("");
    setItems((xs) => [...xs, { role: "user", text }]);
    setStreaming(true);
    try {
      await streamAgent(id!, text, (e) => {
        if (e.type === "thinking") setThinking(e.text || "");
        else if (e.type === "text") {
          setThinking(null);
          setItems((xs) => { const last = xs[xs.length - 1]; return last?.role === "assistant" ? [...xs.slice(0, -1), { ...last, text: last.text + e.text }] : [...xs, { role: "assistant", text: e.text }]; });
        } else if (e.type === "approval" || e.type === "tool_call") {
          setThinking(null);
          setItems((xs) => {
            const i = xs.findIndex((x) => x.role === "tool" && x.id === e.id);
            const it: AgentItem = { role: "tool", id: e.id, name: e.name, input: e.input, status: e.type === "approval" ? "approval" : "running" };
            return i >= 0 ? xs.map((x, j) => (j === i ? it : x)) : [...xs, it];
          });
        } else if (e.type === "tool_result") {
          setItems((xs) => xs.map((x) => (x.role === "tool" && x.id === e.id ? { ...x, status: e.status, result: e.content } : x)));
        } else if (e.type === "error") setItems((xs) => [...xs, { role: "error", text: e.message }]);
      });
    } catch (e) { setErr((e as Error).message); }
    setStreaming(false); setThinking(null);
    loadSession(id!); loadList();
  };
  const decide = async (it: AgentItem & { role: "tool" }, allow: boolean, always = false) => {
    await api.agentApprove(sid!, it.id, it.name, allow, always);
    setItems((xs) => xs.map((x) => (x.role === "tool" && x.id === it.id ? { ...x, status: allow ? "running" : "denied" } : x)));
  };
  const busy = streaming || !!sess?.running;
  const cloud = (sess?.provider ?? provider) === "claude";

  return (
    <div className="page ag-page">
      <div className="ag-wrap">
        <aside className="ag-side">
          <button className="btn small" onClick={() => setSid(null)}><Plus size={14} /> New task</button>
          {sessions.map((s) => (
            <div key={s.id} className={`ag-sess ${sid === s.id ? "on" : ""}`} onClick={() => setSid(s.id)}>
              {s.provider === "claude" ? <Cloud size={13} /> : <Lock size={13} />}<span>{s.title}</span>
              <button className="icon-btn" title="Delete" onClick={(e) => { e.stopPropagation(); api.agentDelete(s.id).then(() => { if (sid === s.id) setSid(null); loadList(); }); }}><Trash2 size={13} /></button>
            </div>
          ))}
        </aside>
        <main className="ag-main">
          {sid == null && items.length === 0 ? (
            <div className="ag-empty">
              <div className="ag-hero"><Bot size={30} /><h1>Research agent</h1>
                <p>Give it a research task. It uses the whole lab (experiments, sweeps, weight variants, layer scans, engines, device presets, the technique playground) and writes and runs Python, and you watch every step. Code only runs after you allow it.</p></div>
              <div className="segmented ag-brain">
                <button className={provider === "local" ? "on" : ""} onClick={() => setProvider("local")}><Lock size={13} /> Local model · private</button>
                <button className={provider === "claude" ? "on" : ""} disabled={!status?.claude} onClick={() => setProvider("claude")}
                  title={status?.claude ? "" : "Add an Anthropic API key in Settings → AI agent"}><Cloud size={13} /> Claude · {status?.claude_model ?? "cloud"}</button>
              </div>
              <div className="row-sub ag-brain-note">{provider === "claude"
                ? "Claude is much better at long research tasks. Your messages and the tool results (experiment data, code output) are sent to Anthropic with your API key."
                : <>Runs on this device's chat model: nothing leaves the device, but small models make more mistakes on long tasks.{!status?.claude && <> For Claude, add an API key in <a href="#/settings">Settings → AI agent</a>.</>}</>}</div>
              <div className="ag-suggest">{SUGGESTIONS.map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}</div>
            </div>
          ) : (
            <div className="ag-log">
              {cloud && <div className="ag-banner"><Cloud size={13} /> Claude: messages and tool results go to Anthropic.</div>}
              {items.map((it, i) => it.role === "user" ? <div key={i} className="ag-user">{it.text}</div>
                : it.role === "assistant" ? <div key={i} className="ag-asst"><Markdown text={it.text} /></div>
                : it.role === "tool" ? <ToolCard key={it.id} it={it} onDecide={(a, al) => decide(it, a, al)} />
                : <div key={i} className="error"><AlertCircle /> {it.text}</div>)}
              {busy && <div className="ag-thinking"><Loader2 className="spin" size={14} /> {thinking ? <span title={thinking}>Thinking: {thinking.slice(-140)}</span> : "Working…"}</div>}
              <div ref={end} />
            </div>
          )}
          {err && <div className="error"><AlertCircle /> {err}</div>}
          <form className="ag-input" onSubmit={(e) => { e.preventDefault(); send(input); }}>
            <textarea value={input} placeholder={sid == null ? "Describe a research task…" : "Reply, or give the next task…"} rows={2}
              onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }} />
            {busy ? <button type="button" className="btn" onClick={() => sid != null && api.agentStop(sid)}><Square size={14} /> Stop</button>
              : <button className="btn" disabled={!input.trim()}><ArrowUp size={16} /></button>}
          </form>
          <div className="row-sub ag-foot"><Wrench size={12} /> It can queue runs and builds that use this machine for minutes or hours, and it never deletes anything. Stop it any time.</div>
        </main>
      </div>
    </div>
  );
}
