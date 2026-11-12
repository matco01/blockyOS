import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import BlockField from "./BlockField";
import Logo from "./Logo";
import ConfidenceView from "./Confidence";
import { AlertCircle, ArrowUp, Brain, Check, ChevronRight, Copy, Lightbulb, ListChecks, PenLine, ShieldCheck, Sparkles, Square } from "lucide-react";
import { api, Message, MessageStats, Sampling, streamChat } from "./api";
import { useSetting } from "./settings";
import { live } from "./live";
import { speedVerdict, useMode } from "./ui";

type Props = {
  initialId: number | null;
  onCreated: (id: number) => void;
  onUpdated: () => void;
  onOpenMemory: () => void;
  sampling: Sampling;
  systemPrompt: string;
  thinkBudget: number;
  confidence: boolean;
  contextSize: number | null;
  onStats: (s: MessageStats) => void;
};

const fmtMs = (v: number | null | undefined) => (v == null ? null : v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${v}ms`);

/** One-line performance summary under each answer. */
function MetaLine({ m, ctx }: { m: Message; ctx: number | null }) {
  const mode = useMode();
  const s = m.stats;
  if (!s) return m.tokens_per_sec != null ? <div className="meta">{m.tokens} tokens · {m.tokens_per_sec.toFixed(1)} tok/s</div> : null;
  if (mode === "simple") {
    const v = speedVerdict(s.predicted_per_second);
    return (
      <div className="meta">
        {(s.predicted_per_second ?? 0).toFixed(0)} tok/s · {v.label}
        {s.ttft_ms != null && ` · first word after ${fmtMs(s.ttft_ms)}`}
      </div>
    );
  }
  const used = (s.prompt_n ?? 0) + (s.cache_n ?? 0) + (s.predicted_n ?? 0);
  const parts = [
    s.ttft_ms != null && `TTFT ${fmtMs(s.ttft_ms)}`,
    s.prompt_n != null && `prompt ${s.prompt_n}${s.cache_n ? ` (+${s.cache_n} cached)` : ""} tok @ ${Math.round(s.prompt_per_second ?? 0)} tok/s`,
    s.predicted_n != null && `gen ${s.predicted_n} tok @ ${(s.predicted_per_second ?? 0).toFixed(1)} tok/s`,
    ctx && `ctx ${((used / ctx) * 100).toFixed(1)}%`,
    s.sampling && Object.keys(s.sampling).length > 0 && Object.entries(s.sampling).map(([k, v]) => `${k}=${v}`).join(" "),
  ].filter(Boolean);
  return <div className="meta mono">{parts.join(" · ")}</div>;
}

const SUGGESTIONS = [
  { icon: ShieldCheck, color: "#7c5cff", title: "How it works", text: "How do you work without the cloud?" },
  { icon: PenLine, color: "#ff5c9d", title: "Write something", text: "Write a short poem about privacy" },
  { icon: ListChecks, color: "#ffa94d", title: "Get organized", text: "Help me plan my week" },
  { icon: Lightbulb, color: "#22d3ee", title: "Learn a thing", text: "Explain quantization like I'm five" },
];

function greeting(): [string, string] {
  const h = new Date().getHours();
  if (h < 5) return ["Up", "late?"];
  if (h < 12) return ["Good", "morning"];
  if (h < 18) return ["Good", "afternoon"];
  return ["Good", "evening"];
}

/** Code block with a language label and a copy button. */
function CodeBlock({ children }: { children?: React.ReactNode }) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  const child = Array.isArray(children) ? children[0] : children;
  const lang = /language-(\w+)/.exec((child as React.ReactElement<{ className?: string }>)?.props?.className ?? "")?.[1];

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(ref.current?.innerText ?? "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard blocked (e.g. non-HTTPS on another device) */ }
  };

  return (
    <div className="code-block">
      <div className="code-head">
        <span>{lang ?? "code"}</span>
        <button onClick={copy}>{copied ? <><Check /> Copied</> : <><Copy /> Copy</>}</button>
      </div>
      <pre ref={ref}>{children}</pre>
    </div>
  );
}

const MD_COMPONENTS = { pre: CodeBlock };

/** Rendered answer text (Markdown, math, code blocks with copy). Shared with the Arena. */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={MD_COMPONENTS}>{text}</ReactMarkdown>
    </div>
  );
}

/** The model's reasoning: live "Thinking… 12s" while it works, then a collapsed "Thought for 12s". */
function Thinking({ text, live, secs, budgetHit }: { text: string; live: boolean; secs?: number; budgetHit?: boolean }) {
  const start = useRef(Date.now());
  const [now, setNow] = useState(Date.now());
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, [live]);

  useEffect(() => {
    if (live && bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [text, live]);

  const label = (live
    ? `Thinking… ${Math.floor((now - start.current) / 1000)}s`
    : secs != null ? `Thought for ${secs}s` : "Thought process") + (budgetHit ? " · budget reached, answered early" : "");

  return (
    <details className={`reasoning ${live ? "live" : ""}`} open={live}>
      <summary>
        <ChevronRight className="chev" />
        <span className="label">{label}</span>
      </summary>
      <div ref={bodyRef} className="reasoning-body">{text}</div>
    </details>
  );
}

export default function Chat({ initialId, onCreated, onUpdated, onOpenMemory, sampling, systemPrompt, thinkBudget, confidence, contextSize, onStats }: Props) {
  const [confView, setConfView] = useState<Set<number>>(new Set()); // answers shown as confidence heatmap
  const [convId, setConvId] = useState<number | null>(initialId);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(initialId != null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [think, setThink] = useSetting("thinking", true);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Runs once per opened chat (App remounts this component when you switch chats).
  // Must NOT depend on initialId: App updates it when a new chat gets its ID,
  // and re-running this would abort the answer that's currently streaming.
  useEffect(() => {
    if (initialId != null) {
      api.messages(initialId)
        .then((ms) => {
          setMessages(ms);
          const lastStats = [...ms].reverse().find((m) => m.stats)?.stats;
          if (lastStats) onStats(lastStats);
        })
        .catch(() => setError("Couldn't load this chat."))
        .finally(() => setLoading(false));
    }
    inputRef.current?.focus();
    return () => abortRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  const send = async (text: string) => {
    const message = text.trim();
    if (!message || busy) return;
    setInput("");
    setError(null);
    setBusy(true);
    setMessages((ms) => [...ms, { role: "user", content: message }, { role: "assistant", content: "" }]);
    // Updates target this answer by position, so late events (e.g. memory) can't
    // land on a newer message if the user has already sent another one.
    const at = messages.length + 1;
    const update = (fn: (m: Message) => Message) =>
      setMessages((ms) => ms.map((m, i) => (i === at ? fn(m) : m)));

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    live.start();
    let gotDone = false;
    let gotError = false;
    let thinkStart: number | null = null;
    let thinkTimed = false;
    try {
      const extra = {
        ...(Object.keys(sampling).length ? { sampling } : {}),
        ...(systemPrompt.trim() ? { system_prompt: systemPrompt } : {}),
      };
      if (confidence) setConfView((s) => new Set(s).add(at)); // show the heatmap right away
      await streamChat({ message, conversation_id: convId, think, think_budget: thinkBudget, confidence, ...extra }, (e) => {
        if (e.type === "meta" && convId == null) {
          setConvId(e.conversation_id);
          onCreated(e.conversation_id);
        } else if (e.type === "token") {
          if (thinkStart != null && !thinkTimed) {
            thinkTimed = true;
            const secs = Math.max(1, Math.round((Date.now() - thinkStart) / 1000));
            update((m) => ({ ...m, thought_secs: secs }));
          }
          live.token("c");
          update((m) => ({ ...m, content: m.content + e.text }));
        } else if (e.type === "reasoning") {
          live.token("r");
          thinkStart ??= Date.now();
          update((m) => ({ ...m, reasoning: (m.reasoning ?? "") + e.text }));
        } else if (e.type === "done") {
          gotDone = true;
          live.stop();
          setBusy(false); // answer is complete; memory may still be updating in the background
          update((m) => ({ ...m, tokens: e.tokens, tokens_per_sec: e.tokens_per_sec, stats: e.stats }));
          if (e.stats) onStats(e.stats);
        } else if (e.type === "probs") {
          update((m) => ({ ...m, probs: [...(m.probs ?? []), ...e.items] }));
        } else if (e.type === "budget") {
          update((m) => ({ ...m, budget_hit: true }));
        } else if (e.type === "memory") {
          update((m) => ({ ...m, remembered: e.added.map((x) => x.content) }));
        } else if (e.type === "error") {
          gotError = true;
          setError(e.message);
        }
      }, ctrl.signal);
      if (!gotDone && !gotError) setError("The answer was cut off (connection lost). Try again.");
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        if (!gotDone) update((m) => ({ ...m, content: m.content || "_Stopped._" }));
      } else {
        setError(`Couldn't reach Blocky: ${(err as Error).message}`);
      }
    } finally {
      live.stop();
      setBusy(false);
      abortRef.current = null;
      onUpdated();
      inputRef.current?.focus();
    }
  };

  const composer = (
    <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input); }}>
      <textarea
        ref={inputRef}
        value={input}
        rows={1}
        placeholder="Ask anything…"
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); }
        }}
      />
      <div className="composer-actions">
        <button
          type="button"
          className={`chip-toggle ${think ? "on" : ""}`}
          onClick={() => setThink(!think)}
          title={think ? "Reasons before answering: slower, smarter" : "Answers immediately: faster"}
        >
          <Sparkles /> Think
        </button>
        {busy ? (
          <button type="button" className="send stop" onClick={() => abortRef.current?.abort()} aria-label="Stop">
            <Square fill="currentColor" />
          </button>
        ) : (
          <button type="submit" className="send" disabled={!input.trim()} aria-label="Send">
            <ArrowUp strokeWidth={2.5} />
          </button>
        )}
      </div>
    </form>
  );

  if (messages.length === 0 && !loading) {
    return (
      <div className="chat">
        <div className="welcome">
          <BlockField />
          <Logo size={68} animated />
          <h1>{greeting()[0]} <span className="grad-text">{greeting()[1]}</span></h1>
          <p>Private AI, running entirely on this computer.</p>
          {composer}
          <div className="suggestions">
            {SUGGESTIONS.map(({ icon: Icon, color, title, text }) => (
              <button key={text} className="suggestion" style={{ "--c": color } as React.CSSProperties} onClick={() => send(text)}>
                <span className="s-icon"><Icon /></span>
                <span>
                  <span className="s-title">{title}</span>
                  <span className="s-sub">{text}</span>
                </span>
              </button>
            ))}
          </div>
          {error && <div className="error" style={{ marginTop: 16 }}><AlertCircle /> {error}</div>}
        </div>
      </div>
    );
  }

  return (
    <div className="chat">
      <div className="messages">
        <div className="thread">
          {messages.map((m, i) => (
            <div key={m.id ?? `tmp-${i}`} className={`msg ${m.role}`}>
              {m.role === "user" ? (
                <div className="bubble">{m.content}</div>
              ) : (
                <>
                  {m.reasoning && (
                    <Thinking text={m.reasoning} live={busy && i === messages.length - 1 && !m.content} secs={m.thought_secs}
                      budgetHit={m.budget_hit || m.stats?.budget_hit} />
                  )}
                  {m.content ? (
                    m.probs?.length && confView.has(i) ? <ConfidenceView probs={m.probs} /> : <Markdown text={m.content} />
                  ) : busy && i === messages.length - 1 && !m.reasoning ? (
                    <div className="typing"><span /><span /><span /></div>
                  ) : null}
                  {m.probs?.length ? (
                    <button className="conf-toggle" onClick={() => setConfView((s) => { const n = new Set(s); if (n.has(i)) n.delete(i); else n.add(i); return n; })}>
                      {confView.has(i) ? "Show normal text" : "Show confidence"}
                    </button>
                  ) : null}
                  <MetaLine m={m} ctx={contextSize} />
                  {m.remembered && m.remembered.length > 0 && (
                    <button className="remembered" onClick={onOpenMemory} title="Manage memory">
                      <Brain /> <span>Remembered: {m.remembered.join(" · ")}</span>
                    </button>
                  )}
                </>
              )}
            </div>
          ))}
          {error && <div className="error"><AlertCircle /> {error}</div>}
          <div ref={endRef} />
        </div>
      </div>
      <div className="composer-wrap">
        {composer}
        <div className="hint">Runs locally · nothing leaves this device</div>
      </div>
    </div>
  );
}
