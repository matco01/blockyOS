import { useCallback, useEffect, useState } from "react";
import { Activity, Boxes, Brain, ClipboardCheck, FlaskConical, LayoutDashboard, PanelLeft, Swords, Plus, Search, Settings as SettingsIcon, SlidersHorizontal, Trash2 } from "lucide-react";
import { api, Conversation, MessageStats, Sampling, SearchHit, Status } from "./api";
import ArenaPage from "./ArenaPage";
import BenchPage from "./BenchPage";
import Chat from "./Chat";
import Dashboard from "./Dashboard";
import Inspector from "./Inspector";
import TestsPage from "./TestsPage";
import { Mode, ModeContext } from "./ui";
import ModelsPage from "./ModelsPage";
import MemoryPage from "./MemoryPage";
import SystemPage from "./SystemPage";
import SettingsPage from "./SettingsPage";
import Logo from "./Logo";
import { Accent, BLOCK_COLORS, Theme, useSetting } from "./settings";

export type Page = "chat" | "home" | "models" | "arena" | "tests" | "bench" | "memory" | "system" | "settings";

const PAGES = [
  { id: "home", label: "Overview", icon: LayoutDashboard },
  { id: "models", label: "Models", icon: Boxes },
  { id: "arena", label: "Arena", icon: Swords },
  { id: "tests", label: "Tests", icon: ClipboardCheck },
  { id: "bench", label: "Benchmarks", icon: FlaskConical },
  { id: "memory", label: "Memory", icon: Brain },
  { id: "system", label: "System", icon: Activity },
  { id: "settings", label: "Settings", icon: SettingsIcon },
] as const;

/** Renders a search snippet, highlighting the [matched] words. */
function Snippet({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\[[^\]]*\])/).map((part, i) =>
        part.startsWith("[") ? <mark key={i}>{part.slice(1, -1)}</mark> : part,
      )}
    </>
  );
}

/** URL hash <-> view: #/chat/5, #/memory, #/system, #/settings. Enables back button & bookmarks. */
function parseHash(): { page: Page; id: number | null } {
  const [, rawP, id] = window.location.hash.split("?")[0].split("/");
  const p = rawP ?? "";
  if (p === "home" || p === "models" || p === "arena" || p === "tests" || p === "bench" || p === "memory" || p === "system" || p === "settings") return { page: p, id: null };
  return { page: "chat", id: p === "chat" && id ? Number(id) || null : null };
}

export default function App() {
  const initial = parseHash();
  const [page, setPage] = useState<Page>(initial.page);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<number | null>(initial.id);
  const [status, setStatus] = useState<Status | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chatKey, setChatKey] = useState(0); // bumps only when the user switches chats
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [theme, setTheme] = useSetting<Theme>("theme", "system");

  const [accent, setAccent] = useSetting<Accent>("accent", "violet");
  const [sampling, setSampling] = useSetting<Sampling>("sampling", {});
  const [systemPrompt, setSystemPrompt] = useSetting("systemPrompt", "");
  const [inspectorOpen, setInspectorOpen] = useSetting("inspector", window.location.hash.includes("inspector=1"));
  const [lastStats, setLastStats] = useState<MessageStats | null>(null);
  const [mode, setMode] = useSetting<Mode>("mode", "simple");
  const [thinkBudget, setThinkBudget] = useSetting("thinkBudget", 4096);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") delete root.dataset.theme;
    else root.dataset.theme = theme;
    if (accent === "violet") delete root.dataset.accent;
    else root.dataset.accent = accent;
  }, [theme, accent]);

  const refresh = useCallback(() => {
    api.conversations().then(setConversations).catch(() => {});
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll status: fast while the Lab is busy or the model is loading, slow otherwise.
  const fastPoll = !!status?.busy || !status?.model_ready;
  useEffect(() => {
    const poll = () => api.status().then(setStatus).catch(() => setStatus(null));
    poll();
    const t = setInterval(poll, fastPoll ? 1500 : 5000);
    return () => clearInterval(t);
  }, [fastPoll]);

  // Search as you type (debounced).
  useEffect(() => {
    const q = query.trim();
    if (q.length < 3) { setHits(null); return; }
    const t = setTimeout(() => api.search(q).then(setHits).catch(() => setHits([])), 250);
    return () => clearTimeout(t);
  }, [query]);

  // Keep the URL in sync, and follow it when the user presses back/forward.
  useEffect(() => {
    const want = page === "chat" ? (activeId != null ? `#/chat/${activeId}` : "#/") : `#/${page}`;
    const current = window.location.hash.split("?")[0];
    if (current !== want && !(want === "#/" && current === "")) {
      // Pages with their own query (#/bench?compare=…, #/arena?match=…) manage it themselves.
      window.history.pushState(null, "", want);
    }
  }, [page, activeId]);

  useEffect(() => {
    const onPop = () => {
      const h = parseHash();
      setPage(h.page);
      setActiveId((cur) => {
        if (h.page === "chat" && h.id !== cur) setChatKey((k) => k + 1);
        return h.id;
      });
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const go = (p: Page) => { setPage(p); setSidebarOpen(false); };

  const open = (id: number | null) => {
    setActiveId(id);
    setChatKey((k) => k + 1);
    go("chat");
  };

  const remove = async (id: number) => {
    await api.remove(id);
    if (id === activeId) open(null);
    refresh();
  };

  const title =
    page === "chat"
      ? conversations.find((c) => c.id === activeId)?.title ?? ""
      : PAGES.find((p) => p.id === page)?.label ?? "";

  return (
    <ModeContext.Provider value={mode}>
      <div className="backdrop" />
      <div className={`app mode-${mode}`}>
        <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
          <div className="brand">
            <Logo size={24} />
            <span className="brand-name">Blocky</span>
          </div>

          <button className="side-btn" onClick={() => open(null)}>
            <Plus /> New chat
          </button>
          <div className="search">
            <Search />
            <input
              type="search"
              placeholder="Search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>

          <div className="section-label">{hits !== null ? "Results" : "Recents"}</div>
          <nav className="history">
            {hits !== null ? (
              hits.length === 0 ? (
                <div className="empty-hint">No matches</div>
              ) : (
                hits.map((h) => (
                  <div key={h.conversation_id} className="history-item hit" onClick={() => { setQuery(""); open(h.conversation_id); }}>
                    <span className="title">{h.title}</span>
                    <span className="snippet"><Snippet text={h.snippet} /></span>
                  </div>
                ))
              )
            ) : conversations.length === 0 ? (
              <div className="empty-hint">Your chats will appear here</div>
            ) : (
              conversations.map((c) => (
                <div
                  key={c.id}
                  className={`history-item ${c.id === activeId && page === "chat" ? "active" : ""}`}
                  onClick={() => open(c.id)}
                >
                  <span className="dot-block" style={{ "--c": BLOCK_COLORS[c.id % BLOCK_COLORS.length] } as React.CSSProperties} />
                  <span className="title">{c.title}</span>
                  <button
                    className="delete"
                    title="Delete chat"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm("Delete this conversation?")) remove(c.id);
                    }}
                  >
                    <Trash2 />
                  </button>
                </div>
              ))
            )}
          </nav>

          <div className="side-nav">
            {PAGES.map(({ id, label, icon: Icon }) => (
              <button key={id} className={`side-btn ${page === id ? "active" : ""}`} onClick={() => go(id)}>
                <Icon /> {label}
              </button>
            ))}
            <div className="mode-switch" title="Simple: plain language and the essentials. Pro: every number and setting.">
              <div className="segmented">
                <button className={mode === "simple" ? "on" : ""} onClick={() => setMode("simple")}>Simple</button>
                <button className={mode === "pro" ? "on" : ""} onClick={() => setMode("pro")}>Pro</button>
              </div>
            </div>
          </div>
        </aside>
        {sidebarOpen && <div className="scrim" onClick={() => setSidebarOpen(false)} />}

        <main className="main">
          <header className="topbar">
            <button className="icon-btn menu" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">
              <PanelLeft />
            </button>
            <span className="title">{title}</span>
            <button className={`status-pill ${status?.model_ready && !status.busy ? "ready" : ""}`} onClick={() => go("home")}
              title={status?.context_size ? `${status.context_size.toLocaleString()} token context · click to manage models` : "Manage models"}>
              <span className="dot" />
              {status?.busy === "benchmark" ? "Benchmarking…" : status?.busy === "switching" ? "Switching model…"
                : status?.busy === "arena" ? "Arena match…"
                : status?.busy === "eval" ? "Running tests…"
                : status?.model_ready ? status.model : "Starting model…"}
            </button>
            {page === "chat" && (
              <button className={`icon-btn ${inspectorOpen ? "on" : ""}`} title="Inspector: sampling & timings" onClick={() => setInspectorOpen(!inspectorOpen)}>
                <SlidersHorizontal />
              </button>
            )}
          </header>

          {page === "chat" && (
            <div className="chat-split">
              <Chat
                key={chatKey}
                initialId={activeId}
                onCreated={(id) => { setActiveId(id); refresh(); }}
                onUpdated={refresh}
                onOpenMemory={() => go("memory")}
                sampling={sampling}
                systemPrompt={systemPrompt}
                thinkBudget={thinkBudget}
                contextSize={status?.context_size ?? null}
                onStats={setLastStats}
              />
              {inspectorOpen && (
                <Inspector sampling={sampling} setSampling={setSampling} systemPrompt={systemPrompt} setSystemPrompt={setSystemPrompt}
                  thinkBudget={thinkBudget} setThinkBudget={setThinkBudget}
                  last={lastStats} ctx={status?.context_size ?? null} onClose={() => setInspectorOpen(false)} />
              )}
            </div>
          )}
          {page === "home" && <Dashboard status={status} go={go} />}
          {page === "models" && <ModelsPage status={status} />}
          {page === "arena" && <ArenaPage status={status} />}
          {page === "tests" && <TestsPage status={status} />}
          {page === "bench" && <BenchPage status={status} />}
          {page === "memory" && <MemoryPage />}
          {page === "system" && <SystemPage status={status} />}
          {page === "settings" && <SettingsPage status={status} theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent} />}
        </main>
      </div>
    </ModeContext.Provider>
  );
}
