import { useEffect, useState } from "react";
import { Brain, Plus, Sparkles, Trash2 } from "lucide-react";
import { api, Memory, Settings } from "./api";

function MemoryRow({ m, onChange }: { m: Memory; onChange: () => void }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(m.content);

  const save = async () => {
    const t = text.trim();
    if (t && t !== m.content) await api.updateMemory(m.id, t);
    setEditing(false);
    onChange();
  };

  return (
    <div className="row">
      <div className="row-main">
        {editing ? (
          <input
            className="field inline"
            autoFocus
            value={text}
            maxLength={300}
            onChange={(e) => setText(e.target.value)}
            onBlur={save}
            onKeyDown={(e) => {
              if (e.key === "Enter") save();
              if (e.key === "Escape") { setText(m.content); setEditing(false); }
            }}
          />
        ) : (
          <span className="memory-text" onClick={() => setEditing(true)} title="Click to edit">{m.content}</span>
        )}
      </div>
      <span className={`badge ${m.source === "user" ? "you" : ""}`}>{m.source === "user" ? "Added by you" : "Learned"}</span>
      <button className="icon-btn danger reveal" title="Forget" onClick={async () => { await api.deleteMemory(m.id); onChange(); }}>
        <Trash2 />
      </button>
    </div>
  );
}

export default function MemoryPage() {
  const [memories, setMemories] = useState<Memory[] | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [draft, setDraft] = useState("");

  const load = () => api.memories().then(setMemories).catch(() => setMemories([]));

  useEffect(() => {
    load();
    api.settings().then(setSettings).catch(() => {});
  }, []);

  const toggle = async (on: boolean) => setSettings(await api.updateSettings({ memory_enabled: on }));

  const add = async () => {
    const t = draft.trim();
    if (!t) return;
    await api.addMemory(t);
    setDraft("");
    load();
  };

  return (
    <div className="page">
      <div className="page-inner">
        <h1 className="page-title">Memory</h1>
        <p className="page-sub">What Blocky remembers about you across chats. Stored only on this device. Click any memory to edit it.</p>

        <div className="group-label">Learning</div>
        <div className="group">
          <label className="row">
            <span className="row-icon" style={{ background: "linear-gradient(180deg,#bf8cff,#8b5cf6)" }}><Sparkles /></span>
            <div className="row-main">
              <div>Learn from conversations</div>
              <div className="row-sub">Remember lasting facts and preferences you share. When off, Blocky neither learns nor uses memories.</div>
            </div>
            <input
              type="checkbox"
              className="switch"
              checked={settings?.memory_enabled ?? false}
              disabled={!settings}
              onChange={(e) => toggle(e.target.checked)}
            />
          </label>
        </div>

        <div className="group-label">{memories ? `${memories.length} ${memories.length === 1 ? "memory" : "memories"}` : "Memories"}</div>
        <div className="group">
          <form className="row" onSubmit={(e) => { e.preventDefault(); add(); }}>
            <input
              className="field"
              value={draft}
              maxLength={300}
              placeholder="Add something for Blocky to remember…"
              onChange={(e) => setDraft(e.target.value)}
            />
            <button type="submit" className="btn" disabled={!draft.trim()}><Plus /> Add</button>
          </form>
          {memories === null ? null : memories.length === 0 ? (
            <div className="row">
              <div className="empty-state" style={{ width: "100%" }}>
                <Brain /><br />
                Nothing yet. Tell Blocky about yourself in a chat, or add a memory above.
              </div>
            </div>
          ) : (
            memories.map((m) => <MemoryRow key={m.id} m={m} onChange={load} />)
          )}
        </div>

        {memories && memories.length > 0 && (
          <div style={{ marginTop: 18, textAlign: "center" }}>
            <button
              className="btn plain"
              onClick={async () => {
                if (confirm(`Forget all ${memories.length} memories? This can't be undone.`)) {
                  await api.deleteAllMemories();
                  load();
                }
              }}
            >
              Forget everything
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
