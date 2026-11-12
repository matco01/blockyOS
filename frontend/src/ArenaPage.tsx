import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, ChevronRight, EyeOff, Loader2, Play, Scale, Square, Swords, Trash2, Trophy } from "lucide-react";
import { api, ArenaMatch, ArenaResult, Contestant, LeaderRow, ModelFile, Status } from "./api";
import { Markdown } from "./Chat";
import { Pro, speedVerdict, Tabs, Term, useMode } from "./ui";

/* Colors follow the on-screen position (Answer 1 / Answer 2) so blind mode can't leak identity. */
const SIDE = ["var(--series-1)", "var(--series-2)"];

const SAMPLE_PROMPTS = [
  { label: "Logic puzzle", text: "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain briefly." },
  { label: "Creative", text: "Write a four-line poem about a computer that dreams of the ocean." },
  { label: "Coding", text: "Write a Python function that returns the n-th Fibonacci number in O(log n) time. Keep it short." },
  { label: "Explain", text: "Explain how a refrigerator works to a 10-year-old in under 100 words." },
  { label: "Facts", text: "What is the capital of Australia, and why isn't it Sydney? Two sentences." },
];

type Side = { model: string; think: boolean; temperature: string; system_prompt: string };

const label = (c: Contestant) =>
  `${c.model.replace(/\.gguf$/, "")}${c.think ? " · thinking" : ""}${c.sampling?.temperature != null ? ` · temp ${c.sampling.temperature}` : ""}${c.system_prompt ? " · custom prompt" : ""}`;

const STATE_TEXT: Record<ArenaResult["state"], string> = {
  waiting: "Waiting its turn…", loading: "Loading model onto the GPU…", reading: "Reading the prompt…",
  thinking: "Thinking…", writing: "Writing…", done: "Done", skipped: "Skipped",
};

function fmtMs(ms: number | null | undefined) {
  if (ms == null) return "–";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`;
}
function fmtEnergy(j: number | null | undefined) {
  if (j == null) return "–";
  return j >= 3600 ? `${(j / 3600).toFixed(2)} Wh` : `${Math.round(j)} J`;
}

/** Side-by-side bars per metric, with a plain-language takeaway. */
function Comparison({ m }: { m: ArenaMatch }) {
  const mode = useMode();
  const s = m.display.map((i) => m.results[i].stats);
  if (!s[0] || !s[1]) return null;
  const rows: { key: string; label: React.ReactNode; vals: (number | null)[]; fmt: (v: number | null) => string; lowerBetter: boolean; pro?: boolean }[] = [
    { key: "ttft", label: <Term k="ttft">First word after</Term>, vals: s.map((x) => x!.ttft_ms), fmt: fmtMs, lowerBetter: true },
    { key: "total", label: "Total time", vals: s.map((x) => x!.total_ms), fmt: fmtMs, lowerBetter: true },
    { key: "tps", label: <Term k="tps">Writing speed</Term>, vals: s.map((x) => x!.predicted_per_second), fmt: (v) => (v == null ? "–" : `${v.toFixed(0)} tok/s`), lowerBetter: false },
    { key: "len", label: "Length", vals: s.map((x) => x!.predicted_n), fmt: (v) => (v == null ? "–" : `${v} tokens`), lowerBetter: true },
    { key: "energy", label: "Energy used", vals: s.map((x) => x!.energy_j), fmt: fmtEnergy, lowerBetter: true },
    { key: "power", label: "GPU power", vals: s.map((x) => x!.power_w), fmt: (v) => (v == null ? "–" : `${v.toFixed(0)} W`), lowerBetter: true, pro: true },
    { key: "tpj", label: <Term k="tpj">Efficiency</Term>, vals: s.map((x) => x!.tokens_per_joule), fmt: (v) => (v == null ? "–" : `${v.toFixed(2)} tok/J`), lowerBetter: false, pro: true },
  ];

  const takeaways: string[] = [];
  const name = (i: number) => `Answer ${i + 1}`;
  const ratio = (a: number | null, b: number | null) => (a && b ? Math.max(a, b) / Math.min(a, b) : null);
  const tr = ratio(s[0].total_ms, s[1].total_ms);
  if (tr && tr >= 1.3) takeaways.push(`${name(s[0].total_ms < s[1].total_ms ? 0 : 1)} finished ${tr.toFixed(1)}× sooner.`);
  const er = ratio(s[0].energy_j, s[1].energy_j);
  if (er && er >= 1.3) takeaways.push(`${name(s[0].energy_j! < s[1].energy_j! ? 0 : 1)} used ${er.toFixed(1)}× less energy.`);
  const lr = ratio(s[0].predicted_n, s[1].predicted_n);
  if (lr && lr >= 2) takeaways.push(`${name(s[0].predicted_n! > s[1].predicted_n! ? 0 : 1)} wrote ${lr.toFixed(1)}× more${m.contestants.some((c) => c.think) ? " (thinking counts too)" : ""}.`);
  if (s.some((x) => x!.hit_limit)) takeaways.push("One answer hit the length limit and was cut off.");

  return (
    <div className="arena-compare">
      {takeaways.length > 0 && <div className="takeaways">{takeaways.map((t) => <span key={t}>{t}</span>)}</div>}
      {rows.filter((r) => !r.pro || mode === "pro").map((r) => {
        const max = Math.max(...r.vals.map((v) => v ?? 0), 1e-9);
        const best = r.vals[0] != null && r.vals[1] != null && r.vals[0] !== r.vals[1]
          ? (r.lowerBetter ? (r.vals[0] < r.vals[1] ? 0 : 1) : (r.vals[0] > r.vals[1] ? 0 : 1)) : null;
        return (
          <div className="cmp-row" key={r.key}>
            <div className="cmp-label">{r.label}</div>
            <div className="cmp-bars">
              {r.vals.map((v, i) => (
                <div className="cmp-line" key={i}>
                  <div className="cmp-bar" style={{ width: `${((v ?? 0) / max) * 100}%`, background: SIDE[i] }} />
                  <span className={`cmp-val ${best === i ? "best" : ""}`}>{r.fmt(v)}{best === i && " ✓"}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
      <div className="row-sub">Each answer ran alone on the GPU with the same seed. Energy = average GPU power × time{s.some((x) => (x!.predicted_n ?? 0) < 30) ? "; very short answers give rough energy numbers" : ""}.</div>
    </div>
  );
}

function AnswerColumn({ m, pos }: { m: ArenaMatch; pos: number }) {
  const idx = m.display[pos];
  const r = m.results[idx];
  const c = m.contestants[idx];
  const hidden = m.blind && !m.vote;
  const won = m.vote === String(idx);
  const live = r.state !== "done" && r.state !== "skipped";
  return (
    <div className={`answer-col ${won ? "won" : ""}`} style={{ "--c": SIDE[pos] } as React.CSSProperties}>
      <div className="answer-head">
        <span className="answer-tag">{pos + 1}</span>
        <div className="answer-title">
          <b>{hidden ? `Answer ${pos + 1}` : label(c)}</b>
          <span className="row-sub">{live ? <><Loader2 className="spin" size={12} /> {STATE_TEXT[r.state]}</> : r.stats
            ? `${fmtMs(r.stats.total_ms)} · ${r.stats.predicted_per_second?.toFixed(0) ?? "–"} tok/s · ${speedVerdict(r.stats.predicted_per_second).label}`
            : STATE_TEXT[r.state]}</span>
        </div>
        {won && <span className="badge you"><Trophy size={12} /> Winner</span>}
      </div>
      <div className="answer-body">
        {r.reasoning && (
          <details className="reasoning" open={r.state === "thinking"}>
            <summary><ChevronRight className="chev" /><span className={r.state === "thinking" ? "label shimmer" : "label"}>
              {r.state === "thinking" ? "Thinking…" : r.stats?.think_ms != null
                ? `Thought for ${(r.stats.think_ms / 1000).toFixed(1)} s`
                : "Thought process"}</span></summary>
            <div className="reasoning-body">{r.reasoning}</div>
          </details>
        )}
        {r.text ? <Markdown text={r.text} /> : live ? <div className="typing"><span /><span /><span /></div> : <div className="muted small">No answer.</div>}
      </div>
    </div>
  );
}

export default function ArenaPage({ status }: { status: Status | null }) {
  const mode = useMode();
  const [models, setModels] = useState<ModelFile[]>([]);
  const [prompt, setPrompt] = useState("");
  const [sides, setSides] = useState<Side[]>([
    { model: "", think: false, temperature: "", system_prompt: "" },
    { model: "", think: false, temperature: "", system_prompt: "" },
  ]);
  const [blind, setBlind] = useState(true);
  const [current, setCurrent] = useState<ArenaMatch | null>(null);
  const [history, setHistory] = useState<ArenaMatch[]>([]);
  const [board, setBoard] = useState<LeaderRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"match" | "history">("match");

  const loadHistory = useCallback(() => {
    api.arenaList().then(setHistory).catch(() => {});
    api.arenaBoard().then(setBoard).catch(() => {});
  }, []);

  useEffect(() => {
    api.models().then((d) => {
      setModels(d.models);
      const active = d.runtime.model ?? d.models[0]?.file ?? "";
      const other = d.models.find((m) => m.file !== active)?.file ?? active;
      setSides((s) => [{ ...s[0], model: s[0].model || other }, { ...s[1], model: s[1].model || active }]);
    }).catch(() => {});
    loadHistory();
    const linked = Number(new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("match"));
    if (linked) api.arenaGet(linked).then(setCurrent).catch(() => {});
  }, [loadHistory]);

  // Keep a shareable link to the open match.
  useEffect(() => {
    const want = current ? `#/arena?match=${current.id}` : "#/arena";
    if (window.location.hash !== want) window.history.replaceState(null, "", want);
  }, [current?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Poll the open match while it runs.
  useEffect(() => {
    if (!current || !(current.status === "running" || current.status === "queued")) return;
    const t = setInterval(() => {
      api.arenaGet(current.id).then((m) => {
        setCurrent(m);
        if (m.status !== "running" && m.status !== "queued") loadHistory();
      }).catch(() => {});
    }, 400);
    return () => clearInterval(t);
  }, [current, loadHistory]);

  const setSide = (i: number, patch: Partial<Side>) => setSides((s) => s.map((x, j) => (j === i ? { ...x, ...patch } : x)));

  const matchups = useMemo(() => {
    const active = models.find((m) => m.active)?.file ?? models[0]?.file ?? "";
    const other = models.find((m) => m.file !== active)?.file;
    const base = { temperature: "", system_prompt: "" };
    return [
      other && { label: "Model vs model", a: { ...base, model: other, think: false }, b: { ...base, model: active, think: false } },
      { label: "Thinking vs not", a: { ...base, model: active, think: true }, b: { ...base, model: active, think: false } },
      { label: "Careful vs creative", a: { ...base, model: active, think: false, temperature: "0.2" }, b: { ...base, model: active, think: false, temperature: "1.2" } },
    ].filter(Boolean) as { label: string; a: Side; b: Side }[];
  }, [models]);

  const run = async () => {
    setError(null);
    try {
      const { id } = await api.arenaStart({
        prompt, blind,
        contestants: sides.map((s) => ({
          model: s.model, think: s.think,
          sampling: s.temperature !== "" ? { temperature: Number(s.temperature) } : {},
          ...(s.system_prompt.trim() ? { system_prompt: s.system_prompt } : {}),
        })),
      });
      setCurrent(await api.arenaGet(id));
      window.scrollTo({ top: 0 });
    } catch (e) { setError((e as Error).message); }
  };

  const vote = async (winner: "0" | "1" | "tie") => {
    if (!current) return;
    setCurrent(await api.arenaVote(current.id, winner));
    loadHistory();
  };

  const running = current && (current.status === "running" || current.status === "queued");
  const [stopping, setStopping] = useState(false);
  useEffect(() => { if (!running) setStopping(false); }, [running]);
  const swapNeeded = sides.some((s) => s.model !== models.find((m) => m.active)?.file);

  return (
    <div className="page">
      <div className="page-inner wide">
        <h1 className="page-title">Arena</h1>
        <p className="page-sub">Ask two models (or two settings) the same question and compare the answers side by side.</p>
        <Tabs value={tab} onChange={setTab} tabs={[
          { id: "match", label: "Match" },
          { id: "history", label: "Leaderboard & history", badge: history.length },
        ]} />

        {tab === "match" && (<>
        {/* Setup */}
        <div className="group-label">Matchup</div>
        <div className="group">
          <div className="row"><div className="chips" style={{ margin: 0 }}>
            {matchups.map((mu) => <button key={mu.label} className="chip" onClick={() => setSides([mu.a, mu.b])}>{mu.label}</button>)}
          </div></div>
          <div className="row contestants">
            {sides.map((s, i) => (
              <div key={i} className="contestant" style={{ "--c": SIDE[i] } as React.CSSProperties}>
                <div className="contestant-head"><span className="answer-tag">{i + 1}</span> Contestant {i + 1}</div>
                <select className="field select" value={s.model} onChange={(e) => setSide(i, { model: e.target.value })}>
                  {models.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file} · {m.quant}</option>)}
                </select>
                <label className="toggle-line">
                  <input type="checkbox" className="switch" checked={s.think} onChange={(e) => setSide(i, { think: e.target.checked })} />
                  <span>Think first</span>
                </label>
                <label className="toggle-line">
                  <span><Term k="temperature">Temperature</Term></span>
                  <input className="field num" placeholder="default" value={s.temperature} onChange={(e) => setSide(i, { temperature: e.target.value.replace(/[^\d.]/g, "") })} />
                </label>
                <Pro>
                  <textarea className="field" rows={2} placeholder="Custom system prompt (optional)" value={s.system_prompt} onChange={(e) => setSide(i, { system_prompt: e.target.value })} />
                </Pro>
              </div>
            ))}
          </div>
          <div className="row block">
            <textarea className="field arena-prompt" rows={3} placeholder="What should they both answer?" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            <div className="chips" style={{ marginTop: 10, marginBottom: 0 }}>
              {SAMPLE_PROMPTS.map((p) => <button key={p.label} className="chip" onClick={() => setPrompt(p.text)}>{p.label}</button>)}
            </div>
          </div>
          <div className="row apply-row">
            <label className="toggle-line" style={{ flex: 1 }}>
              <input type="checkbox" className="switch" checked={blind} onChange={(e) => setBlind(e.target.checked)} />
              <span><EyeOff size={14} style={{ verticalAlign: -2 }} /> Blind: hide which is which until you vote</span>
            </label>
            {running ? (
              <button className="btn ghost" disabled={stopping} onClick={() => { if (current) { setStopping(true); api.arenaCancel(current.id); } }}>
                {stopping ? <><Loader2 className="spin" /> Stopping…</> : <><Square /> Stop</>}
              </button>
            ) : (
              <button className="btn" disabled={!prompt.trim() || !!status?.busy || sides.some((s) => !s.model)} onClick={run}><Play /> Fight</button>
            )}
          </div>
          {swapNeeded && !running && (
            <div className="row"><div className="row-sub">A different model than the chat one is involved: each takes ~5–10 s to load, and chat pauses until the match ends.</div></div>
          )}
        </div>
        {error && <div className="error" style={{ marginTop: 12 }}><AlertCircle /> {error}</div>}

        {/* Match */}
        {current && (
          <>
            <div className="group-label">Match #{current.id}{current.status === "error" ? " · failed" : current.status === "cancelled" ? " · stopped" : ""}</div>
            <div className="match">
              <div className="match-prompt"><Swords size={15} /> {current.prompt}</div>
              {current.error && <div className="error"><AlertCircle /> {current.error}</div>}
              <div className="answers">
                <AnswerColumn m={current} pos={0} />
                <AnswerColumn m={current} pos={1} />
              </div>
              {current.status === "done" && (
                <div className="vote-bar">
                  {current.vote ? (
                    <span className="vote-done"><Trophy size={15} /> {current.vote === "tie" ? "You called it a tie." : `You picked Answer ${current.display.indexOf(Number(current.vote)) + 1}.`}
                      {current.blind && " Here's who was who."}</span>
                  ) : (
                    <>
                      <span>Which answer is better?</span>
                      <button className="btn vote-1" onClick={() => vote(String(current.display[0]) as "0" | "1")}>Answer 1</button>
                      <button className="btn ghost" onClick={() => vote("tie")}><Scale /> Tie</button>
                      <button className="btn vote-2" onClick={() => vote(String(current.display[1]) as "0" | "1")}>Answer 2</button>
                    </>
                  )}
                </div>
              )}
              {current.status === "done" && (!current.blind || current.vote) && <Comparison m={current} />}
              {current.status === "done" && current.blind && !current.vote && <div className="row-sub center">Vote first, then you'll see the numbers and which model is which.</div>}
            </div>
          </>
        )}

        </>)}

        {tab === "history" && (<>
        {board.length === 0 && history.length === 0 && (
          <div className="group" style={{ marginTop: 20 }}><div className="empty-state"><Swords /><br />No matches yet. Run one in the Match tab and vote for the better answer.</div></div>
        )}
        {/* Leaderboard */}
        {board.length > 0 && (
          <>
            <div className="group-label">Leaderboard <span className="muted" style={{ textTransform: "none", letterSpacing: 0 }}>· from your votes</span></div>
            <div className="group pad">
              {board.map((b, i) => (
                <div className="lb-row" key={b.label}>
                  <span className="lb-rank">{i + 1}</span>
                  <div className="lb-main">
                    <div className="lb-name">{b.label}</div>
                    <div className="lb-bar"><div style={{ width: `${b.score * 100}%` }} /></div>
                  </div>
                  <div className="lb-num"><b>{Math.round(b.score * 100)}%</b><span>{b.wins}W {b.ties}T {b.losses}L</span></div>
                  {mode === "pro" && <div className="lb-num"><b>{b.avg_tps?.toFixed(0) ?? "–"}</b><span>tok/s</span></div>}
                </div>
              ))}
              <div className="row-sub" style={{ marginTop: 10 }}>Score = wins + half of ties, over matches you voted on. Needs more votes to mean much.</div>
            </div>
          </>
        )}

        {/* History */}
        {history.length > 0 && (
          <>
            <div className="group-label">Past matches</div>
            <div className="group">
              {history.map((h) => (
                <div className="row hist-row" key={h.id} onClick={() => { setCurrent(h); setTab("match"); }}>
                  <div className="row-main">
                    <div className="hist-prompt">{h.prompt}</div>
                    <div className="row-sub">#{h.id} · {label(h.contestants[0])} vs {label(h.contestants[1])}
                      {h.vote ? ` · ${h.vote === "tie" ? "tie" : `winner: ${label(h.contestants[Number(h.vote)])}`}` : h.status === "done" ? " · not voted" : ` · ${h.status}`}</div>
                  </div>
                  <button className="icon-btn danger" title="Delete" onClick={(e) => { e.stopPropagation(); api.arenaDelete(h.id).then(loadHistory); if (current?.id === h.id) setCurrent(null); }}><Trash2 /></button>
                </div>
              ))}
            </div>
          </>
        )}
        </>)}
      </div>
    </div>
  );
}
