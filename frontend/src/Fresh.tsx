import { useState } from "react";
import { Calculator, Check, Code2, Plus, Puzzle, ShieldCheck, Sparkles, User } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { api, EvalRun, FreshSet } from "./api";

const ICONS: Record<string, LucideIcon> = { calculator: Calculator, puzzle: Puzzle, code: Code2 };
const SIZES = [10, 25, 50, 100];

/** Fresh sets: questions generated at run time, so they can't have been memorized. */
export function FreshPicker({ sets, picked, setPicked }: {
  sets: FreshSet[]; picked: Record<string, number>; setPicked: (p: Record<string, number>) => void;
}) {
  const [examples, setExamples] = useState<Record<string, string[]>>({});
  return (
    <div className="fresh">
      <div className="fresh-head">
        <span className="fresh-badge"><ShieldCheck /> Can't be memorized</span>
        <span className="row-sub">New questions are generated the moment you press Run. A model can't have seen them during training, so the score shows real skill.</span>
      </div>
      <div className="fresh-grid">
        {sets.map((s) => {
          const Icon = ICONS[s.icon] ?? User;
          const n = picked[s.id];
          const on = n !== undefined;
          return (
            <div key={s.id} className={`fresh-card ${on ? "on" : ""}`} style={{ "--c": s.color } as React.CSSProperties}>
              <div className="fresh-top">
                <span className="suite-icon"><Icon /></span>
                <b>{s.title}</b>
                <button className={`btn small ${on ? "" : "ghost"}`} onClick={() => {
                  const { [s.id]: _, ...rest } = picked;
                  setPicked(on ? rest : { ...picked, [s.id]: 25 });
                }}>{on ? <><Check /> Added</> : <><Plus /> Add</>}</button>
              </div>
              <div className="row-sub">{s.about}</div>
              {on && (
                <div className="size-pick">
                  <span className="row-sub">Questions:</span>
                  <div className="segmented">
                    {SIZES.map((x) => <button key={x} className={n === x ? "on" : ""} onClick={() => setPicked({ ...picked, [s.id]: x })}>{x}</button>)}
                  </div>
                </div>
              )}
              <button className="link-btn" onClick={() => api.freshPreview(s.id).then((qs) => setExamples((e) => ({ ...e, [s.id]: qs.map((q) => q.q) })))}>
                <Sparkles size={13} /> {examples[s.id] ? "Generate other examples" : "Show example questions"}
              </button>
              {examples[s.id] && <ul className="fresh-examples">{examples[s.id].map((q, i) => <li key={i}>{q}</li>)}</ul>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Contamination warning for a benchmark, from the year its questions became public. */
export function ContaminationTag({ published }: { published?: string | null }) {
  if (!published) return null;
  const year = parseInt(published, 10);
  const old = year <= 2024;
  return (
    <span className={`contam ${old ? "old" : "recent"}`}
      title={old ? "These questions have been public for years. Newer models were very likely trained on them, which inflates scores."
        : "Fairly recent, but models released after this may still have seen some questions."}>
      Public since {published}{old ? " · likely in training data" : ""}
    </span>
  );
}

// Famous benchmark ↔ fresh set measuring the same skill.
const PAIRS: { famous: string; fresh: string; skill: string }[] = [
  { famous: "bench:gsm8k", fresh: "fresh:math", skill: "Word-problem math" },
  { famous: "bench:humaneval_plus", fresh: "fresh:code", skill: "Small coding tasks" },
  { famous: "bench:humaneval", fresh: "fresh:code", skill: "Small coding tasks" },
];

export const MEMORIZATION_SPECS = ["bench:gsm8k@25", "fresh:math@25", "bench:humaneval_plus@25", "fresh:code@25"];

/** Compares each model's score on famous (public) vs fresh (never-seen) questions of the same kind. */
export function MemorizationCheck({ runs, label }: { runs: EvalRun[]; label: (r: EvalRun) => string }) {
  type Row = { model: string; skill: string; famous: number; fresh: number; nF: number; nR: number };
  const rows: Row[] = [];
  const latest = (r: EvalRun, prefix: string) => {
    const k = r.suites.find((s) => s.split("@")[0] === prefix);
    return k ? r.summary?.per_suite[k] : undefined;
  };
  const done = runs.filter((r) => r.status === "done" && r.summary);
  const models = Array.from(new Set(done.map(label)));
  for (const m of models) {
    for (const p of PAIRS) {
      if (rows.some((x) => x.model === m && x.skill === p.skill)) continue;
      const fam = done.find((r) => label(r) === m && latest(r, p.famous));
      const fr = done.find((r) => label(r) === m && latest(r, p.fresh));
      if (fam && fr) {
        const a = latest(fam, p.famous)!, b = latest(fr, p.fresh)!;
        rows.push({ model: m, skill: p.skill, famous: a.score, fresh: b.score, nF: a.total, nR: b.total });
      }
    }
  }
  if (rows.length === 0) {
    return <div className="row-sub">Run the <b>Memorization check</b> preset (in New test) to compare a model on famous benchmarks against fresh questions of the same kind.</div>;
  }
  return (
    <div className="memo">
      {rows.map((r) => {
        const gap = Math.round((r.famous - r.fresh) * 100);
        const verdict = gap >= 15 ? "Big gap: a good part of the famous score is probably memorized."
          : gap >= 6 ? "Some gap: the famous score may be a bit inflated."
          : gap <= -6 ? "Does better on fresh questions: no sign of memorization."
          : "About the same: no sign of memorization.";
        return (
          <div className="memo-row" key={r.model + r.skill}>
            <div className="memo-title"><b>{r.model}</b> · {r.skill}</div>
            <div className="memo-bars">
              <div className="memo-line"><span>Famous benchmark</span><div className="memo-track"><div style={{ width: `${r.famous * 100}%`, background: "var(--series-3)" }} /></div><b>{Math.round(r.famous * 100)}%</b></div>
              <div className="memo-line"><span>Fresh questions</span><div className="memo-track"><div style={{ width: `${r.fresh * 100}%`, background: "var(--series-1)" }} /></div><b>{Math.round(r.fresh * 100)}%</b></div>
            </div>
            <div className={`memo-verdict ${gap >= 15 ? "bad" : gap >= 6 ? "meh" : "good"}`}>{gap > 0 ? `${gap} points lower on fresh questions. ` : ""}{verdict}</div>
            <div className="row-sub">Based on {r.nF} + {r.nR} questions; small samples are noisy (±10 points is normal at 25 questions).</div>
          </div>
        );
      })}
    </div>
  );
}
