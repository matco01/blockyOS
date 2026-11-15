import { useMemo, useState } from "react";
import type { ExpRun } from "./api";
import { colorOf, nameOf } from "./expmetrics";
import { detectableDelta, normInv, questionsNeeded, repeatsNeeded } from "./stats";

/* ---------- RULER-style long-context results ---------- */

export function RulerTable({ runs }: { runs: ExpRun[] }) {
  const rs = runs.filter((r) => r.results.ruler?.cells?.length);
  if (!rs.length) return null;
  const lengths = Array.from(new Set(rs.flatMap((r) => r.results.ruler!.lengths))).sort((a, b) => a - b);
  const tasks = rs[0].results.ruler!.tasks;
  return (
    <div className="xruler">
      <div className="mini-title">Long-context suite (RULER-style) <span className="muted">· average score per context length; effective length = longest with ≥ {Math.round(rs[0].results.ruler!.threshold * 1000) / 10}%</span></div>
      <div className="table-wrap"><table className="data">
        <thead><tr><th>Run</th>{lengths.map((L) => <th key={L} className="r">{L / 1024}k</th>)}<th className="r">Effective</th>{tasks.map((t) => <th key={t} className="r">{t}</th>)}</tr></thead>
        <tbody>{rs.map((r) => {
          const x = r.results.ruler!;
          return (
            <tr key={r.id}>
              <td><span className="xdot" style={{ background: colorOf(runs.indexOf(r)) }} />{nameOf(r, runs.indexOf(r))}</td>
              {lengths.map((L) => {
                const v = x.by_length[String(L)];
                const ci = x.cells.filter((c) => c.length === L).map((c) => c.ci95);
                return <td key={L} className="r" title={ci.length ? `± ${(Math.max(...ci) * 100).toFixed(0)} points (95% CI, per task)` : ""}
                  style={v != null ? { background: `color-mix(in srgb, var(--success) ${Math.round(v * 35)}%, transparent)` } : undefined}>{v != null ? `${(v * 100).toFixed(0)}%` : "–"}</td>;
              })}
              <td className="r"><b>{x.effective_length ? `${x.effective_length / 1024}k` : x.lengths.length ? "< " + (x.lengths[0] / 1024) + "k" : "Not measured"}</b></td>
              {tasks.map((t) => <td key={t} className="r">{x.by_task[t] != null ? `${(x.by_task[t] * 100).toFixed(0)}%` : "–"}</td>)}
            </tr>
          );
        })}</tbody>
      </table></div>
      <div className="row-sub">{rs[0].results.ruler!.method}. Tasks: multi-key retrieval with distractors, all values of one key, variable tracking through chained assignments, most common words in a long list.</div>
    </div>
  );
}

/* ---------- Planning: how many repeats / questions are needed ---------- */

const SPEED = [
  { key: "tg", label: "Decode (llama-bench)", get: (r: ExpRun) => r.results.speed?.tg, sd: (r: ExpRun) => r.results.speed?.tg_sd },
  { key: "pp", label: "Prefill (llama-bench)", get: (r: ExpRun) => r.results.speed?.pp, sd: (r: ExpRun) => r.results.speed?.pp_sd },
  { key: "srv", label: "Serving decode", get: (r: ExpRun) => r.results.serving?.tg, sd: (r: ExpRun) => r.results.serving?.tg_sd },
];

export function Planner({ runs }: { runs: ExpRun[] }) {
  const [mk, setMk] = useState("tg");
  const [effPct, setEffPct] = useState(3);
  const [accPts, setAccPts] = useState(2);
  const [power, setPower] = useState(0.8);
  const k = Math.max(1, runs.length - 1);
  const alpha = 0.05 / k;  // planning for the strict (Bonferroni/Holm) case
  const m = SPEED.find((x) => x.key === mk)!;

  const speed = useMemo(() => {
    const withV = runs.filter((r) => m.get(r) != null);
    if (!withV.length) return null;
    const mean = withV.reduce((s, r) => s + m.get(r)!, 0) / withV.length;
    // Between-run spread from repeats of an identical setup, if there are any; that includes session drift.
    const groups = new Map<string, number[]>();
    for (const r of withV) { const key = JSON.stringify(r.config); groups.set(key, [...(groups.get(key) ?? []), m.get(r)!]); }
    const reps = [...groups.values()].filter((g) => g.length > 1);
    let sigma: number, source: string;
    if (reps.length) {
      let ss = 0, df = 0;
      for (const g of reps) { const gm = g.reduce((a, b) => a + b, 0) / g.length; ss += g.reduce((a, b) => a + (b - gm) ** 2, 0); df += g.length - 1; }
      sigma = Math.sqrt(ss / df); source = `measured from ${reps.reduce((a, g) => a + g.length, 0)} runs of repeated setups (includes session drift)`;
    } else {
      const sds = withV.map((r) => m.sd(r)).filter((x): x is number => x != null).sort((a, b) => a - b);
      if (!sds.length) return null;
      sigma = (1.2533 * sds[Math.floor(sds.length / 2)]) / Math.sqrt(5);
      source = "estimated from the spread inside each run (5 repetitions); repeat a setup once to measure drift between sessions, usually larger";
    }
    const z = normInv(1 - alpha / 2) + normInv(power);
    return { mean, sigma, source, n: repeatsNeeded(sigma, (effPct / 100) * mean, alpha, power), mde: (z * sigma * Math.SQRT2) / mean };
  }, [runs, mk, effPct, power, alpha]); // eslint-disable-line react-hooks/exhaustive-deps

  const mmlu = useMemo(() => {
    const pairs = runs.map((r) => r.stats?.mmlu_flips && r.results.mmlu?.n ? (r.stats.mmlu_flips.lost + r.stats.mmlu_flips.gained) / r.results.mmlu.n : null).filter((x): x is number => x != null && x > 0);
    const disc = pairs.length ? pairs.sort((a, b) => a - b)[Math.floor(pairs.length / 2)] : 0.12;
    const n = runs.find((r) => r.results.mmlu?.n)?.results.mmlu?.n ?? 560;
    return { disc, measured: pairs.length > 0, n, need: questionsNeeded(accPts / 100, disc, alpha, power), mde: detectableDelta(n, disc, alpha, power) };
  }, [runs, accPts, power, alpha]);

  return (
    <div className="xplan">
      <div className="mini-title">How many runs or questions do I need?</div>
      <div className="row-sub">Planned for {k} comparison{k > 1 ? "s" : ""} against the baseline at once (α = 0.05 / {k}), with {Math.round(power * 100)}% power: the chance of detecting a real difference of that size.
        {" "}<label>Power <select className="field select small-select" value={power} onChange={(e) => setPower(Number(e.target.value))}>{[0.8, 0.9, 0.95].map((p) => <option key={p} value={p}>{p * 100}%</option>)}</select></label></div>
      <div className="xplan-grid">
        <div className="xplan-card">
          <b>Speed</b>
          <div className="xplan-row"><select className="field select small-select" value={mk} onChange={(e) => setMk(e.target.value)}>{SPEED.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}</select>
            <span>difference to detect</span><input className="field num" type="number" min={0.5} step={0.5} value={effPct} onChange={(e) => setEffPct(Number(e.target.value))} /><span>%</span></div>
          {speed ? <>
            <div className="xplan-big">{speed.n ?? "–"} <span>runs per setup</span></div>
            <div className="row-sub">Run-to-run spread σ ≈ {speed.sigma.toFixed(2)} tok/s ({((speed.sigma / speed.mean) * 100).toFixed(1)}%), {speed.source}. With one run per setup, the smallest detectable difference is about <b>{(speed.mde * 100).toFixed(1)}%</b>.</div>
          </> : <div className="row-sub">No speed results yet.</div>}
        </div>
        <div className="xplan-card">
          <b>MMLU-Pro accuracy</b>
          <div className="xplan-row"><span>difference to detect</span><input className="field num" type="number" min={0.5} step={0.5} value={accPts} onChange={(e) => setAccPts(Number(e.target.value))} /><span>points</span></div>
          <div className="xplan-big">{mmlu.need != null ? mmlu.need.toLocaleString() : "–"} <span>questions</span></div>
          <div className="row-sub">Two setups disagree on {(mmlu.disc * 100).toFixed(0)}% of questions ({mmlu.measured ? "measured in this experiment" : "typical value; measured once you have two runs"}). With the current {mmlu.n} questions the smallest detectable difference is about <b>{(mmlu.mde * 100).toFixed(1)} points</b>.
            {mmlu.need != null && mmlu.need > mmlu.n && <> Use a larger size in “Add tests” (1,400 or 3,500 questions){mmlu.need > 3500 ? ", or look at KL divergence, which is far more sensitive" : ""}.</>}</div>
        </div>
      </div>
    </div>
  );
}
