import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, Check, Cpu, Download, FlaskConical, Gauge, Loader2, MemoryStick, Play, Plus, Sparkles, Square, Trash2, Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { api, ExpConfig, Experiment, ExpResults, ExpRun, ModelFile, Status } from "./api";
import { LineChart, Term } from "./ui";

/* ---------- Metric registry: one place defines every number, its noise and which direction is better ---------- */

type Ctx = { run: ExpRun; base: ExpRun };
type Metric = {
  key: string; group: string; label: string; term?: string; unit: string; better: "higher" | "lower" | null;
  get: (r: ExpResults) => number | null | undefined;
  sd?: (r: ExpResults) => number | null | undefined;        // ± noise of one measurement (for significance)
  sig?: (c: Ctx) => boolean | null;                          // custom significance test
  fmt?: (v: number) => string; note?: string;
  minRel?: number;                                           // smallest relative change treated as real (no noise model)
};
const n1 = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
const pct = (v: number) => v.toFixed(1);
const gb = (mb?: number | null) => (mb != null ? mb / 1024 : null);

const METRICS: Metric[] = [
  { key: "tg", group: "Speed", label: "Decode (writing)", term: "tg", unit: "tok/s", better: "higher", get: (r) => r.speed?.tg, sd: (r) => r.speed?.tg_sd },
  { key: "tg_long", group: "Speed", label: "Decode at 4k context", unit: "tok/s", better: "higher", get: (r) => r.speed?.tg_d4096, sd: (r) => r.speed?.tg_d4096_sd },
  { key: "pp", group: "Speed", label: "Prefill (reading)", term: "pp", unit: "tok/s", better: "higher", get: (r) => r.speed?.pp, sd: (r) => r.speed?.pp_sd, fmt: (v) => v.toFixed(0) },
  { key: "pp_long", group: "Speed", label: "Prefill at 4k context", unit: "tok/s", better: "higher", get: (r) => r.speed?.pp_d4096, sd: (r) => r.speed?.pp_d4096_sd, fmt: (v) => v.toFixed(0) },
  { key: "ttft", group: "Speed", label: "Latency, 512-token prompt", unit: "ms", better: "lower", get: (r) => (r.speed?.pp ? (512 / r.speed.pp) * 1000 : null), fmt: (v) => v.toFixed(0) },
  { key: "mmlu", group: "Quality", label: "MMLU-Pro accuracy", unit: "%", better: "higher", get: (r) => (r.mmlu ? r.mmlu.acc * 100 : null), fmt: pct,
    sig: (c) => (c.run.stats?.mmlu_p != null ? c.run.stats.mmlu_p < 0.05 : null), note: "paired McNemar test, p < 0.05" },
  { key: "mmlu_p", group: "Quality", label: "MMLU-Pro P(correct)", unit: "%", better: "higher", get: (r) => (r.mmlu ? r.mmlu.p_correct_mean * 100 : null), fmt: pct },
  { key: "kld", group: "Quality", label: "KL divergence vs baseline", unit: "", better: "lower", get: (r) => r.kld, fmt: (v) => v.toFixed(4) },
  { key: "kld99", group: "Quality", label: "KLD, 99th percentile", unit: "", better: "lower", get: (r) => r.kld_p99, fmt: (v) => v.toFixed(3) },
  { key: "same_top", group: "Quality", label: "Same top token as baseline", unit: "%", better: "higher", get: (r) => r.same_top, sd: (r) => r.same_top_sd, fmt: pct },
  { key: "ppl", group: "Quality", label: "Perplexity (WikiText-2)", unit: "", better: "lower", get: (r) => r.ppl, sd: (r) => r.ppl_sd, fmt: (v) => v.toFixed(3) },
  { key: "vram_model", group: "Memory", label: "VRAM, model buffers", term: "vram", unit: "GB", better: "lower", get: (r) => gb(r.hw?.headroom?.vram_model_mb), fmt: (v) => v.toFixed(2) },
  { key: "vram_free", group: "Memory", label: "VRAM free", unit: "GB", better: "higher", get: (r) => gb(r.hw?.headroom?.vram_free_mb), fmt: (v) => v.toFixed(2) },
  { key: "kv", group: "Memory", label: "Context memory per 1k tokens", term: "kv", unit: "MB", better: "lower", get: (r) => r.hw?.headroom?.kv_mb_per_1k_tokens },
  { key: "w_cpu", group: "Memory", label: "Weights in system RAM", unit: "GB", better: "lower", get: (r) => r.hw?.weights_on_cpu_gb, fmt: (v) => v.toFixed(2) },
  { key: "load", group: "Memory", label: "Load time", unit: "s", better: "lower", get: (r) => r.load_seconds, minRel: 0.4, note: "Depends on the disk cache: only large changes count" },
  { key: "eff_bw", group: "Hardware", label: "Effective weight bandwidth", unit: "GB/s", better: "higher", get: (r) => r.hw?.decode?.effective_gbps, fmt: (v) => v.toFixed(0) },
  { key: "roof", group: "Hardware", label: "Decode, % of memory roof", unit: "%", better: "higher", get: (r) => (r.hw?.decode?.roof_util != null ? r.hw.decode.roof_util * 100 : null), fmt: (v) => v.toFixed(0) },
  { key: "tflops", group: "Hardware", label: "Prefill compute", unit: "TFLOPS", better: "higher", get: (r) => r.hw?.prefill?.achieved_tflops },
  { key: "croof", group: "Hardware", label: "Prefill, % of compute roof", unit: "%", better: "higher", get: (r) => (r.hw?.prefill?.roof_util != null ? r.hw.prefill.roof_util * 100 : null), fmt: (v) => v.toFixed(0) },
  { key: "clock", group: "Hardware", label: "SM clock, % of max", unit: "%", better: "higher", get: (r) => (r.hw?.clocks.ratio != null ? r.hw.clocks.ratio * 100 : null), fmt: (v) => v.toFixed(0) },
  { key: "temp", group: "Hardware", label: "GPU temperature, peak", unit: "°C", better: "lower", get: (r) => r.hw?.clocks.temp_max_c, fmt: (v) => v.toFixed(0) },
  { key: "gpu_util", group: "Hardware", label: "GPU busy (decode)", unit: "%", better: null, get: (r) => r.hw?.clocks.gpu_util_avg, fmt: (v) => v.toFixed(0) },
  { key: "cpu_util", group: "Hardware", label: "CPU busy (decode)", unit: "%", better: null, get: (r) => r.hw?.clocks.cpu_util_avg, fmt: (v) => v.toFixed(0) },
  { key: "power", group: "Energy", label: "GPU power (decode)", unit: "W", better: "lower", get: (r) => r.hw?.clocks.power_avg_w, fmt: (v) => v.toFixed(0) },
  { key: "tpj", group: "Energy", label: "Tokens per joule (GPU)", term: "tpj", unit: "tok/J", better: "higher",
    get: (r) => (r.speed?.tg && r.hw?.clocks.power_avg_w ? r.speed.tg / r.hw.clocks.power_avg_w : null), fmt: (v) => v.toFixed(2) },
];

/** Difference vs baseline, with a noise check: inside 2σ of the combined measurement noise = not significant. */
function delta(m: Metric, c: Ctx, session = 0) {
  const v = m.get(c.run.results), b = m.get(c.base.results);
  if (v == null || b == null || b === 0) return null;
  const d = (v - b) / Math.abs(b);
  let sig: boolean | null = m.sig ? m.sig(c) : null;
  if (sig == null && m.sd) {
    const s1 = m.sd(c.run.results), s2 = m.sd(c.base.results);
    if (s1 != null && s2 != null) sig = Math.abs(v - b) > 2 * Math.sqrt(s1 * s1 + s2 * s2);
  }
  if (sig == null) sig = Math.abs(d) >= (m.minRel ?? 0.02);
  // Session-to-session drift (measured by repeating the baseline) must also be exceeded.
  if (sig && !m.sig && session > 0 && Math.abs(v - b) <= 2 * session) sig = false;
  const text = `${d > 0 ? "+" : ""}${(d * 100).toFixed(Math.abs(d) < 0.1 ? 1 : 0)}%`;
  if (!sig) return { text, tone: "same", title: "Within measurement noise" };
  if (!m.better) return { text, tone: "same", title: "" };
  const good = m.better === "higher" ? d > 0 : d < 0;
  return { text, tone: good ? "good" : "bad", title: "Significant" };
}

const PHASES = [
  { id: "loading", label: "Load", what: "memory layout, load time" },
  { id: "quality", label: "Quality", what: "MMLU-Pro, 560 questions" },
  { id: "speed", label: "Speed", what: "llama-bench, 5 reps" },
  { id: "fidelity", label: "Fidelity", what: "PPL + KLD, 16k tokens" },
];
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];
const colorOf = (i: number) => SERIES[i] ?? "var(--text-2)";
const nameOf = (r: ExpRun, i: number) => (i === 0 ? "Baseline" : r.label);

/* ---------- Pieces ---------- */

function Tile({ icon: Icon, color, label, value, unit, sub }: { icon: LucideIcon; color: string; label: React.ReactNode; value: string; unit?: string; sub?: string }) {
  return (
    <div className="xt" style={{ "--c": color } as React.CSSProperties}>
      <div className="xt-head"><span className="xt-icon"><Icon /></span>{label}</div>
      <div className="xt-value">{value}{unit && <small>{unit}</small>}</div>
      {sub && <div className="xt-sub">{sub}</div>}
    </div>
  );
}

function Profile({ run }: { run: ExpRun }) {
  const r = run.results, s = r.speed, m = r.mmlu, hw = r.hw;
  return (
    <div className="xt-grid">
      <Tile icon={Gauge} color="#7c5cff" label={<Term k="tg">Decode</Term>} value={s?.tg ? n1(s.tg) : "–"} unit={s?.tg_sd ? `± ${s.tg_sd.toFixed(1)} tok/s` : "tok/s"}
        sub={hw?.decode ? `${hw.decode.ms_per_token} ms/token · ${hw.decode.effective_gbps} GB/s effective` : undefined} />
      <Tile icon={Zap} color="#ffa94d" label={<Term k="pp">Prefill</Term>} value={s?.pp ? s.pp.toFixed(0) : "–"} unit={s?.pp_sd ? `± ${s.pp_sd.toFixed(0)} tok/s` : "tok/s"}
        sub={hw?.prefill ? `${hw.prefill.achieved_tflops} TFLOPS of ${hw.peaks.tflops ?? "?"} peak` : undefined} />
      <Tile icon={Sparkles} color="#ff5c9d" label="MMLU-Pro" value={m ? (m.acc * 100).toFixed(1) : "–"} unit="%"
        sub={m ? `95% CI ${(m.ci95[0] * 100).toFixed(0)}–${(m.ci95[1] * 100).toFixed(0)}% · n = ${m.n} · chance 10%` : "Not measured in this run (older profile)"} />
      <Tile icon={MemoryStick} color="#22d3ee" label={<Term k="vram">VRAM</Term>} value={hw?.headroom ? (hw.headroom.vram_model_mb / 1024).toFixed(2) : "–"} unit="GB"
        sub={hw?.headroom ? `${(hw.headroom.vram_free_mb / 1024).toFixed(2)} GB free of ${(hw.headroom.vram_capacity_mb / 1024).toFixed(0)} GB · ${r.memory?.layers?.gpu}/${r.memory?.layers?.total} layers on GPU` : undefined} />
    </div>
  );
}

function Running({ run, onStop }: { run: ExpRun; onStop: () => void }) {
  const idx = PHASES.findIndex((p) => p.id === run.phase);
  const mp = run.results.mmlu_progress;
  return (
    <div className="xrun">
      <div className="xrun-head"><Loader2 className="spin" /> <b>Profiling “{run.label}”</b><span className="row-sub">4–8 minutes · chat is paused</span>
        <button className="btn small ghost" onClick={onStop}><Square /> Stop</button></div>
      <div className="xsteps">
        {PHASES.map((p, i) => (
          <div key={p.id} className={`xstep ${i < idx ? "done" : i === idx ? "now" : ""}`}>
            <span>{i < idx ? <Check size={13} /> : i + 1}</span><div><b>{p.label}</b><em>{p.id === "quality" && mp && i === idx ? `${mp.done}/${mp.total} · ${mp.correct} correct` : p.what}</em></div>
          </div>
        ))}
      </div>
    </div>
  );
}

const CTX = [4096, 8192, 16384, 32768, 65536];
const VRAM_SIM = [null, 6144, 4096, 2048];

function ChangePanel({ base, models, onRun, disabled }: { base: ExpConfig; models: ModelFile[]; onRun: (label: string, cfg: Partial<ExpConfig>) => void; disabled: boolean }) {
  const [cfg, setCfg] = useState<ExpConfig>(base);
  useEffect(() => setCfg(base), [base]);
  const baseModel = models.find((m) => m.file === base.model);
  const family = models.filter((m) => baseModel && m.architecture === baseModel.architecture && m.layers === baseModel.layers);
  const changed = (Object.keys(cfg) as (keyof ExpConfig)[]).filter((k) => String(cfg[k] ?? "") !== String(base[k] ?? ""));
  const set = (patch: Partial<ExpConfig>) => setCfg((c) => ({ ...c, ...patch }));
  const Opt = ({ k, v, label }: { k: keyof ExpConfig; v: unknown; label: string }) => (
    <button className={String(cfg[k] ?? "") === String(v ?? "") ? "on" : ""} onClick={() => set({ [k]: v } as Partial<ExpConfig>)}>{label}</button>
  );
  return (
    <div className="xchange">
      <div className="xc-row"><div><b>Weights</b><span>Same model, different <Term k="quant">quantization</Term>{family.length < 2 && <> · create more in <a href="#/models?tab=compress">Models → Compress</a></>}</span></div>
        <div className="segmented wrap">{family.map((m) => <Opt key={m.file} k="model" v={m.file} label={`${m.quant} · ${(m.size_bytes / 2 ** 30).toFixed(1)} GB`} />)}</div></div>
      <div className="xc-row"><div><b><Term k="kv">KV cache type</Term></b><span>-ctk / -ctv</span></div>
        <div className="segmented"><Opt k="kv" v="f16" label="f16" /><Opt k="kv" v="q8_0" label="q8_0" /><Opt k="kv" v="q4_0" label="q4_0" /></div></div>
      <div className="xc-row"><div><b><Term k="context">Context size</Term></b><span>-c</span></div>
        <div className="segmented">{CTX.map((c) => <Opt key={c} k="ctx" v={c} label={`${c / 1024}k`} />)}</div></div>
      <div className="xc-row"><div><b>VRAM limit</b><span>Simulates a smaller GPU (--fit-target)</span></div>
        <div className="segmented">{VRAM_SIM.map((v) => <Opt key={String(v)} k="vram_limit_mb" v={v} label={v ? `${v / 1024} GB` : "Full"} />)}</div></div>
      <div className="xc-row"><div><b><Term k="layers">GPU layers</Term></b><span>-ngl</span></div>
        <div className="segmented"><Opt k="ngl" v="auto" label="Auto" />{[24, 16, 8, 0].map((n) => <Opt key={n} k="ngl" v={n} label={String(n)} />)}</div></div>
      <div className="xc-row"><div><b><Term k="fa">Flash attention</Term></b><span>-fa</span></div>
        <div className="segmented"><Opt k="fa" v="auto" label="Auto" /><Opt k="fa" v="on" label="On" /><Opt k="fa" v="off" label="Off" /></div></div>
      <div className="xc-foot">
        <span className="row-sub">{changed.length === 0 ? "Pick one variable to change against the baseline."
          : changed.length === 1 ? "One variable changed: differences can be attributed to it." : `${changed.length} variables changed at once: effects will be confounded.`}</span>
        <div className="xc-btns">
          <button className="btn ghost" disabled={disabled} onClick={() => onRun("Repeat of baseline", base)}
            title="Runs the baseline setup again: the difference between the two shows how much results drift between sessions on this machine.">Repeat baseline</button>
          <button className="btn" disabled={disabled || changed.length === 0} onClick={() => onRun("", cfg)}><Play /> Profile this change</button>
        </div>
      </div>
    </div>
  );
}

function Board({ runs, metric, setMetric }: { runs: ExpRun[]; metric: string; setMetric: (k: string) => void }) {
  const base = runs[0];
  const groups = Array.from(new Set(METRICS.map((m) => m.group)));
  const hasData = (m: Metric) => runs.some((r) => m.get(r.results) != null);
  return (
    <div className="table-wrap xboard">
      <table className="data">
        <thead>
          <tr><th>Metric</th>{runs.map((r, i) => (
            <th key={r.id} className="r run-label" title={r.label}><span className="xdot" style={{ background: colorOf(i) }} />{nameOf(r, i)}</th>
          ))}</tr>
        </thead>
        {groups.map((g) => {
          const rows = METRICS.filter((x) => x.group === g);
          const any = rows.some(hasData);
          return (
            <tbody key={g}>
              <tr className="grp-row"><td colSpan={runs.length + 1}>{g}{g === "Energy" && !any && <span className="muted"> · not measurable here: the GPU driver reports power as N/A</span>}</td></tr>
              {rows.filter(hasData).map((x) => (
                <tr key={x.key} className={metric === x.key ? "sel" : ""} onClick={() => setMetric(x.key)} title={x.note ?? "Click to chart"}>
                  <td className="mlabel">{x.term ? <Term k={x.term}>{x.label}</Term> : x.label}{x.unit && <span className="u"> {x.unit}</span>}</td>
                  {runs.map((r, i) => {
                    const v = x.get(r.results), sd = x.sd?.(r.results);
                    const repeats = runs.slice(1).filter((q) => q.label === "Repeat of baseline" && q.id !== r.id);
                    const bv = x.get(base.results);
                    const session = Math.max(0, ...repeats.map((q) => { const qv = x.get(q.results); return qv != null && bv != null ? Math.abs(qv - bv) : 0; }));
                    const d = i === 0 ? null : delta(x, { run: r, base }, session);
                    return (
                      <td key={r.id} className="r">
                        {v == null ? "–" : (x.fmt ?? n1)(v)}{sd ? <span className="sd"> ±{(x.fmt ?? n1)(sd)}</span> : null}
                        {d && <span className={`d ${d.tone}`} title={d.title}>{d.text}</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          );
        })}
      </table>
    </div>
  );
}

function MetricBars({ runs, metric }: { runs: ExpRun[]; metric: string }) {
  const m = METRICS.find((x) => x.key === metric) ?? METRICS[0];
  const vals = runs.map((r) => m.get(r.results));
  const max = Math.max(...vals.map((v, i) => (v ?? 0) + (m.sd?.(runs[i].results) ?? 0)), 1e-9);
  return (
    <div className="xchart">
      <div className="mini-title">{m.label}{m.unit && <span className="muted"> ({m.unit}{m.better ? `, ${m.better} is better` : ""})</span>}</div>
      {runs.map((r, i) => {
        const v = vals[i], sd = m.sd?.(r.results);
        return (
          <div className="xbar-row" key={r.id} title={`${r.label}: ${v == null ? "no data" : (m.fmt ?? n1)(v)}${sd ? ` ± ${(m.fmt ?? n1)(sd)}` : ""}`}>
            <span>{nameOf(r, i)}</span>
            <div className="xbar-track">
              <div style={{ width: `${((v ?? 0) / max) * 100}%`, background: colorOf(i) }} />
              {v != null && sd ? <i style={{ left: `${(Math.max(0, v - sd) / max) * 100}%`, width: `${((2 * sd) / max) * 100}%` }} /> : null}
            </div>
            <b>{v == null ? "–" : (m.fmt ?? n1)(v)}</b>
          </div>
        );
      })}
    </div>
  );
}

/** Roofline: attainable TFLOPS = min(compute peak, bandwidth × arithmetic intensity). Log–log. */
function Roofline({ runs }: { runs: ExpRun[] }) {
  const hw0 = runs.find((r) => r.results.hw?.peaks.tflops)?.results.hw;
  const [hover, setHover] = useState<string | null>(null);
  if (!hw0?.peaks.tflops || !hw0.peaks.vram_gbps) return <div className="row-sub">Peak numbers for this GPU aren't in Blocky's table yet.</div>;
  const W = 560, H = 300, L = 46, R = 14, T = 14, B = 38;
  const x0 = Math.log10(0.5), x1 = Math.log10(10000), y0 = Math.log10(0.05), y1 = Math.log10(hw0.peaks.tflops * 2.5);
  const X = (ai: number) => L + ((Math.log10(ai) - x0) / (x1 - x0)) * (W - L - R);
  const Y = (tf: number) => T + (1 - (Math.log10(tf) - y0) / (y1 - y0)) * (H - T - B);
  const peak = hw0.peaks.tflops, bw = hw0.peaks.vram_gbps, ram = hw0.peaks.ram_gbps;
  const line = (gbps: number) => { const r = (peak * 1000) / gbps; return `M${X(0.5)},${Y((gbps * 0.5) / 1000)} L${X(Math.min(r, 10000))},${Y(Math.min(peak, (gbps * Math.min(r, 10000)) / 1000))}`; };
  const xt = [1, 10, 100, 1000, 10000], yt = [0.1, 1, 10, 100].filter((v) => Math.log10(v) <= y1);
  const pts = runs.flatMap((r, i) => {
    const h = r.results.hw;
    const out: { id: string; x: number; y: number; c: string; sq: boolean; label: string }[] = [];
    if (h?.decode?.arithmetic_intensity && h.decode.achieved_tflops)
      out.push({ id: `${r.id}d`, x: h.decode.arithmetic_intensity, y: h.decode.achieved_tflops, c: colorOf(i), sq: false, label: `${nameOf(r, i)}, decode: ${h.decode.achieved_tflops} TFLOPS` });
    if (h?.prefill?.arithmetic_intensity)
      out.push({ id: `${r.id}p`, x: h.prefill.arithmetic_intensity, y: h.prefill.achieved_tflops, c: colorOf(i), sq: true, label: `${nameOf(r, i)}, prefill: ${h.prefill.achieved_tflops} TFLOPS` });
    return out;
  });
  const hv = pts.find((p) => p.id === hover);
  const ev = (id: string) => ({ onMouseEnter: () => setHover(id), onMouseLeave: () => setHover(null) });
  return (
    <div className="xroof">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Roofline chart">
        {xt.map((v) => <g key={v}><line x1={X(v)} x2={X(v)} y1={T} y2={H - B} className="grid" /><text x={X(v)} y={H - B + 15} textAnchor="middle" className="tick">{v}</text></g>)}
        {yt.map((v) => <g key={v}><line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} className="grid" /><text x={L - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text></g>)}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="axis">FLOP per byte of weights read (log)</text>
        <text x={11} y={(T + H - B) / 2} textAnchor="middle" className="axis" transform={`rotate(-90 11 ${(T + H - B) / 2})`}>TFLOPS (log)</text>
        <path d={`${line(bw)} L${X(10000)},${Y(peak)}`} className="roof" />
        {ram && <path d={line(ram)} className="roof ram" />}
        <text x={X(25)} y={Y((bw * 25) / 1000) - 10} textAnchor="end" className="roof-label">VRAM {bw} GB/s</text>
        {ram && <text x={X(40)} y={Y((ram * 40) / 1000) + 18} className="roof-label">RAM {ram} GB/s</text>}
        <text x={W - R - 4} y={Y(peak) - 6} textAnchor="end" className="roof-label">{peak} TFLOPS peak</text>
        {pts.map((p) => p.sq
          ? <rect key={p.id} x={X(p.x) - 5.5} y={Y(p.y) - 5.5} width={11} height={11} rx={2} fill={p.c} className="pt" {...ev(p.id)} />
          : <circle key={p.id} cx={X(p.x)} cy={Y(p.y)} r={hover === p.id ? 7 : 5.5} fill={p.c} className="pt" {...ev(p.id)} />)}
      </svg>
      <div className="xlegend">
        <span><i className="dot" /> decode</span><span><i className="sq" /> prefill (512-token batch)</span>
        {runs.map((r, i) => <span key={r.id}><i className="dot" style={{ background: colorOf(i) }} />{nameOf(r, i)}</span>)}
      </div>
      <div className="row-sub">{hv ? hv.label : "Hover a point. Points under the slanted line are memory-bound, under the flat line compute-bound. The gap to the line is unused hardware."}</div>
    </div>
  );
}

/** Where each generated token's time goes: reading weights from VRAM, from system RAM, and everything else. */
function TimeSplit({ runs }: { runs: ExpRun[] }) {
  const rows = runs.map((r, i) => ({ r, i })).filter(({ r }) => r.results.hw?.decode);
  const max = Math.max(...rows.map(({ r }) => r.results.hw!.decode!.ms_per_token), 1);
  const parts = [
    { k: "vram_reads" as const, label: "Weights from VRAM, at peak bandwidth", c: "var(--series-1)" },
    { k: "ram_reads" as const, label: "Weights from system RAM", c: "var(--series-3)" },
    { k: "other" as const, label: "Everything else", c: "var(--split-other)" },
  ];
  return (
    <div className="xsplit">
      <div className="mini-title">Time per generated token <span className="muted">(ms; minimum possible weight-read time vs the rest)</span></div>
      {rows.map(({ r, i }) => {
        const s = r.results.hw!.decode!.time_split_ms, tot = r.results.hw!.decode!.ms_per_token;
        return (
          <div className="xbar-row" key={r.id}>
            <span>{nameOf(r, i)}</span>
            <div className="xbar-track stack">
              {parts.map((p) => (s[p.k] ?? 0) > 0.05 && (
                <div key={p.k} style={{ width: `${((s[p.k] ?? 0) / max) * 100}%`, background: p.c }} title={`${p.label}: ${(s[p.k] ?? 0).toFixed(1)} ms`} />
              ))}
            </div>
            <b>{tot.toFixed(1)}</b>
          </div>
        );
      })}
      <div className="xlegend">{parts.map((p) => <span key={p.k}><i className="sq" style={{ background: p.c }} />{p.label}</span>)}</div>
    </div>
  );
}

function Telemetry({ run }: { run: ExpRun }) {
  const s = run.results.series ?? [];
  if (s.length < 5) return null;
  const pts = (k: "gpu" | "mem_bw" | "power" | "vram_mb" | "sm_clock" | "cpu") => s.filter((x) => x[k] != null).map((x) => ({ x: x.t, y: k === "vram_mb" ? (x[k] as number) / 1024 : (x[k] as number) }));
  const charts = [
    { k: "gpu" as const, t: "GPU busy", u: "%", c: "var(--series-1)", max: 100 },
    { k: "mem_bw" as const, t: "Memory controller busy", u: "%", c: "var(--series-4)", max: 100 },
    { k: "sm_clock" as const, t: "SM clock", u: "MHz", c: "var(--series-3)" },
    { k: "cpu" as const, t: "CPU busy", u: "%", c: "var(--series-2)", max: 100 },
    { k: "vram_mb" as const, t: "VRAM used, all programs", u: "GB", c: "var(--series-2)" },
    { k: "power" as const, t: "GPU power", u: "W", c: "var(--series-3)" },
  ].filter((c) => pts(c.k).length > 2);
  const phases = PHASES.map((p) => { const xs = s.filter((x) => x.phase === p.id); return xs.length ? `${p.label} ${xs[0].t.toFixed(0)}–${xs[xs.length - 1].t.toFixed(0)} s` : null; }).filter(Boolean);
  return (
    <div className="xtl">
      <div className="mini-title">Sampled every 0.5 s <span className="muted">· {phases.join(" · ")}</span></div>
      <div className="tool-charts three">
        {charts.map((c) => <div key={c.k}><div className="mini-title">{c.t} <span className="muted">({c.u})</span></div>
          <LineChart compact points={pts(c.k)} yMax={c.max} height={56} color={c.c} unit={c.u} fmtX={(x) => `${x.toFixed(0)} s`} /></div>)}
      </div>
    </div>
  );
}

/* ---------- Page ---------- */

export default function ExperimentsPage({ status }: { status: Status | null }) {
  const [list, setList] = useState<Experiment[]>([]);
  const [id, setId] = useState<number | null>(null);
  const [exp, setExp] = useState<Experiment | null>(null);
  const [models, setModels] = useState<ModelFile[]>([]);
  const [newModel, setNewModel] = useState("");
  const [creating, setCreating] = useState(false);
  const [focus, setFocus] = useState<number | null>(null);
  const [metric, setMetric] = useState("tg");
  const [err, setErr] = useState<string | null>(null);

  const loadList = useCallback(() => api.experiments().then((l) => { setList(l); setId((cur) => cur ?? l[0]?.id ?? null); }).catch(() => {}), []);
  useEffect(() => {
    loadList();
    api.models().then((d) => { setModels(d.models); setNewModel(d.runtime.model ?? d.models[0]?.file ?? ""); }).catch(() => {});
  }, [loadList]);
  const load = useCallback(() => { if (id) api.experiment(id).then(setExp).catch(() => {}); }, [id]);
  useEffect(() => { setExp(null); load(); }, [load]);

  const runs = exp?.runs_list ?? [];
  const running = runs.find((r) => r.status === "running" || r.status === "queued");
  useEffect(() => {
    const t = setInterval(load, running ? 2000 : 15000);
    return () => clearInterval(t);
  }, [running, load]);

  const done = useMemo(() => runs.filter((r) => r.status === "done"), [runs]);
  const baseline = done[0];
  const shown = done.find((r) => r.id === focus) ?? done[done.length - 1];

  const create = async () => {
    setErr(null);
    try {
      const m = models.find((x) => x.file === newModel);
      const { id: nid } = await api.experimentCreate(`${m?.name ?? newModel}`, newModel);
      setCreating(false); setId(nid); loadList();
    } catch (e) { setErr((e as Error).message); }
  };
  const runProfile = async (label: string, cfg: Partial<ExpConfig>) => {
    if (!exp) return;
    setErr(null);
    try { await api.experimentRun(exp.id, label, cfg); load(); window.scrollTo({ top: 0 }); } catch (e) { setErr((e as Error).message); }
  };
  const L = exp?.limits;

  return (
    <div className="page">
      <div className="page-inner wide">
        <h1 className="page-title">Experiments</h1>
        <p className="page-sub">One model, one variable at a time. Every run measures the same things the same way, and the noise is shown, so you can tell real differences from luck.</p>

        <div className="xbar">
          <select className="field select" value={id ?? ""} onChange={(e) => { setId(Number(e.target.value)); setFocus(null); }} disabled={list.length === 0}>
            {list.length === 0 && <option>No experiments yet</option>}
            {list.map((e) => <option key={e.id} value={e.id}>{e.name} · {e.runs} run{e.runs === 1 ? "" : "s"}</option>)}
          </select>
          <button className="btn small ghost" onClick={() => setCreating(!creating)}><Plus /> New experiment</button>
          {exp && <>
            <a className="btn small ghost" href={`/api/experiments/${exp.id}/export.csv`}><Download /> CSV</a>
            <a className="btn small ghost" href={`/api/experiments/${exp.id}/export.json`}><Download /> JSON</a>
            <button className="icon-btn danger" title="Delete experiment" onClick={() => { if (confirm(`Delete "${exp.name}" and all its runs?`)) api.experimentDelete(exp.id).then(() => { setId(null); setExp(null); loadList(); }); }}><Trash2 /></button>
          </>}
        </div>

        {(creating || list.length === 0) && (
          <div className="group pad xnew">
            <b>Which model do you want to study?</b>
            <div className="row-sub">An experiment follows one model: a baseline profile first, then variations of its setup.</div>
            <div className="xnew-row">
              <select className="field select" value={newModel} onChange={(e) => setNewModel(e.target.value)}>
                {models.map((m) => <option key={m.file} value={m.file}>{m.name ?? m.file} · {m.quant}</option>)}
              </select>
              <button className="btn" onClick={create}><FlaskConical /> Start experiment</button>
            </div>
          </div>
        )}
        {err && <div className="error" style={{ marginTop: 12 }}><AlertCircle /> {err}</div>}

        {exp && L && (
          <>
            <div className="xhwline"><Cpu size={14} />
              <span>{L.gpu?.replace("NVIDIA GeForce ", "") ?? "No GPU"}</span>
              <span>VRAM {(L.vram_mb / 1024).toFixed(0)} GB · {L.gpu_bandwidth_gbps ?? "?"} GB/s</span>
              <span>{L.gpu_tflops ?? "?"} TFLOPS <span className="muted">({L.gpu_tflops_kind ?? "unknown"})</span></span>
              <span>RAM {(L.ram_mb / 1024).toFixed(0)} GB · {L.ram_bandwidth_gbps ?? "?"} GB/s measured</span>
              <span>{L.cpu}</span>
            </div>

            {running && <Running run={running} onStop={() => api.experimentCancel(running.id).then(load)} />}

            <div className="group-label">1 · Baseline</div>
            {baseline ? (
              <div className="group pad">
                <div className="xcap"><b>{baseline.config.model.replace(".gguf", "")}</b><span className="row-sub">ctx {baseline.config.ctx} · KV {baseline.config.kv} · ngl {String(baseline.config.ngl)} · fa {baseline.config.fa}</span></div>
                <Profile run={baseline} />
              </div>
            ) : !running && (
              <div className="group pad xcta">
                <div><b>Measure the starting point</b>
                  <span className="row-sub">Load (memory layout from llama.cpp's own log) → MMLU-Pro, 560 questions, 5-shot log-likelihood → llama-bench decode and prefill at 0 and 4k context, 5 repetitions (median, robust spread) → perplexity on 16k tokens of WikiText-2, saving logits for the KL divergence of later runs. GPU and CPU telemetry throughout.</span></div>
                <button className="btn" disabled={!!status?.busy} onClick={() => runProfile("Baseline", {})}><Play /> Run baseline</button>
              </div>
            )}

            {baseline && (
              <>
                <div className="group-label">2 · Change one variable</div>
                <div className="group pad"><ChangePanel base={baseline.config} models={models} onRun={runProfile} disabled={!!running || !!status?.busy} /></div>

                <div className="group-label">3 · Results</div>
                <div className="group pad">
                  <Board runs={done} metric={metric} setMetric={setMetric} />
                  <div className="row-sub xnote">Speeds are the median of the repetitions; ± is the robust spread (1.4826 × median absolute deviation). A difference is colored only when it is bigger than the noise: 2σ for speeds, a paired McNemar test (p &lt; 0.05) for MMLU-Pro. Grey = not significant. If you add a “Repeat baseline” run, changes must also exceed twice the drift between the two baseline runs (on a desktop PC, other apps share the GPU and add noise). KL divergence measures how far each token's probability distribution drifted from the baseline's.</div>
                  {done.length > 1 && <MetricBars runs={done} metric={metric} />}
                  {runs.filter((r) => !["done", "running", "queued"].includes(r.status)).map((r) => (
                    <div key={r.id} className="xfailed"><AlertCircle size={14} /> “{r.label}” {r.status}{r.error ? `: ${r.error}` : ""}
                      <button className="icon-btn danger" onClick={() => api.experimentDeleteRun(r.id).then(load)}><Trash2 /></button></div>
                  ))}
                </div>

                <div className="group-label">4 · Hardware utilization</div>
                <div className="group pad xhw-grid">
                  <div>
                    <div className="mini-title">Roofline <span className="muted">(achieved compute vs the GPU's memory and compute limits)</span></div>
                    <Roofline runs={done} />
                  </div>
                  <div className="xside">
                    <TimeSplit runs={done} />
                    <div className="xlimiters">
                      <div className="mini-title">Binding constraint for decode</div>
                      {done.map((r, i) => r.results.hw?.limiter && (
                        <div key={r.id} className="xlim"><span className="xdot" style={{ background: colorOf(i) }} /><div><b>{r.results.hw.limiter.what}</b><span>{r.results.hw.limiter.evidence}</span></div></div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="group-label history-head">
                  <span>5 · Telemetry</span>
                  <select className="field select small-select" value={shown?.id ?? ""} onChange={(e) => setFocus(Number(e.target.value))}>
                    {done.map((r, i) => <option key={r.id} value={r.id}>{nameOf(r, i)}</option>)}
                  </select>
                </div>
                <div className="group pad">{shown && <Telemetry run={shown} />}
                  {!shown?.results.hw?.clocks.power_avg_w && <div className="row-sub xnote">Power: this GPU driver (NVIDIA 610.88) reports power draw as “N/A”, on Windows too, so energy can't be measured on this machine. On a dedicated Linux box with NVIDIA's driver, nvidia-smi normally reports it, and Blocky reads CPU package power from RAPL.</div>}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
