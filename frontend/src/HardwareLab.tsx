import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { AlertCircle, Cpu, Download, Loader2, Play, SlidersHorizontal, Square } from "lucide-react";
import { api, HwCatalog, HwChip, HwParams, HwResult, HwStat, TorchJob, TorchStatus } from "./api";
import { LineChart, Term, useMode } from "./ui";
import "./hwlab.css";

/** How a chip computes, from device physics to the numbers the lab reports. */
const EXPLAIN = String.raw`**A weight is a pair of devices.** An analog chip stores each weight as the difference of two conductances, $w \propto G^{+} - G^{-}$. An input arrives as a voltage $V$, and each device passes a current $I = G\,V$ (Ohm's law).

**A column computes a dot product.** All devices on a column share one wire, so their currents add up (Kirchhoff's current law):

$$I_j = \sum_i \left(G^{+}_{ij} - G^{-}_{ij}\right) V_i$$

One column is one output of a matrix product, computed where the weights are stored: they never travel to a processor, which is where much of a normal chip's energy goes.

**Converters connect it to the digital world.** A DAC turns each input into a voltage in $2^{b}$ steps; an ADC reads each column's current back over a fixed range (larger values clip) in $2^{b}$ steps. Large matrices are split into tiles of, say, 512 rows, and the tiles' outputs are added digitally.

**Real devices are imperfect.** Programming lands near the target: $G = G_{\text{target}} + \sigma_{\text{prog}}(G)\,\varepsilon$, with $\varepsilon \sim \mathcal{N}(0,1)$. Phase-change memory then drifts, $G(t) = G_{\text{prog}}\left(\tfrac{t+t_0}{t_0}\right)^{-\nu}$ with $\nu \approx 0.05$, and every read fluctuates. Some devices are stuck at $0$ or at $G_{\max}$. Drift compensation rescales each tile by the average conductance it lost.

**Digital chips round numbers instead.** INT8 keeps 255 levels with one scale per row; FP8 (E4M3) has 3 mantissa bits and a wide range; MXFP4 gives every 32 values a shared power-of-two scale and 4-bit values $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$.

**What the lab measures.** The same text runs through the exact model and through the model on the chip: KL divergence per token (0 = identical predictions), how often the top token agrees, and perplexity. Each trial is a separately manufactured chip, so the spread over trials is chip-to-chip variation.`;

const SWEEP_HINT: Record<string, string> = {
  adc_bits: "4, 5, 6, 7, 8, 10", dac_bits: "4, 5, 6, 7, 8, 10", levels: "4, 8, 16, 32, 64", tile_rows: "64, 128, 256, 512, 1024",
  t_seconds: "0, 3600, 86400, 2592000, 31557600", out_noise: "0, 0.01, 0.02, 0.05, 0.1", "stuck.zero": "0, 0.001, 0.005, 0.01, 0.02",
  "prog_noise.sigma": "0, 0.01, 0.02, 0.05, 0.1", "read_noise.sigma": "0, 0.01, 0.02, 0.05", "drift.nu": "0, 0.02, 0.05, 0.1",
};
const FORMAT_LABEL: Record<string, string> = { fp16: "FP16 (exact)", int8: "INT8", int4: "INT4", fp8_e4m3: "FP8 E4M3", mxfp4: "MXFP4" };
const BITS = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16];

type Noise = { model: "pcm" | "constant"; sigma?: number; nu?: number; nu_sd?: number } | null;

const pct = (v: number | null | undefined, d = 2) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
const num = (v: number) => (v !== 0 && Math.abs(v) < 0.001 ? v.toExponential(1) : v.toFixed(4));
const spread = (s: HwStat | undefined, f: (v: number) => string = num) => (!s ? "–" : s.sd != null ? `${f(s.mean)} ± ${f(s.sd)}` : f(s.mean));
const timeLabel = (s: number) => (s === 0 ? "right after programming" : s < 3600 ? `${+(s / 60).toFixed(1)} min` : s < 86400 ? `${+(s / 3600).toFixed(1)} h`
  : s < 31557600 ? `${+(s / 86400).toFixed(1)} days` : `${+(s / 31557600).toFixed(1)} years`);
const params = (j: TorchJob) => j.params as unknown as HwParams;
const result = (j: TorchJob) => (j.status === "done" ? (j.results as unknown as HwResult) : null);
const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));

export default function HardwareLab() {
  const mode = useMode();
  const [cat, setCat] = useState<HwCatalog | null>(null);
  const [status, setStatus] = useState<TorchStatus | null>(null);
  const [runs, setRuns] = useState<TorchJob[]>([]);
  const [f, setF] = useState({ checkpoint: "", preset: "pcm-1h", trials: 5, chunks: 16, label: "" });
  const [chip, setChip] = useState<HwChip | null>(null);
  const [sweep, setSweep] = useState({ key: "", values: "" });
  const [sel, setSel] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.hwCatalog().then(setCat).catch((e) => setErr((e as Error).message));
    api.torchStatus().then(setStatus).catch(() => {});
  }, []);
  useEffect(() => {
    let stop = false;
    const tick = () => api.hwRuns().then((r) => { if (!stop) setRuns(r); }).catch(() => {});
    tick();
    const t = setInterval(tick, 4000);
    return () => { stop = true; clearInterval(t); };
  }, []);
  const preset = cat?.presets.find((p) => p.id === f.preset) ?? null;
  useEffect(() => { if (preset) setChip(clone(preset.chip)); }, [preset]);
  useEffect(() => { if (!f.checkpoint && status?.checkpoints[0]) setF((x) => ({ ...x, checkpoint: status.checkpoints[0].id })); }, [status]); // eslint-disable-line react-hooks/exhaustive-deps

  const changed = !!(chip && preset && JSON.stringify(chip) !== JSON.stringify(preset.chip));
  const body = () => ({ checkpoint: f.checkpoint, preset: f.preset, trials: f.trials, chunks: f.chunks, label: f.label, ...(changed ? { chip } : {}) });
  const act = async (fn: () => Promise<number>) => {
    setErr(null);
    setBusy(true);
    try { setSel(await fn()); setRuns(await api.hwRuns()); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  const run = () => act(async () => (await api.hwRun(body())).id);
  const runSweep = () => act(async () => {
    const values = sweep.values.split(/[,\s]+/).filter(Boolean).map(Number);
    return (await api.hwSweep({ ...body(), vary: { key: sweep.key, values } })).ids[0];
  });
  const set = (k: string, v: unknown) => setChip((c) => (c ? { ...c, [k]: v } : c));

  const shown = runs.find((j) => j.id === sel) ?? runs[0];
  const checkpoints = status?.checkpoints ?? [];
  const groups: [string, string][] = [["check", "Checks"], ["digital", "Digital chips"], ["analog", "Analog in-memory chips"]];
  return (
    <div className="page hw"><div className="page-inner hw-inner">
      <h1 className="page-title"><Cpu /> Hardware lab</h1>
      <p className="page-sub">Run a model on a simulated chip and measure what the chip's arithmetic does to it: analog in-memory computing with
        measured device physics, or digital number formats. Each run compares the model on the chip with the exact model, token by token, on
        several simulated chips.</p>

      <details className="hw-explain">
        <summary>How a chip computes: from the physics of a device to the numbers below</summary>
        <div className="md"><ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{EXPLAIN}</ReactMarkdown></div>
      </details>

      <section className="ws">
        <div className="ws-grid">
          <label><span>Model (Hugging Face checkpoint)</span>
            <select className="field select" value={f.checkpoint} onChange={(e) => setF({ ...f, checkpoint: e.target.value })}>
              {checkpoints.map((c) => <option key={c.id} value={c.id}>{c.repo}</option>)}</select>
            {!checkpoints.length && <em>No checkpoints on this device yet: download one under <a href="#/internals">Internals</a> (a small one, such as a 0.5–1B model, runs fastest).</em>}
          </label>
          <label><span>Chips simulated (trials) · text chunks</span>
            <div className="ft-row">
              <input className="field" type="number" min={1} max={20} value={f.trials} onChange={(e) => setF({ ...f, trials: Number(e.target.value) })} />
              <input className="field" type="number" min={2} max={64} value={f.chunks} onChange={(e) => setF({ ...f, chunks: Number(e.target.value) })} />
            </div>
            <em>Each trial is a separately manufactured chip (new device noise). Chips without randomness run once. Chunks are 512 tokens of held-out text.</em>
          </label>
          <label><span>Name (optional)</span><input className="field" value={f.label} maxLength={60} onChange={(e) => setF({ ...f, label: e.target.value })} /></label>
        </div>

        {cat && groups.map(([g, title]) => (
          <div key={g}>
            <div className="mini-title">{title}</div>
            <div className="hw-presets">
              {cat.presets.filter((p) => p.group === g).map((p) => (
                <button key={p.id} className={`hw-preset ${f.preset === p.id ? "sel" : ""}`} onClick={() => setF({ ...f, preset: p.id })}>
                  <b>{p.label}{p.illustrative && <span className="hw-tag">illustrative</span>}</b>
                  <span>{p.about}</span>
                  {p.sources?.map((s) => <small key={s}>{s}</small>)}
                </button>
              ))}
            </div>
          </div>
        ))}

        {chip && cat && (
          <details className="hw-settings" open={mode === "pro" ? true : undefined}>
            <summary><SlidersHorizontal size={14} /> Chip settings{changed && <span className="hw-tag">changed</span>}
              {changed && <button className="btn small ghost" onClick={(e) => { e.preventDefault(); if (preset) setChip(clone(preset.chip)); }}>Reset</button>}</summary>
            {chip.kind === "analog" ? <AnalogSettings chip={chip} set={set} cat={cat} /> : <DigitalSettings chip={chip} set={set} cat={cat} />}
          </details>
        )}

        <div className="xc-btns">
          <button className="btn" disabled={!f.checkpoint || !chip || busy} onClick={run}><Play /> Run on this chip</button>
          {chip?.kind === "analog" && cat && (
            <span className="hw-sweep-form">
              or sweep
              <select className="field select" value={sweep.key} onChange={(e) => setSweep({ key: e.target.value, values: SWEEP_HINT[e.target.value] ?? "" })}>
                <option value="">a setting…</option>
                {cat.sweep_keys.map((k) => <option key={k.key} value={k.key}>{k.label}</option>)}
              </select>
              {sweep.key && <>
                <input className="field" value={sweep.values} onChange={(e) => setSweep({ ...sweep, values: e.target.value })} title="2 to 12 values, separated by commas" />
                <button className="btn ghost" disabled={!f.checkpoint || busy} onClick={runSweep}><Play /> Run sweep</button>
              </>}
            </span>
          )}
        </div>
        <div className="row-sub">Runs use the GPU: the chat model pauses meanwhile, and GPU safety stops the run if the GPU gets too hot.</div>
        {err && <div className="error"><AlertCircle /> {err}</div>}
      </section>

      {runs.length > 0 && (
        <section>
          <div className="table-wrap"><table className="data">
            <thead><tr><th>Model</th><th>Chip</th><th>Chips</th><th><Term k="kld">KL divergence</Term></th><th>Same top token</th>
              <th>Perplexity</th><th title="Effective weights on the chip vs the exact weights: ||W_chip − W|| / ||W||">Weight error</th>
              <th title="Each matrix's outputs on the chip vs exact, on the same inputs">Output error</th><th /></tr></thead>
            <tbody>{runs.map((j) => {
              const p = params(j), r = result(j);
              return (
                <tr key={j.id} className={shown?.id === j.id ? "sel" : ""} onClick={() => setSel(j.id)} style={{ cursor: "pointer" }}>
                  <td>{p.checkpoint}</td>
                  <td>{p.label || p.preset_label}{p.sweep && <span className="hw-tag">{p.sweep.label} = {p.sweep.key === "t_seconds" ? timeLabel(p.sweep.value) : p.sweep.value}</span>}</td>
                  {r ? <>
                    <td>{r.trials}</td>
                    <td title={r.summary.kld.ci95 != null ? `95% interval over chips: ± ${num(r.summary.kld.ci95)}` : "one chip (nothing random)"}>{spread(r.summary.kld)}</td>
                    <td>{spread(r.summary.same_top, (v) => `${v.toFixed(1)}%`)}</td>
                    <td>{spread(r.summary.ppl, (v) => v.toFixed(3))} <span className="muted">(exact {r.ppl_base.toFixed(3)})</span></td>
                    <td>{pct(r.weight_error)}</td><td>{pct(r.output_error)}</td>
                  </> : <td colSpan={6} className="muted">{j.status === "running" ? <><Loader2 className="spin" size={13} /> {j.step ?? "starting"}…</> : j.status}{j.error ? `: ${j.error}` : ""}</td>}
                  <td onClick={(e) => e.stopPropagation()}>
                    {r && <a className="btn small ghost" href={`/api/hwlab/runs/${j.id}/csv`} title="Download every chip's results as CSV"><Download size={13} /></a>}
                    {(j.status === "running" || j.status === "queued") && <button className="btn small ghost" onClick={() => api.torchCancel(j.id)}><Square size={13} /></button>}
                  </td>
                </tr>
              );
            })}</tbody>
          </table></div>
          {shown && <RunDetail j={shown} runs={runs} cat={cat} />}
        </section>
      )}
    </div></div>
  );
}

function AnalogSettings({ chip, set, cat }: { chip: HwChip; set: (k: string, v: unknown) => void; cat: HwCatalog }) {
  const t = Number(chip.t_seconds);
  const stuck = chip.stuck as { zero: number; max: number };
  const range = chip.adc_range as "auto" | number | null;
  return (
    <div className="ws-grid">
      <label><span>Time after programming</span>
        <select className="field select" value={t} onChange={(e) => set("t_seconds", Number(e.target.value))}>
          {cat.times.map((o) => <option key={o.seconds} value={o.seconds}>{o.label}</option>)}
          {!cat.times.some((o) => o.seconds === t) && <option value={t}>{timeLabel(t)}</option>}</select>
        <em><Term k="drift">Drift</Term> and read noise grow with time; drift and read noise only act if they're on below.</em></label>
      <label><span>Programming noise</span>
        <NoisePick value={chip.prog_noise as Noise} onChange={(v) => set("prog_noise", v)} params={[{ key: "sigma", label: "% of G_max", scale: 100, def: 0.02 }]} />
        <em>How far a device lands from its target <Term k="conductance">conductance</Term>. PCM: IBM's measured devices (about 1% of G_max at 0, 4% at G_max).</em></label>
      <label><span>Drift</span>
        <NoisePick value={chip.drift as Noise} onChange={(v) => set("drift", v)}
          params={[{ key: "nu", label: "ν", scale: 1, def: 0.05 }, { key: "nu_sd", label: "± ν", scale: 1, def: 0 }]} />
        <em>G(t) = G₀·((t + 20 s)/20 s)^−ν per device. PCM: ν around 0.05–0.1, larger for small conductances.</em></label>
      <label><span>Read noise</span>
        <NoisePick value={chip.read_noise as Noise} onChange={(v) => set("read_noise", v)} params={[{ key: "sigma", label: "% of G", scale: 100, def: 0.01 }]} />
        <em>1/f fluctuation accumulated up to the read. PCM: about 3.7% of G at G_max, more for small conductances.</em></label>
      <label className="ws-check"><input type="checkbox" checked={!!chip.drift_compensation} onChange={(e) => set("drift_compensation", e.target.checked)} />
        <span><b>Drift compensation</b><em>Rescales each tile by its average conductance loss, measured on the chip.</em></span></label>
      <label><span>Stuck devices: at 0 · at G_max (%)</span>
        <div className="ft-row">
          <input className="field" type="number" step="any" min={0} max={20} value={+(stuck.zero * 100).toFixed(4)} onChange={(e) => set("stuck", { ...stuck, zero: Number(e.target.value) / 100 })} />
          <input className="field" type="number" step="any" min={0} max={20} value={+(stuck.max * 100).toFixed(4)} onChange={(e) => set("stuck", { ...stuck, max: Number(e.target.value) / 100 })} />
        </div><em>Devices that never change, from manufacturing or wear.</em></label>
      <label><span>Levels per device</span>
        <select className="field select" value={Number(chip.levels)} onChange={(e) => set("levels", Number(e.target.value))}>
          {[0, 2, 4, 8, 16, 32, 64, 256].map((v) => <option key={v} value={v}>{v ? `${v} levels (${Math.log2(v)} bits per device)` : "continuous"}</option>)}</select>
        <em>How many conductances a device can be set to reliably.</em></label>
      <label><span>Tile size (rows)</span>
        <select className="field select" value={Number(chip.tile_rows)} onChange={(e) => set("tile_rows", Number(e.target.value))}>
          {[64, 128, 256, 512, 1024, 2048, 0].map((v) => <option key={v} value={v}>{v ? `${v} rows` : "one tile per matrix"}</option>)}</select>
        <em>A <Term k="crossbar">crossbar</Term> has a fixed size; larger matrices span several tiles, added digitally.</em></label>
      <label><span>Input <Term k="dac">DAC</Term> · output <Term k="adc">ADC</Term> bits</span>
        <div className="ft-row">
          <select className="field select" value={Number(chip.dac_bits)} onChange={(e) => set("dac_bits", Number(e.target.value))}>
            {BITS.filter((b) => b !== 2 && b !== 3).map((b) => <option key={b} value={b}>{b ? `${b}-bit DAC` : "ideal DAC"}</option>)}</select>
          <select className="field select" value={Number(chip.adc_bits)} onChange={(e) => { const b = Number(e.target.value); set("adc_bits", b); if (b && range == null) set("adc_range", "auto"); }}>
            {BITS.filter((b) => b !== 2 && b !== 3).map((b) => <option key={b} value={b}>{b ? `${b}-bit ADC` : "ideal ADC"}</option>)}</select>
        </div><em>Inputs are scaled per token so the largest uses the full range.</em></label>
      <label><span>ADC range</span>
        <div className="ft-row">
          <select className="field select" value={range == null ? "none" : range === "auto" ? "auto" : "fixed"}
            onChange={(e) => set("adc_range", e.target.value === "auto" ? "auto" : e.target.value === "none" ? null : 10)}>
            <option value="auto">calibrated per tile</option><option value="fixed">fixed</option>
            {!Number(chip.adc_bits) && <option value="none">no clipping</option>}</select>
          {typeof range === "number" && <input className="field" type="number" step="any" min={0.01} value={range} onChange={(e) => set("adc_range", Number(e.target.value))} />}
        </div><em>Calibrated: 99.99th percentile of each tile's outputs on calibration text (WikiText-2 train). Fixed: in units of one full-scale weight × one full-scale input.</em></label>
      <label><span>Output noise (σ, full-scale units)</span>
        <input className="field" type="number" step="any" min={0} max={10} value={Number(chip.out_noise)} onChange={(e) => set("out_noise", Number(e.target.value))} />
        <em>Fresh Gaussian noise on every tile output, every time: thermal noise of the analog circuits.</em></label>
      <label><span>G_max (µS) · weight scaling</span>
        <div className="ft-row">
          <input className="field" type="number" step="any" min={0.1} max={1000} value={Number(chip.g_max_us)} onChange={(e) => set("g_max_us", Number(e.target.value))} />
          <select className="field select" value={String(chip.weight_scaling)} onChange={(e) => set("weight_scaling", e.target.value)}>
            <option value="column">per column and tile</option><option value="tile">per tile</option></select>
        </div><em>The largest weight of each column (or tile) maps to G_max. PCM's measured noise was fitted at 25 µS.</em></label>
    </div>
  );
}

function NoisePick({ value, onChange, params: ps }: { value: Noise; onChange: (v: Noise) => void; params: { key: "sigma" | "nu" | "nu_sd"; label: string; scale: number; def: number }[] }) {
  return (
    <div className="ft-row">
      <select className="field select" value={value?.model ?? "off"}
        onChange={(e) => onChange(e.target.value === "off" ? null : e.target.value === "pcm" ? { model: "pcm" } : { model: "constant", ...Object.fromEntries(ps.map((p) => [p.key, p.def])) })}>
        <option value="off">off</option><option value="pcm">PCM (measured)</option><option value="constant">constant</option></select>
      {value?.model === "constant" && ps.map((p) => (
        <span key={p.key} className="hw-num"><input className="field" type="number" step="any" min={0} value={+((value[p.key] ?? p.def) * p.scale).toFixed(6)}
          onChange={(e) => onChange({ ...value, [p.key]: Number(e.target.value) / p.scale })} /> {p.label}</span>
      ))}
    </div>
  );
}

function DigitalSettings({ chip, set, cat }: { chip: HwChip; set: (k: string, v: unknown) => void; cat: HwCatalog }) {
  const w = chip.weights as { format: string; group: number };
  const x = chip.inputs as { format: string };
  return (
    <div className="ws-grid">
      <label><span>Weights</span>
        <div className="ft-row">
          <select className="field select" value={w.format} onChange={(e) => set("weights", { ...w, format: e.target.value })}>
            {cat.weight_formats.map((v) => <option key={v} value={v}>{FORMAT_LABEL[v] ?? v}</option>)}</select>
          {(w.format === "int8" || w.format === "int4" || w.format === "fp8_e4m3") && (
            <select className="field select" value={w.group} onChange={(e) => set("weights", { ...w, group: Number(e.target.value) })}>
              {[0, 32, 64, 128, 256].map((g) => <option key={g} value={g}>{g ? `groups of ${g}` : "one scale per output"}</option>)}</select>)}
        </div><em>Smaller groups follow the weights more closely but store more scales.</em></label>
      <label><span>Inputs (activations)</span>
        <select className="field select" value={x.format} onChange={(e) => set("inputs", { format: e.target.value })}>
          {cat.input_formats.map((v) => <option key={v} value={v}>{FORMAT_LABEL[v] ?? v}</option>)}</select>
        <em>One scale per token (MXFP4: per 32 values). Products are added exactly.</em></label>
    </div>
  );
}

function RunDetail({ j, runs, cat }: { j: TorchJob; runs: TorchJob[]; cat: HwCatalog | null }) {
  const p = params(j), r = result(j);
  const sources = cat?.presets.find((x) => x.id === p.preset)?.sources ?? [];
  const sweepRuns = p.sweep ? runs.filter((x) => params(x).sweep?.id === p.sweep!.id).sort((a, b) => params(a).sweep!.value - params(b).sweep!.value) : [];
  if (j.status === "error") return <div className="error"><AlertCircle /> {j.error}</div>;
  return (
    <div className="hw-detail">
      {sweepRuns.length > 1 && <SweepChart runs={sweepRuns} />}
      {r && <>
        <div className="row-sub">On the chip: {r.on_chip.matrices} matrices ({(r.on_chip.weights / 1e6).toFixed(0)}M weights). Exact digital: {r.digital}.
          {" "}{r.tokens.toLocaleString()} tokens of held-out text (WikiText-2 test) per chip{r.calibration ? `; ADC ranges calibrated on ${r.calibration.tokens.toLocaleString()} tokens of WikiText-2 train` : ""}.
          {" "}{r.deterministic ? "Nothing random on this chip: one run." : `${r.trials} chips: ± is the standard deviation over chips.`} {r.seconds} s on {r.device}.</div>
        <div className="hw-cols">
          <div>
            <div className="mini-title">Output error by matrix type (all layers, all chips)</div>
            <div className="hw-bars">{Object.entries(r.per_kind).map(([k, e]) => {
              const max = Math.max(1e-9, ...Object.values(r.per_kind).map((v) => v.output_error ?? 0));
              return <div key={k} className="hw-bar"><span>{k}</span><i style={{ width: `${((e.output_error ?? 0) / max) * 100}%` }} /><b>{pct(e.output_error)}</b>
                <small title="Weight error">{pct(e.weight_error)}</small></div>;
            })}</div>
          </div>
          <div>
            <div className="mini-title">Output error by layer</div>
            <LineChart points={r.per_layer.filter((l) => l.layer != null && l.output_error != null).map((l) => ({ x: l.layer!, y: l.output_error! * 100 }))}
              unit="%" xLabel="layer" height={140} />
          </div>
        </div>
        {r.per_trial.length > 1 && (
          <div className="table-wrap"><table className="data compact">
            <thead><tr><th>Chip</th><th>KL divergence</th><th>KLD p99</th><th>Same top token</th><th>Perplexity</th></tr></thead>
            <tbody>{r.per_trial.map((t) => <tr key={t.trial}><td>#{t.trial}</td><td>{num(t.kld)} <span className="muted">± {num(t.kld_sd)}</span></td>
              <td>{num(t.kld_p99)}</td><td>{t.same_top.toFixed(1)}%</td><td>{t.ppl.toFixed(3)}</td></tr>)}</tbody>
          </table></div>
        )}
        {r.not_modeled.length > 0 && <div className="row-sub"><b>Not modeled:</b> {r.not_modeled.join("; ")}.</div>}
        {sources.length > 0 && <div className="row-sub"><b>Source:</b> {sources.join("; ")}.</div>}
      </>}
    </div>
  );
}

/** One swept setting: KL divergence per value (mean over chips, bar = 95% interval), on a log scale. */
function SweepChart({ runs }: { runs: TorchJob[] }) {
  const s = params(runs[0]).sweep!;
  const pts = runs.map((j) => ({ v: params(j).sweep!.value, r: result(j) })).filter((x) => x.r);
  const W = 640, H = 220, L = 56, B = 34, T = 14;
  if (!pts.length) return <div className="row-sub">Sweep of {s.label}: waiting for the first result…</div>;
  const lo = (p: (typeof pts)[number]) => Math.max(1e-7, p.r!.summary.kld.mean - (p.r!.summary.kld.ci95 ?? 0));
  const hi = (p: (typeof pts)[number]) => p.r!.summary.kld.mean + (p.r!.summary.kld.ci95 ?? 0);
  const ymin = Math.log10(Math.min(...pts.map(lo))), ymax = Math.log10(Math.max(...pts.map(hi)));
  const span = Math.max(0.5, ymax - ymin), y0 = ymin - span * 0.08, y1 = ymax + span * 0.08;
  const y = (v: number) => T + (H - T - B) * (1 - (Math.log10(Math.max(v, 1e-7)) - y0) / (y1 - y0));
  const x = (i: number) => L + ((W - L - 16) * (i + 0.5)) / runs.length;
  const ticks = Array.from({ length: Math.floor(y1) - Math.ceil(y0) + 1 }, (_, i) => Math.ceil(y0) + i);
  const label = (v: number) => (s.key === "t_seconds" ? timeLabel(v) : String(v));
  return (
    <div className="hw-sweep">
      <div className="mini-title">Sweep: KL divergence by {s.label} ({pts.length} of {s.n} done · bars: 95% interval over chips) · <a href={`/api/hwlab/sweeps/${s.id}/csv`}>CSV</a></div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`KL divergence by ${s.label}`}>
        {ticks.map((t) => <g key={t}><line x1={L} x2={W - 8} y1={y(10 ** t)} y2={y(10 ** t)} className="grid" />
          <text x={L - 6} y={y(10 ** t) + 4} textAnchor="end">1e{t}</text></g>)}
        {runs.map((j, i) => {
          const v = params(j).sweep!.value, r = result(j);
          return <g key={j.id}>
            <text x={x(i)} y={H - 12} textAnchor="middle">{label(v)}</text>
            {r && <>
              {r.summary.kld.ci95 != null && <line x1={x(i)} x2={x(i)} y1={y(lo({ v, r }))} y2={y(hi({ v, r }))} className="err" />}
              <circle cx={x(i)} cy={y(r.summary.kld.mean)} r={4} className="pt"><title>{`${s.label} = ${label(v)}: KLD ${spread(r.summary.kld)}`}</title></circle>
            </>}
          </g>;
        })}
      </svg>
    </div>
  );
}
