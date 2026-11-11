import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Brain, Clock, Cpu, FlaskConical, Gauge, Layers, Sparkles, Thermometer, Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { api, BenchRun, CatalogItem, MemoryMapData, ModelFile, RecentStat, Runtime, Status, SystemStats } from "./api";
import MemoryMap from "./MemoryMap";
import { energyVerdict, fmtMB, LineChart, Point, Pro, speedVerdict, Term, waitVerdict } from "./ui";
import type { Page } from "./App";

type Tip = { icon: LucideIcon; color: string; title: string; text: string; page: Page; cta: string };

function Hero({ icon: Icon, color, label, term, value, unit, verdict, tone, children }: {
  icon: LucideIcon; color: string; label: string; term?: string; value: string; unit?: string;
  verdict?: string; tone?: string; children?: React.ReactNode;
}) {
  return (
    <div className="hero" style={{ "--c": color } as React.CSSProperties}>
      <div className="hero-head"><span className="hero-icon"><Icon /></span>{term ? <Term k={term}>{label}</Term> : label}</div>
      <div className="hero-value">{value}{unit && <small>{unit}</small>}</div>
      {verdict && <div className={`verdict tone-${tone ?? "meh"}`}>{verdict}</div>}
      {children}
    </div>
  );
}

export default function Dashboard({ status, go }: { status: Status | null; go: (p: Page) => void }) {
  const [mm, setMm] = useState<MemoryMapData | null>(null);
  const [recent, setRecent] = useState<RecentStat[]>([]);
  const [runs, setRuns] = useState<BenchRun[]>([]);
  const [lib, setLib] = useState<{ models: ModelFile[]; catalog: CatalogItem[]; runtime: Runtime } | null>(null);
  const [memCount, setMemCount] = useState<number | null>(null);
  const [hist, setHist] = useState<{ t: number; s: SystemStats }[]>([]);
  const t0 = useRef(Date.now());

  useEffect(() => {
    const loadSlow = () => {
      api.memoryMap().then(setMm).catch(() => {});
      api.recentStats(40).then(setRecent).catch(() => {});
      api.benchRuns().then(setRuns).catch(() => {});
      api.models().then((d) => setLib({ models: d.models, catalog: d.catalog, runtime: d.runtime })).catch(() => {});
      api.memories().then((m) => setMemCount(m.length)).catch(() => {});
    };
    loadSlow();
    const slow = setInterval(loadSlow, 15000);
    const fast = setInterval(() => {
      api.system().then((s) => setHist((h) => [...h.slice(-89), { t: (Date.now() - t0.current) / 1000, s }])).catch(() => {});
    }, 1000);
    return () => { clearInterval(slow); clearInterval(fast); };
  }, []);

  const last = recent[recent.length - 1];
  const speed = last?.predicted_per_second ?? null;
  const sv = speedVerdict(speed);
  const ttfts = recent.slice(-10).map((r) => r.ttft_ms).filter((v): v is number => v != null);
  const avgTtft = ttfts.length ? ttfts.reduce((a, b) => a + b, 0) / ttfts.length : null;
  const activeModel = lib?.models.find((m) => m.active);
  const benchForModel = runs.find((r) => r.status === "done" && r.model === activeModel?.file && r.results?.some((x) => x.tokens_per_joule));
  const tgEff = benchForModel?.results?.find((x) => x.n_gen && !x.n_prompt)?.tokens_per_joule ?? null;
  const ev = energyVerdict(tgEff);

  const speedPts: Point[] = recent.map((r, i) => ({ x: i + 1, y: r.predicted_per_second ?? 0 }));
  const ttftPts: Point[] = recent.map((r, i) => ({ x: i + 1, y: (r.ttft_ms ?? 0) / 1000 }));
  const gpuSeries = (pick: (s: SystemStats) => number | null | undefined): Point[] =>
    hist.map((h) => ({ x: h.t, y: pick(h.s) ?? 0 }));
  const g0 = hist[hist.length - 1]?.s.gpus[0];

  const tips = useMemo<Tip[]>(() => {
    const out: Tip[] = [];
    if (!runs.some((r) => r.status === "done"))
      out.push({ icon: FlaskConical, color: "#7c5cff", title: "Measure your machine", text: "Run a 20-second benchmark to get real speed and energy numbers for this GPU.", page: "bench", cta: "Benchmark" });
    if (mm?.gpu && activeModel && lib) {
      const ctx = lib.runtime.ctx;
      const perTok = activeModel.kv_mb_per_token;
      const extra = perTok ? perTok * ctx : null;
      if (extra && mm.gpu.free - 600 > extra && activeModel.context_length && ctx * 2 <= activeModel.context_length)
        out.push({ icon: Layers, color: "#ff5c9d", title: `Room for ${Math.round((ctx * 2) / 1024)}k context`, text: `You have ${fmtMB(mm.gpu.free)} of free GPU memory. Doubling the context window would only need about ${fmtMB(extra)} more.`, page: "models", cta: "Adjust" });
      if (mm.gpu.other > 1024)
        out.push({ icon: Cpu, color: "#ffa94d", title: `Other apps use ${fmtMB(mm.gpu.other)} of GPU memory`, text: "Windows, browsers and games take VRAM the AI can't use. Closing them makes room for bigger models.", page: "system", cta: "See usage" });
    }
    const bigger = lib?.catalog.find((c) => !c.downloaded && c.fit.verdict === "gpu" && activeModel && c.size > activeModel.size_bytes * 1.2);
    if (bigger) out.push({ icon: Sparkles, color: "#22d3ee", title: `${bigger.title} would fit`, text: `${bigger.note} Bigger models are usually smarter.`, page: "models", cta: "Get it" });
    if (lib && lib.models.length >= 2 && runs.filter((r) => r.status === "done").length < 2)
      out.push({ icon: Gauge, color: "#34d399", title: "Compare your models", text: "Benchmark each model once, then put them side by side on one chart.", page: "bench", cta: "Compare" });
    if (memCount === 0) out.push({ icon: Brain, color: "#7c5cff", title: "Teach Blocky about you", text: "Mention your name, work or preferences in a chat. Blocky remembers them across conversations.", page: "memory", cta: "Memory" });
    return out.slice(0, 4);
  }, [runs, mm, activeModel, lib, memCount]);

  const state = status?.busy === "benchmark" ? "Benchmarking" : status?.busy === "switching" ? "Switching model"
    : status?.model_ready ? "Ready" : "Starting";

  return (
    <div className="page">
      <div className="page-inner wide">
        <div className="dash-head">
          <div>
            <h1 className="page-title">Overview</h1>
            <p className="page-sub">
              <span className={`state-dot ${state === "Ready" ? "ok" : ""}`} /> {state}
              {activeModel && <> · <b>{activeModel.name ?? activeModel.file}</b> <span className="muted">({activeModel.size_label ?? ""} · <Term k="quant">{activeModel.quant}</Term>)</span></>}
            </p>
          </div>
        </div>

        <div className="hero-grid">
          <Hero icon={Gauge} color="#7c5cff" label="Writing speed" term="tps" value={speed != null ? speed.toFixed(0) : "–"} unit="tok/s"
            verdict={speed != null ? sv.label : "Send a chat message to measure"} tone={sv.tone}>
            {speedPts.length > 1 && <LineChart compact points={speedPts} height={44} color="var(--series-1)" unit="tok/s" fmtX={(x) => `answer ${x}`} />}
          </Hero>
          <Hero icon={Clock} color="#ff5c9d" label="Wait before answer" term="ttft" value={avgTtft != null ? (avgTtft / 1000).toFixed(1) : "–"} unit="s"
            verdict={avgTtft != null ? waitVerdict(avgTtft) : "No answers yet"} tone={avgTtft != null && avgTtft < 2000 ? "good" : "meh"}>
            {ttftPts.length > 1 && <LineChart compact points={ttftPts} height={44} color="var(--series-2)" unit="s" fmtX={(x) => `answer ${x}`} />}
          </Hero>
          <Hero icon={Zap} color="#ffa94d" label="Energy per answer" term="tpj" value={ev ? (ev.wh < 0.1 ? (ev.wh * 1000).toFixed(0) : ev.wh.toFixed(2)) : "–"} unit={ev ? (ev.wh < 0.1 ? "mWh" : "Wh") : ""}
            verdict={ev ? ev.compare : "Run a benchmark to measure"} tone={ev ? "good" : "meh"}>
            <Pro>{tgEff != null && <div className="hero-foot mono">{tgEff.toFixed(2)} tok/J · 500-token answer · run #{benchForModel?.id}</div>}</Pro>
          </Hero>
        </div>

        <div className="group-label">Where the model lives</div>
        <div className="group pad">{mm ? <MemoryMap mm={mm} /> : <div className="muted small">Loading…</div>}</div>

        <div className="group-label">Hardware, live</div>
        <div className="live-grid">
          <div className="live-card"><div className="live-head"><Gauge /> GPU load <b>{g0?.util_percent ?? "–"}%</b></div>
            <LineChart compact points={gpuSeries((s) => s.gpus[0]?.util_percent)} yMax={100} height={56} color="var(--series-1)" unit="%" fmtX={(x) => `${x.toFixed(0)} s`} /></div>
          <div className="live-card"><div className="live-head"><Zap /> Power <b>{g0?.power_w != null ? `${g0.power_w.toFixed(0)} W` : "–"}</b></div>
            <LineChart compact points={gpuSeries((s) => s.gpus[0]?.power_w)} height={56} color="var(--series-3)" unit="W" fmtX={(x) => `${x.toFixed(0)} s`} /></div>
          <div className="live-card"><div className="live-head"><Thermometer /> Temperature <b>{g0?.temp_c != null ? `${g0.temp_c}°C` : "–"}</b></div>
            <LineChart compact points={gpuSeries((s) => s.gpus[0]?.temp_c)} height={56} color="var(--series-2)" unit="°C" fmtX={(x) => `${x.toFixed(0)} s`} /></div>
          <div className="live-card"><div className="live-head"><Cpu /> CPU <b>{hist[hist.length - 1]?.s.cpu.percent.toFixed(0) ?? "–"}%</b></div>
            <LineChart compact points={gpuSeries((s) => s.cpu.percent)} yMax={100} height={56} color="var(--series-4)" unit="%" fmtX={(x) => `${x.toFixed(0)} s`} /></div>
        </div>

        {tips.length > 0 && (
          <>
            <div className="group-label">Try next</div>
            <div className="tips">
              {tips.map((t) => (
                <button key={t.title} className="tip" style={{ "--c": t.color } as React.CSSProperties} onClick={() => go(t.page)}>
                  <span className="tip-icon"><t.icon /></span>
                  <span className="tip-body"><b>{t.title}</b><span>{t.text}</span></span>
                  <span className="tip-cta">{t.cta} <ArrowRight /></span>
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
