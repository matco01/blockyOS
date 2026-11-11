import { useEffect, useRef, useState } from "react";
import { Cpu, Gauge, HardDrive, MemoryStick, Thermometer, Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { api, Status, SystemStats } from "./api";

function Stat({ icon: Icon, label, value, unit, sub, pct, wide, color }: {
  icon: LucideIcon; label: string; value: string; unit?: string; sub?: string; pct?: number; wide?: boolean; color?: string;
}) {
  return (
    <div className={`stat ${wide ? "wide" : ""}`} style={color ? ({ "--c": color } as React.CSSProperties) : undefined}>
      <div className="stat-head"><Icon /> {label}</div>
      <div className="stat-value">{value}{unit && <small>{unit}</small>}</div>
      {sub && <div className="stat-sub">{sub}</div>}
      {pct != null && <div className="bar"><div style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} /></div>}
    </div>
  );
}

const gb = (mb: number) => (mb / 1024).toFixed(1);

/** Tail of the model server's log: what llama.cpp is doing, offload decisions, errors. */
function LogView() {
  const [text, setText] = useState("");
  const [live, setLive] = useState(true);
  const [filter, setFilter] = useState("");
  const ref = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const load = () => api.logs(400).then((r) => setText(r.text)).catch(() => {});
    load();
    if (!live) return;
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [live]);

  useEffect(() => {
    if (live && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [text, live]);

  const lines = text.split("\n").filter((l) => !filter || l.toLowerCase().includes(filter.toLowerCase()));

  return (
    <>
      <div className="group-label history-head">
        <span>Model server log</span>
        <span className="exports">
          <label className="small muted" style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} /> Live
          </label>
        </span>
      </div>
      <div className="group">
        <div className="row"><input className="field" placeholder="Filter, e.g. offload, error, CUDA, slot" value={filter} onChange={(e) => setFilter(e.target.value)} /></div>
        <div className="row block"><pre ref={ref} className="log tall">{lines.join("\n") || "No log lines."}</pre></div>
      </div>
    </>
  );
}

export default function SystemPage({ status }: { status: Status | null }) {
  const [s, setS] = useState<SystemStats | null>(null);

  useEffect(() => {
    const poll = () => api.system().then(setS).catch(() => {});
    poll();
    const t = setInterval(poll, 2000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="page">
      <div className="page-inner">
        <h1 className="page-title">System</h1>
        <p className="page-sub">Live view of this device. Updates every two seconds.</p>

        <div className="group-label">Model</div>
        <div className="group">
          <div className="row"><div className="row-main">Running</div><span className="value">{status?.model ?? "Not loaded"}</span></div>
          <div className="row"><div className="row-main">Context window</div><span className="value">{status?.context_size ? `${status.context_size.toLocaleString()} tokens` : "–"}</span></div>
        </div>

        {s && (
          <>
            {s.gpus.map((g, i) => (
              <div key={i}>
                <div className="group-label">Graphics</div>
                <div className="stats">
                  <Stat color="#7c5cff" wide icon={Gauge} label={g.name} value={`${g.util_percent ?? 0}`} unit="%" sub="GPU utilization" pct={g.util_percent ?? 0} />
                  {g.vram_total_mb != null && (
                    <Stat color="#ff5c9d" icon={MemoryStick} label="Video memory" value={gb(g.vram_used_mb ?? 0)} unit={`/ ${gb(g.vram_total_mb)} GB`}
                      pct={((g.vram_used_mb ?? 0) / g.vram_total_mb) * 100} />
                  )}
                  <Stat color="#ffa94d" icon={g.power_w != null ? Zap : Thermometer} label={g.power_w != null ? "Power · Temp" : "Temperature"}
                    value={g.power_w != null ? g.power_w.toFixed(0) : `${g.temp_c ?? "–"}`} unit={g.power_w != null ? "W" : "°C"}
                    sub={g.temp_c != null ? `${g.temp_c}°C` : undefined} />
                </div>
              </div>
            ))}

            <div className="group-label">Computer</div>
            <div className="stats">
              <Stat color="#22d3ee" icon={Cpu} label="Processor" value={s.cpu.percent.toFixed(0)} unit="%" sub={`${s.cpu.cores} threads`} pct={s.cpu.percent} />
              <Stat color="#34d399" icon={MemoryStick} label="Memory" value={gb(s.ram.used_mb)} unit={`/ ${gb(s.ram.total_mb)} GB`} pct={(s.ram.used_mb / s.ram.total_mb) * 100} />
              <Stat color="#7c5cff" wide icon={HardDrive} label="Storage" value={s.disk.used_gb.toFixed(0)} unit={`/ ${s.disk.total_gb.toFixed(0)} GB`} pct={(s.disk.used_gb / s.disk.total_gb) * 100} />
            </div>
            {s.gpus.length === 0 && <p className="page-sub" style={{ marginTop: 16 }}>No NVIDIA GPU detected. Running on the processor.</p>}
          </>
        )}

        <LogView />
      </div>
    </div>
  );
}
