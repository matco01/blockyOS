import { useState } from "react";
import ConfigFields from "./ConfigFields";
import { configObject, modelTitle } from "./workflow";
import { useMode } from "./ui";
import { ExpConfig, ExpRun, ModelFile, RegisteredMethod } from "./api";

export default function MethodChangePanel({ methods, models, parent, onRun }: { methods: RegisteredMethod[]; models: ModelFile[]; parent: ExpRun; onRun: (label: string, cfg: Partial<ExpConfig>) => void }) {
  const mode = useMode();
  const initial = parent.config.model.startsWith("method:") ? parent.config.model : methods[0] ? `method:${methods[0].id}` : models[0]?.file ?? "";
  const [selected, setSelected] = useState(initial);
  const method = methods.find((m) => `method:${m.id}` === selected);
  const [settings, setSettings] = useState(JSON.stringify(parent.config.model === initial ? parent.config.method_config ?? method?.definition.spec.config ?? {} : method?.definition.spec.config ?? {}, null, 2));
  const [error, setError] = useState("");
  const select = (value: string) => { setSelected(value); setError(""); setSettings(JSON.stringify(methods.find((m) => `method:${m.id}` === value)?.definition.spec.config ?? {}, null, 2)); };
  return <div className="ws">
    <b>Evaluate a registered method or conventional model</b>
    <p className="row-sub">Registered methods run their pinned evaluator and report its supported metrics. Shared metrics compare only when their evaluation contracts match. Unsupported native phases stay unavailable.</p>
    <label>Method or model<select aria-label="Method or model" className="field select" value={selected} onChange={(e) => select(e.target.value)}>{methods.map((m) => <option key={m.id} value={`method:${m.id}`}>{m.name} (registered method #{m.id})</option>)}{models.map((m) => <option key={m.file} value={m.file}>{modelTitle(m)} · {m.quant}</option>)}</select></label>
    {method && <><p className="row-sub">Supported metrics: {method.definition.capabilities.map((m) => `${m.name} (${m.unit})`).join(", ")}. {method.definition.spec.gpu ? "Reserves GPU and pauses chat." : "CPU evaluation."}</p>
      <ConfigFields value={settings} onChange={setSettings} /><details className="flow-advanced" open={mode === "pro" || undefined}><summary>Advanced method settings (JSON)</summary><label>Method settings (JSON)<textarea aria-label="Method settings" className="field rp-config" value={settings} onChange={(e) => setSettings(e.target.value)} /></label></details></>}
    {error && <p className="error">{error}</p>}
    <button className="btn" disabled={!selected} onClick={() => { try { const config = method ? configObject(settings) : {}; if (typeof config !== "object" || config === null || Array.isArray(config)) throw new Error("Settings must be a JSON object"); setError(""); onRun(method?.name ?? selected, { model: selected, ...(method ? { method_config: config } : {}) }); } catch (e) { setError((e as Error).message); } }}>Queue evaluation</button>
    {parent.config.model.startsWith("method:") && <button className="btn ghost" onClick={() => onRun(`Repeat ${parent.label}`, parent.config)}>Repeat this method setup</button>}
  </div>;
}
