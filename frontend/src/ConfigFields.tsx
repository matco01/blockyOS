import { configObject } from "./workflow";
import { useEffect, useState } from "react";

function NumberSetting({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  return <input className="field" type="number" step="any" value={text} onChange={(event) => {
    const next = event.target.value; setText(next);
    if (next !== "" && Number.isFinite(Number(next))) onChange(Number(next));
  }} onBlur={() => setText(String(value))} />;
}

export default function ConfigFields({ value, onChange }: { value: string; onChange: (text: string) => void }) {
  let fields: Record<string, unknown>;
  try { fields = configObject(value); } catch { return <p className="error">Settings contain invalid JSON. Open advanced settings to correct it.</p>; }
  const set = (key: string, next: unknown) => onChange(JSON.stringify({ ...fields, [key]: next }, null, 2));
  return <div className="flow-fields">{Object.keys(fields).length === 0 && <p className="row-sub">This program uses its defaults. Add custom parameters under advanced settings.</p>}
    {Object.entries(fields).map(([key, v]) => <label key={key}>{key.replaceAll("_", " ")}
      {typeof v === "boolean" ? <input type="checkbox" checked={v} onChange={(e) => set(key, e.target.checked)} />
        : typeof v === "number" ? <NumberSetting value={v} onChange={(next) => set(key, next)} />
          : typeof v === "string" ? <input className="field" value={v} onChange={(e) => set(key, e.target.value)} />
            : <span className="row-sub">Nested setting — edit its JSON under advanced settings.</span>}
    </label>)}
  </div>;
}
