import { useEffect, useState } from "react";
import { Bot, Code2, Info, Lock, Network, Palette, PlugZap, Server, Trash2 } from "lucide-react";
import { api, Device, Settings, Status } from "./api";
import { Accent, ACCENTS, Theme } from "./settings";

const THEMES: { id: Theme; label: string }[] = [
  { id: "system", label: "Auto" },
  { id: "light", label: "Light" },
  { id: "dark", label: "Dark" },
];

export default function SettingsPage({ status, theme, setTheme, accent, setAccent }: {
  status: Status | null; theme: Theme; setTheme: (t: Theme) => void; accent: Accent; setAccent: (a: Accent) => void;
}) {
  return (
    <div className="page">
      <div className="page-inner">
        <h1 className="page-title">Settings</h1>
        <p className="page-sub">Make Blocky yours.</p>

        <div className="group-label">Appearance</div>
        <div className="group">
          <div className="row">
            <span className="row-icon" style={{ background: "linear-gradient(180deg,#4ad9f0,#16b4d4)" }}><Palette /></span>
            <div className="row-main">Theme</div>
            <div className="segmented">
              {THEMES.map((t) => (
                <button key={t.id} className={theme === t.id ? "on" : ""} onClick={() => setTheme(t.id)}>{t.label}</button>
              ))}
            </div>
          </div>
          <div className="row">
            <span className="row-icon" style={{ background: "var(--accent-grad)" }}><Palette /></span>
            <div className="row-main">Accent color</div>
            <div className="swatches">
              {ACCENTS.map((a) => (
                <button
                  key={a.id}
                  className={`swatch ${accent === a.id ? "on" : ""}`}
                  style={{ "--s": a.rgb } as React.CSSProperties}
                  title={a.id}
                  aria-label={`${a.id} accent`}
                  onClick={() => setAccent(a.id)}
                />
              ))}
            </div>
          </div>
        </div>

        <AgentSettings />
        <EnergySettings />
        <RemoteSettings />

        <div className="group-label">About</div>
        <div className="group">
          <div className="row">
            <span className="row-icon" style={{ background: "linear-gradient(180deg,#a1a1a6,#6e6e73)" }}><Info /></span>
            <div className="row-main">Version</div><span className="value">blockyOS 0.1</span>
          </div>
          <div className="row">
            <span className="row-icon" style={{ background: "linear-gradient(180deg,#5ee08a,#28b350)" }}><Lock /></span>
            <div className="row-main">Privacy</div><span className="value">Local only</span>
          </div>
          <div className="row">
            <span className="row-icon" style={{ background: "linear-gradient(180deg,#ffb340,#ff8a00)" }}><Info /></span>
            <div className="row-main">Model</div><span className="value">{status?.model ?? "–"}</span>
          </div>
        </div>

        <div className="group-label">Developer</div>
        <div className="group">
          <div className="row">
            <span className="row-icon" style={{ background: "linear-gradient(180deg,#3a3a3f,#1d1d1f)" }}><Code2 /></span>
            <div className="row-main">
              <div>Device configuration</div>
              <div className="row-sub">Edit <code>/etc/blocky/blocky.env</code>, then restart the services.</div>
            </div>
          </div>
          <div className="row">
            <span className="row-icon" style={{ background: "linear-gradient(180deg,#b38cff,#7a4dff)" }}><Bot /></span>
            <div className="row-main">
              <div>AI assistants (MCP)</div>
              <div className="row-sub">Let Claude Code, Claude Desktop or another MCP client run experiments: <code>claude mcp add --transport http blocky http://localhost:8000/mcp</code>. See the <a href="#/guide">Guide</a>.</div>
            </div>
          </div>
          <div className="row">
            <span className="row-icon" style={{ background: "linear-gradient(180deg,#3a3a3f,#1d1d1f)" }}><Code2 /></span>
            <div className="row-main">API reference</div>
            <a href="/docs" target="_blank" rel="noreferrer">Open</a>
          </div>
        </div>
      </div>
    </div>
  );
}

function EnergySettings() {
  const [s, setS] = useState<Settings | null>(null);
  const [draft, setDraft] = useState({ url: "", field: "", device: "", kwh: "", currency: "" });
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  useEffect(() => {
    api.settings().then((v) => {
      setS(v);
      setDraft({ url: v.power_meter_url || "", field: v.power_meter_field || "", device: v.device_price?.toString() ?? "",
        kwh: v.electricity_price_kwh?.toString() ?? "", currency: v.currency || "" });
    }).catch(() => {});
  }, []);
  if (!s) return null;
  const save = async () => {
    try {
      setS(await api.updateSettings({ power_meter_url: draft.url.trim(), power_meter_field: draft.field.trim(), currency: draft.currency.trim(),
        device_price: draft.device === "" ? null : Number(draft.device), electricity_price_kwh: draft.kwh === "" ? null : Number(draft.kwh) }));
      setMsg({ ok: true, text: "Saved. New experiment runs use these values." });
    } catch (e) { setMsg({ ok: false, text: String((e as Error).message || e) }); }
  };
  const test = async () => {
    setMsg(null);
    try { const r = await api.testPowerMeter(draft.url.trim(), draft.field.trim()); setMsg({ ok: true, text: `Meter reads ${r.watts.toFixed(1)} W right now.` }); }
    catch (e) { setMsg({ ok: false, text: String((e as Error).message || e) }); }
  };
  const set = (k: keyof typeof draft) => (e: React.ChangeEvent<HTMLInputElement>) => setDraft({ ...draft, [k]: e.target.value });
  return (
    <>
      <div className="group-label">Energy & cost</div>
      <div className="group">
        <div className="row">
          <span className="row-icon" style={{ background: "linear-gradient(180deg,#ffd166,#ff9f1c)" }}><PlugZap /></span>
          <div className="row-main">
            <div>Wall-power meter <span className="row-sub">(optional)</span></div>
            <div className="row-sub">A smart plug on this device's power cable measures what the whole device draws. Shelly (e.g. <code>http://&lt;plug-ip&gt;/rpc/Switch.GetStatus?id=0</code>), Tasmota (<code>http://&lt;plug-ip&gt;/cm?cmnd=Status%208</code>) or any address that returns JSON with the watts. Without one, runs use the GPU and CPU power sensors when the hardware has them.</div>
            <div className="form-grid" style={{ marginTop: 8, gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)" }}>
              <input className="field" placeholder="http://192.168.1.50/rpc/Switch.GetStatus?id=0" value={draft.url} onChange={set("url")} />
              <input className="field" placeholder="Field with the watts (only for other meters), e.g. emeters.0.power" value={draft.field} onChange={set("field")} />
            </div>
          </div>
        </div>
        <div className="row">
          <span className="row-icon" style={{ background: "linear-gradient(180deg,#5ee08a,#28b350)" }}><PlugZap /></span>
          <div className="row-main">
            <div>Prices</div>
            <div className="row-sub">Used to report electricity cost per 1M tokens and speed per device price. Any currency.</div>
            <div className="form-grid" style={{ marginTop: 8, gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
              <input className="field" inputMode="decimal" placeholder="Device price, e.g. 450" value={draft.device} onChange={set("device")} />
              <input className="field" inputMode="decimal" placeholder="Electricity per kWh, e.g. 0.25" value={draft.kwh} onChange={set("kwh")} />
              <input className="field" placeholder="Currency, e.g. USD" value={draft.currency} onChange={set("currency")} />
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center", flexWrap: "wrap" }}>
              <button className="btn small" onClick={save}>Save</button>
              <button className="btn small ghost" onClick={test} disabled={!draft.url.trim()}>Test meter</button>
              {msg && <span className="row-sub" style={{ color: msg.ok ? undefined : "var(--red, #e5484d)" }}>{msg.text}</span>}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function RemoteSettings() {
  const [access, setAccess] = useState<{ enabled: boolean; token: string | null } | null>(null);
  const [devs, setDevs] = useState<Device[]>([]);
  const [d, setD] = useState({ name: "", url: "", token: "" });
  const [checks, setChecks] = useState<Record<number, string>>({});
  const [err, setErr] = useState<string | null>(null);
  const load = () => { api.remoteAccess().then(setAccess).catch(() => {}); api.devices().then(setDevs).catch(() => {}); };
  useEffect(load, []);
  const add = async () => {
    setErr(null);
    try { await api.deviceAdd(d.name, d.url, d.token); setD({ name: "", url: "", token: "" }); load(); } catch (e) { setErr((e as Error).message); }
  };
  const check = async (id: number) => {
    setChecks((c) => ({ ...c, [id]: "Checking…" }));
    const r = await api.deviceCheck(id).catch((e) => ({ ok: false, error: String(e) } as Awaited<ReturnType<typeof api.deviceCheck>>));
    setChecks((c) => ({ ...c, [id]: r.ok ? `${r.hardware?.gpu ?? "no GPU"} · ${r.hardware?.cpu} · ${Math.round((r.hardware?.ram_mb ?? 0) / 1024)} GB RAM · ${r.models?.length ?? 0} models` : r.error ?? "Failed" }));
  };
  return (
    <>
      <div className="group-label">Remote access & devices</div>
      <div className="group">
        <div className="row">
          <span className="row-icon" style={{ background: "linear-gradient(180deg,#5ac8fa,#007aff)" }}><Network /></span>
          <div className="row-main">
            <div>Let other devices control this Blocky</div>
            <div className="row-sub">Off: only this device can use it. On: other devices need the token below. Anyone with the token can run experiments and build engines here, so only turn it on for a network you trust; otherwise use an SSH tunnel (<code>ssh -L 8001:localhost:8000 user@this-device</code>).</div>
            {access?.enabled && access.token && <div className="row-sub" style={{ marginTop: 6 }}>Token: <code style={{ userSelect: "all" }}>{access.token}</code> · in a browser on another device, open <code>http://&lt;this-device&gt;:8000/?key=…</code> once</div>}
          </div>
          <div className="segmented">
            <button className={access?.enabled ? "" : "on"} onClick={async () => setAccess(await api.setRemoteAccess(false))}>Off</button>
            <button className={access?.enabled ? "on" : ""} onClick={async () => setAccess(await api.setRemoteAccess(true))}>On</button>
          </div>
        </div>
        <div className="row">
          <span className="row-icon" style={{ background: "linear-gradient(180deg,#a1a1a6,#6e6e73)" }}><Server /></span>
          <div className="row-main">
            <div>Other Blocky devices</div>
            <div className="row-sub">Add a device to profile runs on it from here (Experiments → Run on another device). Results come back into the same experiment, labeled with the device's hardware.</div>
            {devs.map((v) => (
              <div key={v.id} style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6, flexWrap: "wrap" }}>
                <b>{v.name}</b><span className="row-sub">{v.url}{v.token ? " · token set" : ""}</span>
                <button className="btn small ghost" onClick={() => check(v.id)}>Check</button>
                <button className="btn small ghost" title="Remove" onClick={async () => { await api.deviceDelete(v.id); load(); }}><Trash2 /></button>
                {checks[v.id] && <span className="row-sub">{checks[v.id]}</span>}
              </div>
            ))}
            <div className="form-grid" style={{ marginTop: 8 }}>
              <input className="field" placeholder="Name, e.g. mini-pc" value={d.name} onChange={(e) => setD({ ...d, name: e.target.value })} />
              <input className="field" placeholder="http://192.168.1.40:8000" value={d.url} onChange={(e) => setD({ ...d, url: e.target.value })} />
              <input className="field" placeholder="Its token (from its Settings)" value={d.token} onChange={(e) => setD({ ...d, token: e.target.value })} />
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 8, alignItems: "center" }}>
              <button className="btn small" disabled={!d.url.trim()} onClick={add}>Add device</button>
              {err && <span className="row-sub" style={{ color: "var(--red, #e5484d)" }}>{err}</span>}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function AgentSettings() {
  const [st, setSt] = useState<{ claude: boolean; claude_model: string; claude_models: string[] } | null>(null);
  const [key, setKey] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  useEffect(() => { api.agentStatus().then(setSt).catch(() => {}); }, []);
  const save = async (k: string) => {
    try { setSt(await api.agentKey(k)); setKey(""); setMsg({ ok: true, text: k ? "Key saved on this device." : "Key removed." }); }
    catch (e) { setMsg({ ok: false, text: (e as Error).message }); }
  };
  return (
    <>
      <div className="group-label">AI agent</div>
      <div className="group">
        <div className="row">
          <span className="row-icon" style={{ background: "linear-gradient(180deg,#d4a27f,#b5651d)" }}><Bot /></span>
          <div className="row-main">
            <div>Claude for the research agent <span className="row-sub">(optional)</span></div>
            <div className="row-sub">The Agent page runs on this device's own model by default. With an Anthropic API key it can use Claude instead, which is far better at long research tasks; then messages and tool results are sent to Anthropic. The key is stored only on this device and never shown again.</div>
            <div style={{ display: "flex", gap: 8, marginTop: 8, alignItems: "center", flexWrap: "wrap" }}>
              {st?.claude ? <><span className="row-sub">Key set ✓</span><button className="btn small ghost" onClick={() => save("")}>Remove key</button></>
                : <><input className="field" type="password" placeholder="sk-ant-…" value={key} onChange={(e) => setKey(e.target.value)} style={{ maxWidth: 320 }} />
                  <button className="btn small" disabled={!key.trim()} onClick={() => save(key.trim())}>Save key</button></>}
              {st && <select className="field select" value={st.claude_model} onChange={async (e) => { await api.updateSettings({ agent_model: e.target.value }); setSt({ ...st, claude_model: e.target.value }); }}>
                {st.claude_models.map((m) => <option key={m} value={m}>{m}</option>)}</select>}
              {msg && <span className="row-sub" style={{ color: msg.ok ? undefined : "var(--red, #e5484d)" }}>{msg.text}</span>}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
