import { Code2, Info, Lock, Palette } from "lucide-react";
import { Status } from "./api";
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
            <span className="row-icon" style={{ background: "linear-gradient(180deg,#3a3a3f,#1d1d1f)" }}><Code2 /></span>
            <div className="row-main">API reference</div>
            <a href="/docs" target="_blank" rel="noreferrer">Open</a>
          </div>
        </div>
      </div>
    </div>
  );
}
