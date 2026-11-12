import { useMemo, useState } from "react";
import { TokenProb } from "./api";

/** Shows each generated word-piece tinted by how unsure the model was (one hue: light = sure, strong = unsure). */
export default function ConfidenceView({ probs }: { probs: TokenProb[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const stats = useMemo(() => {
    const ps = probs.map((x) => x.p);
    const avg = ps.reduce((a, b) => a + b, 0) / Math.max(1, ps.length);
    const unsure = ps.filter((p) => p < 0.5).length;
    return { avg, unsure };
  }, [probs]);
  const h = hover != null ? probs[hover] : null;

  return (
    <div className="conf">
      <div className="conf-text" onMouseLeave={() => setHover(null)}>
        {probs.map((x, i) => {
          const u = 1 - x.p; // uncertainty
          return (
            <span key={i} className={hover === i ? "on" : ""} onMouseEnter={() => setHover(i)}
              style={{ background: u > 0.05 ? `rgb(var(--conf-rgb) / ${Math.min(0.75, 0.12 + u * 0.75)})` : undefined }}>
              {x.t}
            </span>
          );
        })}
      </div>
      <div className="conf-info">
        {h ? (
          <>
            <b>"{h.t.trim() || "␣"}": {(h.p * 100).toFixed(h.p > 0.995 ? 1 : 0)}% sure</b>
            {h.alts.length > 0 && <span>It almost wrote: {h.alts.map(([t, p]) => `"${t.trim() || "␣"}" ${(p * 100).toFixed(p < 0.01 ? 1 : 0)}%`).join(" · ")}</span>}
          </>
        ) : (
          <span>Average confidence <b>{(stats.avg * 100).toFixed(0)}%</b> · <b>{stats.unsure}</b> of {probs.length} pieces were coin-flips (under 50%). Stronger color = less sure. Hover any piece to see what else the model considered.</span>
        )}
      </div>
    </div>
  );
}
