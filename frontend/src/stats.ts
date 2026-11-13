/** Statistics helpers: normal distribution, multiple-comparison corrections, sample-size (power) calculations. */

export const normCdf = (z: number) => 0.5 * (1 + erf(z / Math.SQRT2));

function erf(x: number) {
  // Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

/** Inverse of the standard normal CDF (Acklam's rational approximation, relative error < 1.2e-9). */
export function normInv(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - lo) return -normInv(1 - p);
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Two-sided p-value of a difference between two measurements with standard errors s1, s2. */
export const zTestP = (diff: number, s1: number, s2: number) => {
  const se = Math.sqrt(s1 * s1 + s2 * s2);
  return se > 0 ? 2 * (1 - normCdf(Math.abs(diff) / se)) : diff === 0 ? 1 : 0;
};

export type Correction = "none" | "holm" | "bh";

/** Adjusted p-values (same order as the input). Holm controls the family-wise error rate; Benjamini–Hochberg the false discovery rate. */
export function adjust(ps: number[], method: Correction): number[] {
  const m = ps.length;
  if (method === "none" || m <= 1) return ps.slice();
  const idx = ps.map((p, i) => [p, i] as const).sort((x, y) => x[0] - y[0]);
  const out = new Array<number>(m);
  if (method === "holm") {
    let run = 0;
    idx.forEach(([p, i], k) => { run = Math.max(run, Math.min(1, (m - k) * p)); out[i] = run; });
  } else {
    let run = 1;
    for (let k = m - 1; k >= 0; k--) { const [p, i] = idx[k]; run = Math.min(run, Math.min(1, (p * m) / (k + 1))); out[i] = run; }
  }
  return out;
}

/** Measurements per setup to detect a difference `delta` between two setups (two-sample z test). */
export function repeatsNeeded(sigma: number, delta: number, alpha = 0.05, power = 0.8) {
  if (!(sigma > 0) || !(delta > 0)) return null;
  const z = normInv(1 - alpha / 2) + normInv(power);
  return Math.max(2, Math.ceil((2 * z * z * sigma * sigma) / (delta * delta)));
}

/** Questions needed for a paired McNemar test to detect an accuracy difference `delta`, given the share of questions on which
 *  the two setups disagree (`discordant`, e.g. 0.12). Connor (1987) approximation. */
export function questionsNeeded(delta: number, discordant: number, alpha = 0.05, power = 0.8) {
  if (!(delta > 0) || !(discordant > delta)) return null;
  const za = normInv(1 - alpha / 2), zb = normInv(power);
  const n = (za * Math.sqrt(discordant) + zb * Math.sqrt(discordant - delta * delta)) ** 2 / (delta * delta);
  return Math.ceil(n);
}

/** Smallest accuracy difference a paired test on n questions detects (inverse of questionsNeeded). */
export function detectableDelta(n: number, discordant: number, alpha = 0.05, power = 0.8) {
  let lo = 1e-4, hi = Math.min(discordant, 1) - 1e-6;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2, need = questionsNeeded(mid, discordant, alpha, power);
    if (need == null || need > n) lo = mid; else hi = mid;
  }
  return hi;
}

/** The correction researchers chose for this browser (default: Holm). */
export function savedCorrection(): Correction {
  try { const v = localStorage.getItem("blocky.correction"); return v === "none" || v === "bh" ? v : "holm"; } catch { return "holm"; }
}
export function saveCorrection(c: Correction) { try { localStorage.setItem("blocky.correction", c); } catch { /* private mode */ } }
export const CORRECTION_LABEL: Record<Correction, string> = { holm: "Holm", bh: "Benjamini–Hochberg", none: "None" };
