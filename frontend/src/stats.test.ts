import { describe, expect, it } from "vitest";
import { adjust, detectableDelta, normCdf, normInv, questionsNeeded, repeatsNeeded, zTestP } from "./stats";

describe("normal distribution", () => {
  it("inverts the CDF", () => {
    expect(normInv(0.975)).toBeCloseTo(1.959964, 4);
    expect(normInv(0.8)).toBeCloseTo(0.841621, 4);
    expect(normCdf(1.959964)).toBeCloseTo(0.975, 5);
    expect(zTestP(1.959964 * Math.SQRT2, 1, 1)).toBeCloseTo(0.05, 4);
  });
});

describe("multiple comparisons", () => {
  it("Holm and Benjamini–Hochberg match textbook values", () => {
    expect(adjust([0.01, 0.04, 0.03], "holm").map((x) => +x.toFixed(4))).toEqual([0.03, 0.06, 0.06]);
    expect(adjust([0.01, 0.04, 0.03], "bh").map((x) => +x.toFixed(4))).toEqual([0.03, 0.04, 0.04]);
    expect(adjust([0.2], "holm")).toEqual([0.2]);
    expect(adjust([0.01, 0.02], "none")).toEqual([0.01, 0.02]);
  });
});

describe("sample sizes", () => {
  it("repeats for a two-sample comparison", () => {
    // σ = Δ: n = 2 (1.96 + 0.84)^2 ≈ 15.7 → 16
    expect(repeatsNeeded(1, 1)).toBe(16);
    expect(repeatsNeeded(0, 1)).toBeNull();
  });
  it("McNemar questions and the inverse", () => {
    const n = questionsNeeded(0.02, 0.12)!;
    expect(n).toBeGreaterThan(2000);
    expect(n).toBeLessThan(3000);
    expect(detectableDelta(n, 0.12)).toBeCloseTo(0.02, 3);
    expect(questionsNeeded(0.2, 0.1)).toBeNull();   // can't differ by more than the share of disagreements
  });
});
