import { describe, expect, it } from "vitest";
import { evidence } from "./evidence";
import { METRICS } from "./expmetrics";
import type { ExpRun } from "./api";

const run = (results: unknown, config = {}) => ({ id: 1, results, config }) as ExpRun;
const metric = (key: string) => METRICS.find((m) => m.key === key)!;
describe("metric evidence", () => {
  it("distinguishes measurements, estimates, simulations and missing data", () => {
    expect(evidence(metric("tg"), run({ speed: { tg: 30, tg_samples: [29, 30, 31] } }))).toMatchObject({ kind: "Measured", raw: [29, 30, 31] });
    expect(evidence(metric("roof"), run({ hw: { decode: { roof_util: 0.5 } } })).kind).toBe("Estimated");
    expect(evidence(metric("tg"), run({ speed: { tg: 30 } }, { preset: "minipc" })).kind).toBe("Simulated");
    expect(evidence(metric("tg"), run({})).kind).toBe("Unavailable");
  });
  it("never invents raw samples for legacy summaries", () => {
    expect(evidence(metric("tg"), run({ speed: { tg: 30 } })).raw).toBeUndefined();
  });
  it("treats non-finite values as unavailable", () => {
    expect(evidence(metric("tg"), run({ speed: { tg: Infinity } })).kind).toBe("Unavailable");
  });
});
