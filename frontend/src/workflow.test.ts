import { describe, expect, it } from "vitest";
import { configObject, modelTitle } from "./workflow";

describe("research workflow inputs", () => {
  it("uses the filename when model metadata has a generic name", () => {
    for (const name of ["Hf", "Merged", "Unknown", "model", " ", undefined]) {
      expect(modelTitle({ name, file: "Qwen-custom-Q4.gguf" })).toBe("Qwen-custom-Q4");
    }
    expect(modelTitle({ name: "My research model", file: "weights.gguf" })).toBe("My research model");
  });
  it("preserves nested custom configuration without accepting non-object roots", () => {
    expect(configObject('{"layers":[4,8],"optimizer":{"lr":0.01},"train":true}')).toEqual({ layers: [4, 8], optimizer: { lr: 0.01 }, train: true });
    for (const value of ["null", "[]", "1", '"text"', "false", "{"]) expect(() => configObject(value)).toThrow();
  });
});
