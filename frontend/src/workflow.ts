import type { ModelFile } from "./api";

export function modelTitle(model: Pick<ModelFile, "file" | "name">): string {
  const name = model.name?.trim();
  return name && !/^(hf|merged|model|unknown|unnamed)$/i.test(name) ? name : model.file.replace(/\.gguf$/i, "");
}

export function configObject(text: string): Record<string, unknown> {
  const value = JSON.parse(text);
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Settings must be a JSON object");
  return value;
}
