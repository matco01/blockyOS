import { useEffect, useState } from "react";

export type Theme = "system" | "light" | "dark";

export const ACCENTS = [
  { id: "violet", rgb: "124 92 255" },
  { id: "pink", rgb: "240 72 155" },
  { id: "orange", rgb: "245 124 32" },
  { id: "green", rgb: "16 185 129" },
  { id: "blue", rgb: "10 132 255" },
] as const;
export type Accent = (typeof ACCENTS)[number]["id"];

/** Block colors used for small markers (chat dots, cards, stats). */
export const BLOCK_COLORS = ["#7c5cff", "#ff5c9d", "#ffa94d", "#22d3ee", "#34d399"];

// Per-browser UI preferences. (Device-wide settings will live on the server later.)
export function useSetting<T>(key: string, fallback: T): [T, (v: T) => void] {
  const storageKey = `blocky.${key}`;
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      return raw == null ? fallback : (JSON.parse(raw) as T);
    } catch {
      return fallback;
    }
  });
  useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify(value)); } catch { /* private mode */ }
  }, [storageKey, value]);
  return [value, setValue];
}
