// Thin client for the blockyOS backend.

export type Conversation = { id: number; title: string; created_at: number; updated_at: number };

export type Message = {
  id?: number;
  role: "user" | "assistant";
  content: string;
  reasoning?: string | null;
  tokens?: number | null;
  tokens_per_sec?: number | null;
  remembered?: string[]; // UI only: facts learned from this exchange
  thought_secs?: number; // UI only: how long the model reasoned
  budget_hit?: boolean;  // UI only: thinking was cut off by the budget
  probs?: TokenProb[];   // UI only: per-word confidence (when the confidence view was on)
  stats?: MessageStats | null;
};

/** One generated word-piece with the model's confidence and the alternatives it almost picked. */
export type TokenProb = { t: string; p: number; alts: [string, number][] };

export type Status = {
  model_ready: boolean; model: string | null; context_size: number | null;
  busy: "switching" | "benchmark" | "arena" | "eval" | "quantlab" | "throughput" | "experiment" | string | null; switch_error: string | null;
};

export type Contestant = { model: string; think: boolean; sampling: { temperature?: number; top_p?: number; top_k?: number; min_p?: number }; system_prompt?: string; max_tokens?: number; seed?: number };
export type ArenaResult = {
  state: "waiting" | "loading" | "reading" | "thinking" | "writing" | "done" | "skipped";
  text: string; reasoning: string; error: string | null;
  stats: { ttft_ms: number | null; think_ms?: number | null; total_ms: number; prompt_n: number | null; prompt_per_second: number | null; predicted_n: number | null;
    predicted_per_second: number | null; power_w: number | null; energy_j: number | null; tokens_per_joule: number | null; hit_limit: boolean } | null;
};
export type ArenaMatch = {
  id: number; created_at: number; status: "queued" | "running" | "done" | "error" | "cancelled"; prompt: string;
  contestants: Contestant[]; results: ArenaResult[]; display: number[]; blind: boolean; vote: "0" | "1" | "tie" | null; error: string | null;
};
export type BenchmarkInfo = {
  key: string; id: string; title: string; icon: string; color: string; about: string; difficulty: string;
  baseline: number | null; downloaded: boolean; count: number | null;
  download: { done: number; total: number; status: string; error: string | null } | null;
  group: "general" | "coding" | "custom"; custom: boolean; date_from?: string; date_to?: string; published?: string | null;
};
export type FreshSet = { key: string; id: string; title: string; about: string; color: string; icon: string };
export type HfPreview = { splits: { config: string; split: string }[]; columns: string[]; sample: Record<string, string>[] };

export type SuiteInfo = { key: string; title: string; icon: string; color: string; about: string; count: number; custom: boolean; available: boolean };
export type EvalItem = {
  suite: string; index: number; question: string; kind: string; output: string; reasoning_chars: number;
  correct: boolean; got: string; expected: string; seconds: number; tokens: number | null; tps: number | null; energy_j: number | null; truncated: boolean;
  tag?: string | null; budget_hit?: boolean; reasoning_tokens?: number;
};
export type EvalSummary = {
  per_suite: Record<string, { correct: number; total: number; score: number }>;
  overall: number | null; correct: number; total: number; seconds: number; tokens: number;
  avg_tps: number | null; energy_j: number | null; seconds_per_question: number | null; budget_hits?: number;
};
export type EvalRun = {
  id: number; created_at: number; finished_at: number | null; status: "queued" | "running" | "done" | "error" | "cancelled";
  model: string; think: boolean; suites: string[]; results: EvalItem[]; summary: EvalSummary | null; error: string | null;
  options: { think_budget?: number | null; seed?: number };
};

export type QuantResult = {
  model: string; quant: string | null; size_bytes: number; bits: number; reference: boolean; state: string;
  ppl?: number | null; ppl_err?: number | null; kld?: number | null; kld_err?: number | null; kld_99?: number | null;
  same_top?: number | null; pp_tps?: number; tg_tps?: number;
};
export type QuantRun = { id: number; created_at: number; status: string; models: string[]; reference: string; results: QuantResult[]; log: string | null; error: string | null };
export type QuantJob = { id: string; source: string; type: string; output: string; status: string; error: string | null; note: string | null; progress: number };
export type ToolRun = {
  id: number; created_at: number; kind: string; status: string; params: Record<string, unknown>;
  results: Record<string, number | string | null>[]; log: string | null; error: string | null;
};

export type ExpConfig = { model: string; ctx: number; kv: string; ngl: string | number; vram_limit_mb: number | null; fa: string; threads: number | null };
export type Finding = { kind: string; severity: "major" | "minor" | "info"; title: string; text: string };
export type PhaseTelemetry = { gpu_avg?: number; mem_bw_avg?: number; power_avg?: number; power_max?: number; temp_max?: number; vram_max_mb?: number; cpu_avg?: number; ram_max_mb?: number; seconds?: number };
export type ExpResults = {
  load_seconds?: number;
  memory?: { gpu: Record<string, number>; ram: Record<string, number>; layers: { gpu: number; total: number } | null; vram_used_mb?: number };
  quality?: Record<string, { correct: number; total: number; score: number }>; quality_score?: number; chat_tps?: number; ttft_ms?: number;
  speed?: { pp?: number; tg?: number; pp_d4096?: number; tg_d4096?: number; tg_sd?: number };
  ppl?: number; kld?: number; same_top?: number;
  telemetry?: Record<string, PhaseTelemetry>;
  series?: { t: number; phase: string; gpu?: number; mem_bw?: number; power?: number; temp?: number; vram_mb?: number; cpu?: number }[];
  bottleneck?: { findings: Finding[]; primary: string | null; ceiling_tps: number | null; efficiency: number | null; gb_per_token: number; gpu_bandwidth_gbps: number | null; ram_bandwidth_gbps: number | null };
};
export type ExpRun = { id: number; experiment_id: number; created_at: number; status: string; label: string; config: ExpConfig; results: ExpResults; phase: string | null; error: string | null };
export type Experiment = {
  id: number; created_at: number; name: string; model: string; seed: number; runs?: number;
  runs_list?: ExpRun[]; limits?: { gpu: string | null; gpu_bandwidth_gbps: number | null; vram_mb: number; power_limit_w: number | null; ram_bandwidth_gbps: number | null; ram_mb: number; cpu: string };
};

export type LeaderRow = { label: string; model: string; matches: number; wins: number; ties: number; losses: number; avg_tps: number | null; score: number };

export type Fit = { verdict: "gpu" | "split" | "split_moe" | "cpu" | "too_big"; label: string; need_mb: number; kv_mb: number; measured?: boolean };

export type ModelFile = {
  file: string; size_bytes: number; quant: string | null; name?: string; architecture?: string; size_label?: string;
  context_length?: number; layers?: number; embedding?: number; heads?: number; kv_heads?: number; head_dim?: number;
  experts?: number | null; experts_used?: number | null; license?: string | null; error?: string;
  active: boolean; fit: Fit; kv_mb_per_token?: number; fixed_state_mb?: number;
};

export type Runtime = { model: string | null; ctx: number; kv: string; ngl: string; threads: string };

export type Hardware = {
  cpu: string; cpu_threads: number; cpu_cores: number; ram_mb: number;
  gpus: { name: string; driver: string; vram_mb: number; power_limit_w: number | null; pcie: string | null }[];
  os: string; kernel: string; wsl: boolean; llama_cpp: string | null;
};

export type CatalogItem = { repo: string; file: string; size: number; title: string; note: string; moe: boolean; downloaded: boolean; fit: Fit };
export type HfFile = { path: string; size: number; split: boolean; fit: Fit };
export type Download = { id: string; repo: string; file: string; total: number; done: number; status: string; error: string | null; speed: number };

export type BenchResult = {
  test: string; n_prompt: number; n_gen: number; n_depth: number; avg_ts: number; stddev_ts: number; samples_ts: number[];
  power_w: number | null; tokens_per_joule: number | null; peak_vram_mb: number | null;
  n_gpu_layers: number; n_cpu_moe: number; type_k: string; flash_attn: number | string; n_threads: number; build: string;
};
export type BenchRun = {
  id: number; created_at: number; finished_at: number | null; status: "queued" | "running" | "done" | "error" | "cancelled";
  model: string; params: Record<string, unknown>; hardware: Hardware | null; results: BenchResult[] | null; log: string | null; error: string | null;
};

export type MemoryMapData = {
  parsed: boolean;
  gpu: { name: string; total: number; used: number; model: number; context: number; compute: number; other: number; free: number } | null;
  ram: { total: number; used: number; model: number; context: number; compute: number; other: number; free: number };
  layers: { gpu: number; total: number } | null;
  kv: { mib: number; cells: number; attention_layers: number } | null;
  kv_mb: number; fixed_state_mb: number;
  runtime: Runtime;
};

export type RecentStat = MessageStats & { id: number; created_at: number; conversation_id: number };

export type Sampling = {
  temperature?: number; top_p?: number; top_k?: number; min_p?: number;
  repeat_penalty?: number; presence_penalty?: number; max_tokens?: number; seed?: number;
};

export type MessageStats = {
  ttft_ms: number | null; total_ms: number; prompt_n: number | null; prompt_per_second: number | null;
  cache_n: number | null; predicted_n: number | null; predicted_per_second: number | null; think?: boolean;
  sampling?: Sampling; custom_system_prompt?: boolean; system_n?: number | null;
  budget_hit?: boolean; think_budget?: number | null; sampling_recommended?: Sampling;
};

export type Gpu = {
  name: string;
  util_percent: number | null;
  vram_used_mb: number | null;
  vram_total_mb: number | null;
  temp_c: number | null;
  power_w: number | null;
};

export type SystemStats = {
  cpu: { percent: number; cores: number };
  ram: { used_mb: number; total_mb: number };
  disk: { used_gb: number; total_gb: number };
  gpus: Gpu[];
};

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try { const b = await r.json(); if (b?.detail) detail = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail); } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return r.json();
}

export type Memory = {
  id: number;
  content: string;
  source: "auto" | "user";
  conversation_id: number | null;
  created_at: number;
  updated_at: number;
};

export type Settings = { memory_enabled: boolean };

export type SearchHit = { conversation_id: number; title: string; role: string; snippet: string };

const send = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  conversations: () => json<Conversation[]>("/api/conversations"),
  messages: (id: number) => json<Message[]>(`/api/conversations/${id}`),
  remove: (id: number) => json(`/api/conversations/${id}`, { method: "DELETE" }),
  search: (q: string) => json<SearchHit[]>(`/api/search?q=${encodeURIComponent(q)}`),
  status: () => json<Status>("/api/status"),
  system: () => json<SystemStats>("/api/system"),

  memories: () => json<Memory[]>("/api/memories"),
  addMemory: (content: string) => json<Memory>("/api/memories", send("POST", { content })),
  updateMemory: (id: number, content: string) => json(`/api/memories/${id}`, send("PATCH", { content })),
  deleteMemory: (id: number) => json(`/api/memories/${id}`, { method: "DELETE" }),
  deleteAllMemories: () => json("/api/memories", { method: "DELETE" }),

  settings: () => json<Settings>("/api/settings"),
  updateSettings: (s: Partial<Settings>) => json<Settings>("/api/settings", send("PATCH", s)),

  models: () => json<{ models: ModelFile[]; runtime: Runtime; catalog: CatalogItem[]; hardware: Hardware; kv_types: string[] }>("/api/models"),
  setRuntime: (r: Partial<Runtime>) => json<{ ok: boolean; runtime: Runtime }>("/api/models/runtime", send("POST", r)),
  deleteModel: (file: string) => json(`/api/models/${encodeURIComponent(file)}`, { method: "DELETE" }),
  hfFiles: (repo: string) => json<HfFile[]>(`/api/hf/files?repo=${encodeURIComponent(repo)}`),
  download: (repo: string, path: string) => json<{ id: string }>("/api/downloads", send("POST", { repo, path })),
  downloads: () => json<Download[]>("/api/downloads"),
  cancelDownload: (id: string) => json(`/api/downloads/${id}/cancel`, send("POST", {})),

  benchRuns: () => json<BenchRun[]>("/api/bench"),
  startBench: (p: Record<string, unknown>) => json<{ id: number }>("/api/bench", send("POST", p)),
  cancelBench: (id: number) => json(`/api/bench/${id}/cancel`, send("POST", {})),
  deleteBench: (id: number) => json(`/api/bench/${id}`, { method: "DELETE" }),
  logs: (lines = 300) => json<{ text: string }>(`/api/logs?lines=${lines}`),
  memoryMap: () => json<MemoryMapData>("/api/memory-map"),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelAnatomy: (file: string) => json<any>(`/api/models/${encodeURIComponent(file)}/anatomy`),
  benchmarks: () => json<BenchmarkInfo[]>("/api/benchmarks"),
  benchmarkDownload: (id: string) => json(`/api/benchmarks/${id}/download`, send("POST", {})),
  benchmarkDelete: (id: string) => json(`/api/benchmarks/${id}`, { method: "DELETE" }),
  hfDataset: (repo: string) => json<HfPreview>(`/api/hf/dataset?repo=${encodeURIComponent(repo)}`),
  benchmarkImport: (b: Record<string, unknown>) => json<{ key: string }>("/api/benchmarks/custom", send("POST", b)),
  evalSuites: () => json<SuiteInfo[]>("/api/evals/suites"),
  freshSets: () => json<FreshSet[]>("/api/evals/fresh"),
  freshPreview: (id: string) => json<{ q: string; kind: string }[]>(`/api/evals/fresh/${id}/preview?n=3`),
  evalCreateSuite: (name: string, questions: { q: string; answer: string }[]) => json<{ key: string }>("/api/evals/suites", send("POST", { name, questions })),
  evalDeleteSuite: (key: string) => json(`/api/evals/suites/${encodeURIComponent(key)}`, { method: "DELETE" }),
  evalRuns: () => json<EvalRun[]>("/api/evals"),
  evalGet: (id: number) => json<EvalRun>(`/api/evals/${id}`),
  evalStart: (b: { model: string; think: boolean; suites: string[]; think_budget?: number }) => json<{ id: number }>("/api/evals", send("POST", b)),
  evalCancel: (id: number) => json(`/api/evals/${id}/cancel`, send("POST", {})),
  evalDelete: (id: number) => json(`/api/evals/${id}`, { method: "DELETE" }),
  experiments: () => json<Experiment[]>("/api/experiments"),
  experiment: (id: number) => json<Experiment>(`/api/experiments/${id}`),
  experimentCreate: (name: string, model: string) => json<{ id: number }>("/api/experiments", send("POST", { name, model })),
  experimentDelete: (id: number) => json(`/api/experiments/${id}`, { method: "DELETE" }),
  experimentRun: (id: number, label: string, config: Partial<ExpConfig>) => json<{ id: number }>(`/api/experiments/${id}/runs`, send("POST", { label, config })),
  experimentCancel: (runId: number) => json(`/api/experiments/runs/${runId}/cancel`, send("POST", {})),
  experimentDeleteRun: (runId: number) => json(`/api/experiments/runs/${runId}`, { method: "DELETE" }),
  quantLab: () => json<{ runs: QuantRun[]; jobs: QuantJob[]; types: string[] }>("/api/quantlab"),
  quantStart: (models: string[]) => json<{ id: number }>("/api/quantlab", send("POST", { models })),
  quantCancel: (id: number) => json(`/api/quantlab/${id}/cancel`, send("POST", {})),
  quantDelete: (id: number) => json(`/api/quantlab/${id}`, { method: "DELETE" }),
  quantMake: (source: string, type: string) => json<{ id: string }>("/api/quantlab/quantize", send("POST", { source, type })),
  toolRuns: (kind: "throughput" | "speculative") => json<ToolRun[]>(`/api/tools/${kind}`),
  toolStart: (kind: "throughput" | "speculative", body: Record<string, unknown>) => json<{ id: number }>(`/api/tools/${kind}`, send("POST", body)),
  toolCancel: (id: number) => json(`/api/tools/run/${id}/cancel`, send("POST", {})),
  toolDelete: (id: number) => json(`/api/tools/run/${id}`, { method: "DELETE" }),
  arenaList: () => json<ArenaMatch[]>("/api/arena"),
  arenaGet: (id: number) => json<ArenaMatch>(`/api/arena/${id}`),
  arenaStart: (b: { prompt: string; contestants: Partial<Contestant>[]; blind: boolean; max_tokens?: number }) => json<{ id: number }>("/api/arena", send("POST", b)),
  arenaVote: (id: number, winner: "0" | "1" | "tie") => json<ArenaMatch>(`/api/arena/${id}/vote`, send("POST", { winner })),
  arenaCancel: (id: number) => json(`/api/arena/${id}/cancel`, send("POST", {})),
  arenaDelete: (id: number) => json(`/api/arena/${id}`, { method: "DELETE" }),
  arenaBoard: () => json<LeaderRow[]>("/api/arena/leaderboard"),
  recentStats: (limit = 30) => json<RecentStat[]>(`/api/stats/recent?limit=${limit}`),
};

export type ChatEvent =
  | { type: "meta"; conversation_id: number }
  | { type: "token"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "done"; tokens: number | null; tokens_per_sec: number | null; seconds: number; stats?: MessageStats }
  | { type: "memory"; added: Memory[] }
  | { type: "budget"; tokens: number }
  | { type: "probs"; items: TokenProb[] }
  | { type: "error"; message: string };

/** Sends a message and calls onEvent for every streamed piece of the answer. */
export async function streamChat(
  body: { message: string; conversation_id: number | null; think: boolean; sampling?: Sampling; system_prompt?: string; think_budget?: number; confidence?: boolean },
  onEvent: (e: ChatEvent) => void,
  signal?: AbortSignal,
) {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`Chat failed: ${r.status}`);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const line = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
    }
  }
}
