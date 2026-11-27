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
  gpu_pause?: GpuPause | null; gpu_waiting?: { temp_c: number; target_c: number; since: number } | null;
};

/** GPU safety: why GPU work is paused (overheating, a lost GPU, a restart after a crash). */
export type GpuPause = { reason: string; kind: "heat" | "lost" | "restart" | "manual"; since: number };
export type GpuSafety = {
  enabled: boolean; limit_c: number; resume_c: number; temp_c: number | null; gpu_seen: boolean;
  auto: { limit_c: number; resume_c: number; max_operating_c: number | null; slowdown_c: number | null };
  waiting: { temp_c: number; target_c: number } | null; paused: GpuPause | null; events: { at: number; kind: string; text: string }[];
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

export type ExpConfig = { model: string; ctx: number; kv: string; ngl: string | number; vram_limit_mb: number | null; fa: string; threads: number | null; engine?: string;
  method_config?: Record<string, unknown>;
  device?: "gpu" | "cpu"; cores?: number | null; ram_limit_mb?: number | null; preset?: string | null;
  gpus?: number | null; split_mode?: "layer" | "row" | "none" | null; rpc?: string[] | null; pp?: number | null };
export type RpcWorker = { enabled: boolean; expose: boolean; port: number; running: boolean; host: string; installed: boolean; error: string | null };
export type Device = { id: number; name: string; url: string; token: boolean; created_at: number };
export type AiLevel = "read" | "write" | "full";
export type AiConnection = { id: number; created_at: number; client_name: string; redirect: string; level: AiLevel; last_used_at: number | null };
export type AiAccess = { local_level: AiLevel; public_enabled: boolean; public_url: string; levels: Record<AiLevel, string>; pairing_code: string | null;
  local_mcp_url: string; public_mcp_url: string | null; locked: boolean; connections: AiConnection[] };
export type RemoteJob = { id: number; experiment_id: number; device: string; device_id?: number; status: string; step: string | null; error: string | null; imported?: { runs: number } };
export type DevicePreset = { label: string; config: Partial<ExpConfig>; ram_bandwidth_gbps?: number; memory?: string };
export type Finding = { kind: string; severity: "major" | "minor" | "info"; title: string; text: string };
export type PhaseTelemetry = { gpu_avg?: number; mem_bw_avg?: number; power_avg?: number; power_max?: number; temp_max?: number; vram_max_mb?: number; cpu_avg?: number; ram_max_mb?: number; seconds?: number };
export type Mmlu = { benchmark: string; protocol: string; n: number; correct: number; acc: number; ci95: [number, number]; p_correct_mean: number;
  by_category: Record<string, { correct: number; total: number }>; chance: number };
export type HwAnalysis = {
  weights_gb_per_token: number; weights_on_gpu_gb: number; weights_on_cpu_gb: number; params_active_b: number | null;
  peaks: { vram_gbps: number | null; ram_gbps: number | null; tflops: number | null; tflops_kind: string | null; sm_clock_mhz: number | null };
  decode?: { ms_per_token: number; effective_gbps: number; roof_tps: number | null; roof_util: number | null;
    time_split_ms: { vram_reads: number | null; ram_reads: number | null; other: number | null }; achieved_tflops: number | null; arithmetic_intensity: number | null; long_context_ratio: number | null };
  prefill?: { achieved_tflops: number; arithmetic_intensity: number; roof_tflops: number | null; roof_util: number | null; long_context_ratio: number | null };
  clocks: { sm_avg_mhz: number | null; sm_max_mhz: number | null; ratio: number | null; temp_max_c: number | null; gpu_util_avg: number | null;
    mem_util_avg: number | null; cpu_util_avg: number | null; power_avg_w: number | null; cpu_power_avg_w: number | null };
  headroom?: { vram_capacity_mb: number; vram_model_mb: number; vram_used_total_mb: number | null; vram_free_mb: number; kv_mb_per_1k_tokens: number | null; extra_context_tokens: number | null };
  limiter?: { what: string; evidence: string };
  projection?: { target: string; target_ram_gbps: number; memory?: string; bandwidth_share?: number; tg: number | null; method?: string; unavailable?: string };
};
export type ExpResults = {
  research_metrics?: Record<string, ResearchMetric>;
  research_job_id?: number;
  research_provenance?: Record<string, unknown>;
  research_artifacts?: ResearchMetric[];
  method?: { id: number; name: string };
  load_seconds?: number;
  memory?: { gpu: Record<string, number>; ram: Record<string, number>; layers: { gpu: number; total: number } | null; vram_used_mb?: number; ram_needed_mb?: number };
  mmlu?: Mmlu; mmlu_progress?: { done: number; total: number; correct: number };
  speed?: { pp?: number; tg?: number; pp_d4096?: number; tg_d4096?: number; pp_sd?: number; tg_sd?: number; pp_d4096_sd?: number; tg_d4096_sd?: number };
  ppl?: number; ppl_sd?: number; kld?: number; kld_sd?: number; kld_p99?: number; same_top?: number; same_top_sd?: number;
  telemetry?: Record<string, PhaseTelemetry>;
  series?: { t: number; phase: string; gpu?: number; mem_bw?: number; power?: number; temp?: number; vram_mb?: number; cpu?: number; sm_clock?: number }[];
  hw?: HwAnalysis;
  provenance?: Provenance;
  commands?: Record<string, string>;
  token_note?: string;
  fidelity_note?: string;
  serving?: { pp: number | null; pp_sd: number | null; tg: number | null; tg_sd: number | null; ttft_ms: number | null; ttft_sd: number | null; method?: string };
  kernels?: { available: boolean; note?: string; prefill?: KernelProfile; decode?: KernelProfile };
  hardware?: { gpu: string | null; gpu_bandwidth_gbps: number | null; gpu_tflops: number | null; vram_mb: number; ram_mb: number; ram_bandwidth_gbps: number | null;
    cpu: string; cpu_threads: number; power_limit_w: number | null; sm_clock_max: number | null; compute_capability?: string | null; device?: string };
  imported_from?: { device: string | null; gpu: string; experiment: string; imported_at: number };
  needle?: { lengths: number[]; depths: number[]; cells: { length: number; depth: number; ok: boolean; prefill_tps: number | null }[]; acc: number | null;
    by_length: Record<string, number>; prefill_tps_by_length: Record<string, number | null>; max_length: number | null };
  needle_progress?: { done: number; total: number; correct: number };
  lm_eval?: { tasks?: Record<string, { metrics: Record<string, number>; stderr: Record<string, number> }>; limit?: number; error?: string; command?: string };
  energy?: Energy;
  ruler?: { lengths: number[]; tasks: string[]; samples: number; cells: { task: string; length: number; score: number; ci95: number; n: number }[];
    by_length: Record<string, number>; by_task: Record<string, number>; score: number | null; effective_length: number | null; threshold: number; method: string };
  ruler_progress?: { done: number; total: number };
  remote_device?: { id: number; name: string; url: string };
  cost?: { currency: string; electricity_per_mtok?: number; tps_per_1k_price?: number };
};
export type PowerStat = { avg: number; sd: number; n: number };
export type EnergySource = { idle_w: PowerStat | null; load_w: PowerStat; j_per_token: number; tokens_per_wh: number | null; net_j_per_token?: number };
export type Energy = { tokens?: number; seconds?: number; tps?: number; sources?: Record<string, EnergySource>; primary?: string | null;
  primary_label?: string | null; note?: string; error?: string; meter?: boolean };
export type Provenance = {
  protocol: string;
  llama_cpp: { version: string | null; commit: string | null; built_with: string | null };
  model: { file: string; sha256: string | null; bytes: number };
  data: { mmlu_pro_sha256: string | null; mmlu_seed: number; wikitext_sha256: string | null };
  gpu: { name: string | null; driver: string | null; vram_mb: number | null; pcie: string | null; cuda_driver_api: string | null; cuda_toolkit: string | null };
  host: { cpu: string; threads: number; ram_mb: number; ram_gb?: number; os: string; kernel: string; wsl: boolean; device?: string };
  blocky: { commit: string | null; dirty: boolean };
  started_at: string;
};
export type KernelProfile = { total_ms: number; categories: { category: string; ms: number; share: number; calls: number; kernels: number }[];
  top: { name: string; ms: number; share: number; calls: number; category: string }[] };
export type ExpRun ={ id: number; experiment_id: number; created_at: number; status: string; label: string; config: ExpConfig; results: ExpResults;
  log?: string | null;
  phase: string | null; error: string | null; parent_id?: number | null;
  stats?: { mmlu_p?: number | null; mmlu_diff?: PairedDiff | null; mmlu_equiv?: Equivalence | null; mmlu_flips?: { lost: number; gained: number }; env_mismatch?: string[]; hw_diff?: string[]; baseline_id?: number;
    vs_parent?: { mmlu_p?: number | null; mmlu_diff?: PairedDiff | null; mmlu_equiv?: Equivalence | null; mmlu_flips?: { lost: number; gained: number }; env_mismatch?: string[] } } };
export type InstrumentCheck = { id: string; title: string; status: "pass" | "warn" | "fail" | "not_run"; detail: string; todo?: string; hint?: string;
  runs?: number[]; value?: number | null; inversions?: [string, string][];
  matches?: { run: number; metric: string; measured: number; reference: number; tolerance: number; source: string; ok: boolean }[] };
export type InstrumentReport = { version: string; checks: InstrumentCheck[]; counts?: Record<"pass" | "warn" | "fail" | "not_run", number>; note: string };
/** Equivalence test (TOST) of the accuracy difference, in percentage points. */
export type Equivalence = { ci90_pts: [number, number]; min_margin_pts: number; method: string; margin_pts?: number; equivalent?: boolean };
export type Preregistration = { id: number; created_at: number; sha256: string; confirmatory_runs: number[]; note: string;
  plan: { hypothesis: string; primary_metric: string; claim: string; direction: string; alpha: number; analysis: string; protocol: string;
    min_effect?: number; equivalence_margin_pts?: number; secondary_metrics: string[] } };
/** Accuracy difference (run minus comparison run, as a fraction) with a paired 95% interval. */
export type PairedDiff = { diff: number; ci95: [number, number]; n: number; method: string };
export type TorchStatus = { installed: boolean; converter: boolean; torch?: string; cuda?: boolean; transformers?: string; peft?: string;
  checkpoints: { id: string; repo: string; architecture: string; layers: number | null; size_bytes: number }[] };
export type TorchJob = { id: number; created_at: number; kind: string; status: string; step: string | null; error: string | null;
  params: Record<string, string>; results: Record<string, unknown>; log_tail?: string[] };
export type LayerMeasure ={ layer: number; kld: number | null; kld_p99: number | null; same_top: number | null; ppl: number | null; excess_kld: number; bytes: number; seconds: number };
export type LayerScan = { id: number; created_at: number; model: string; mode: "quantize" | "prune"; qtype: string | null; status: string; step: string | null; error: string | null;
  results: { layers?: LayerMeasure[]; n_layers?: number; source_type?: string; control?: { kld: number | null; same_top: number | null }; reference?: { ppl: number | null };
    most_sensitive?: number[]; least_sensitive?: number[] } };
export type Engine = { id: string; name: string; kind?: "llamacpp" | "vllm" | "sglang" | "trtllm"; repo: string; ref: string; commit_sha: string | null; status: string; step?: string | null;
  error?: string | null; built_at: number | null; created_at?: number };
export type WeightFile ={ file: string; quant: string | null; bits: number; size_bytes: number; layers: number | null; recipe: Record<string, unknown> | null };
export type TensorGroup = { pattern: string; count: number; bytes: number; types: string[]; part: string };
export type VariantSpec = { source: string; qtype: string; imatrix: boolean; overrides: Record<string, string>; output_type?: string | null;
  embedding_type?: string | null; prune: number[]; name?: string };
export type WeightJob = VariantSpec & { id: string; out: string; status: string; step: string | null; error: string | null; started: number;
  size_bytes?: number; imatrix_cached?: boolean; log_tail: string[] };
export type CanvasNote ={ id: string; x: number; y: number; text: string };
export type CanvasState = { pos: Record<string, { x: number; y: number }>; notes: CanvasNote[]; view?: { x: number; y: number; k: number }; devices?: number[] };
export type Experiment = {
  validations?: ValidationBatch[];
  id: number; created_at: number; name: string; model: string; seed: number; runs?: number; canvas?: CanvasState | null; project_id?: number | null;
  protocol?: { lm_eval?: { tasks: string; limit: number }; kernels?: boolean; mmlu_per_category?: number; ruler?: { samples: number }; ppl_chunks?: number;
    fidelity_text?: string; equivalence_margin_pts?: number; lock_clocks_mhz?: number; cooldown_s?: number } | null;
  preregistrations?: Preregistration[];
  runs_list?: ExpRun[]; limits?: { gpu: string | null; gpu_bandwidth_gbps: number | null; gpu_tflops: number | null; gpu_tflops_kind: string | null; sm_clock_max: number | null;
    vram_mb: number; power_limit_w: number | null; ram_bandwidth_gbps: number | null; ram_mb: number; cpu: string; cpu_power_source: string | null };
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

export type Settings = { memory_enabled: boolean; gpu_guard?: boolean; gpu_max_temp_c?: number | null; gpu_resume_temp_c?: number | null; anthropic_api_key_set?: boolean; agent_model?: string; power_meter_url: string; power_meter_field: string; device_price: number | null;
  electricity_price_kwh: number | null; currency: string };

export type SearchHit = { conversation_id: number; title: string; role: string; snippet: string };

/** A project: what someone is making, holding its experiments (model lines) and code. */
export type ProjectResults = { mmlu: number | null; tg: number | null; ppl: number | null };
export type ProjectExperiment = {
  id: number; name: string; model: string; created_at: number; runs: number; done: number; active: number; last_run: number | null;
  baseline: { label: string; results: ProjectResults | null } | null; latest: { label: string; results: ProjectResults | null } | null;
};
export type ProjectCode = { id: number; name: string; created_at: number };
export type Project = {
  id: number; name: string; goal: string; created_at: number; updated_at: number; last_activity: number;
  experiments: ProjectExperiment[]; code: ProjectCode[];
};

/** Sends a file as the raw request body (PUT), reporting progress from 0 to 1. Large files stream to disk on the box. */
export function upload<T>(url: string, file: Blob, onProgress?: (fraction: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("PUT", url);
    x.setRequestHeader("Content-Type", "application/octet-stream");
    x.upload.onprogress = (e) => { if (e.lengthComputable && e.total > 0) onProgress?.(e.loaded / e.total); };
    x.onload = () => {
      if (x.status >= 200 && x.status < 300) { resolve(JSON.parse(x.responseText || "{}") as T); return; }
      let detail = `${x.status} ${x.statusText}`;
      try { detail = JSON.parse(x.responseText).detail ?? detail; } catch { /* not JSON */ }
      reject(new Error(detail));
    };
    x.onerror = () => reject(new Error("The upload was interrupted. Check that Blocky is running and try again."));
    x.send(file);
  });
}

/** A code workspace's own Python environment. */
export type EnvStatus = {
  status: "none" | "installing" | "ready" | "error"; error: string | null; python: string | null;
  requirements: { file: string | null; packages: string[]; sha256: string | null };
  up_to_date: boolean; reuses_ml: boolean; ml_available: boolean; log: string;
};

export type Dataset = { name: string; path: string; bytes: number; rows: number | null; kind: "jsonl" | "txt" };

const send = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  researchMethods: () => json<RegisteredMethod[]>("/api/research/methods"),
  researchRegister: (jid: number, name: string) => json<RegisteredMethod>(`/api/research/jobs/${jid}/register`, send("POST", { name })),
  researchProjects: () => json<{ projects: ResearchProject[]; python: string; environments: { label: string; python: string }[] }>("/api/research/projects"),
  researchPromote: (jid: number, name: string) => json<{ experiment_id: number; run_id: number }>(`/api/research/jobs/${jid}/experiment`, send("POST", { name })),
  researchCreate: (name: string, template: string) => json<{ id: number }>("/api/research/projects", send("POST", { name, template })),
  researchFiles: (pid: number) => json<string[]>(`/api/research/projects/${pid}/files`),
  researchRead: (pid: number, name: string) => json<{ content: string }>(`/api/research/projects/${pid}/file?name=${encodeURIComponent(name)}`),
  researchWrite: (pid: number, name: string, content: string) => json(`/api/research/projects/${pid}/file`, send("PUT", { name, content })),
  researchJobs: (pid: number) => json<ResearchJob[]>(`/api/research/projects/${pid}/jobs`),
  researchQueue: (pid: number, body: unknown) => json<{ id: number }>(`/api/research/projects/${pid}/jobs`, send("POST", body)),
  researchCancel: (jid: number) => json(`/api/research/jobs/${jid}/cancel`, send("POST", {})),
  researchCompare: (ids: string[]) => json<ResearchComparison>(`/api/research/compare?ids=${encodeURIComponent(ids.join(","))}`),
  researchProfileRuns: () => json<{ id: number; label: string; experiment: string }[]>("/api/research/profile-runs"),
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
  gpuSafety: () => json<GpuSafety>("/api/gpu-safety"),
  gpuResume: () => json<GpuSafety>("/api/gpu-safety/resume", send("POST", {})),
  testPowerMeter: (power_meter_url: string, power_meter_field: string) => json<{ watts: number }>("/api/power/test", send("POST", { power_meter_url, power_meter_field })),

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
  experimentEvidence: (id: number) => json<{ outputs: Record<string, { text: string; truncated: boolean }>; note: string }>(`/api/experiments/runs/${id}/evidence`),
  instrumentCheck: (id: number) => json<InstrumentReport>(`/api/experiments/${id}/instrument-check`),
  experimentValidate: (id: number, body: { reference_id: number; candidate_id: number; pairs: number; metric: string; min_effect_pct: number }) => json<{ id: number; run_ids: number[] }>(`/api/experiments/${id}/validate`, send("POST", body)),
  validationCancel: (id: number, vid: number) => json(`/api/experiments/${id}/validations/${vid}/cancel`, send("POST", {})),
  experimentCreate: (name: string, model: string, projectId?: number) => json<{ id: number }>("/api/experiments", send("POST", { name, model, project_id: projectId ?? null })),
  projects: () => json<{ projects: Project[]; unassigned: { experiments: ProjectExperiment[]; code: ProjectCode[] } }>("/api/projects"),
  project: (id: number) => json<Project>(`/api/projects/${id}`),
  projectCreate: (name: string, goal: string) => json<{ id: number }>("/api/projects", send("POST", { name, goal })),
  projectUpdate: (id: number, body: { name?: string; goal?: string }) => json<Project>(`/api/projects/${id}`, send("PATCH", body)),
  projectDelete: (id: number) => json(`/api/projects/${id}`, { method: "DELETE" }),
  projectAssign: (kind: "experiment" | "code", id: number, projectId: number | null) => json("/api/projects/assign", send("PUT", { kind, id, project_id: projectId })),
  projectData: (id: number) => json<{ datasets: Dataset[] }>(`/api/projects/${id}/data`),
  projectDataUpload: (id: number, file: File, onProgress?: (f: number) => void) =>
    upload<{ name: string; path: string }>(`/api/projects/${id}/data?name=${encodeURIComponent(file.name)}`, file, onProgress),
  projectDataDelete: (id: number, name: string) => json(`/api/projects/${id}/data?name=${encodeURIComponent(name)}`, { method: "DELETE" }),
  modelUpload: (file: File, onProgress?: (f: number) => void) =>
    upload<{ file: string; bytes: number }>(`/api/models/upload?name=${encodeURIComponent(file.name)}`, file, onProgress),
  researchUpload: (rid: number, path: string, file: Blob) =>
    upload<{ path: string }>(`/api/research/projects/${rid}/upload?path=${encodeURIComponent(path)}`, file),
  researchImportZip: (rid: number, file: File, onProgress?: (f: number) => void) =>
    upload<{ files: number; entry_candidates: string[] }>(`/api/research/projects/${rid}/import-zip`, file, onProgress),
  researchEntries: (rid: number) => json<{ entries: string[] }>(`/api/research/projects/${rid}/entries`),
  researchImportGit: (rid: number, url: string, ref: string) =>
    json<{ files: number; lfs_pointers: number; entry_candidates: string[] }>(`/api/research/projects/${rid}/import-git`, send("POST", { url, ref })),
  researchEnv: (rid: number) => json<EnvStatus>(`/api/research/projects/${rid}/env`),
  researchEnvInstall: (rid: number, reuseMl: boolean) => json<EnvStatus>(`/api/research/projects/${rid}/env/install`, send("POST", { reuse_ml: reuseMl })),
  researchEnvDelete: (rid: number) => json(`/api/research/projects/${rid}/env`, { method: "DELETE" }),
  urlDownload: (url: string, name = "") => json<{ id: string }>("/api/downloads/url", send("POST", { url, name })),
  checkpointUpload: (name: string, path: string, file: Blob) =>
    upload<{ path: string }>(`/api/torch/checkpoints/upload?name=${encodeURIComponent(name)}&path=${encodeURIComponent(path)}`, file),
  checkpointFinish: (name: string) => json<{ id: string; repo: string }>("/api/torch/checkpoints/finish", send("POST", { name })),
  experimentDelete: (id: number) => json(`/api/experiments/${id}`, { method: "DELETE" }),
  experimentRun: (id: number, label: string, config: Partial<ExpConfig>, parentId?: number | null) =>
    json<{ id: number }>(`/api/experiments/${id}/runs`, send("POST", { label, config, parent_id: parentId ?? null })),
  experimentSweep: (id: number, grid: Record<string, unknown[]>, repeats: number, parentId?: number | null) =>
    json<{ ids: number[] }>(`/api/experiments/${id}/sweep`, send("POST", { grid, repeats, parent_id: parentId ?? null })),
  torchStatus: () => json<TorchStatus>("/api/torch/status"),
  torchJobs: (kind?: string) => json<TorchJob[]>(`/api/torch/jobs${kind ? `?kind=${kind}` : ""}`),
  torchJob: (id: number) => json<TorchJob>(`/api/torch/jobs/${id}`),
  torchCancel: (id: number) => json(`/api/torch/jobs/${id}/cancel`, send("POST", {})),
  torchDownload: (repo: string) => json<{ id: number }>("/api/torch/download", send("POST", { repo })),
  torchActivations: (checkpoint: string, prompt: string) => json<{ id: number }>("/api/torch/activations", send("POST", { checkpoint, prompt })),
  torchLora: (body: Record<string, unknown>) => json<{ id: number }>("/api/torch/lora", send("POST", body)),
  torchTrain: (body: Record<string, unknown>) => json<{ id: number }>("/api/torch/train", send("POST", body)),
  torchQuantsim: (body: Record<string, unknown>) => json<{ id: number }>("/api/torch/quantsim", send("POST", body)),
  torchQuantsimExport: (id: number, type: string) => json<{ id: number }>(`/api/torch/quantsim/${id}/export`, send("POST", { type, profile: true })),
  notebooks: () => json<{ installed: boolean; running: boolean; url: string | null; folder: string }>("/api/notebooks"),
  scans: (model?: string) => json<LayerScan[]>(`/api/scans${model ? `?model=${encodeURIComponent(model)}` : ""}`),
  scanStart: (model: string, mode: string, qtype: string) => json<{ id: number }>("/api/scans", send("POST", { model, mode, qtype })),
  scanCancel: (id: number) => json(`/api/scans/${id}/cancel`, send("POST", {})),
  scanDelete: (id: number) => json(`/api/scans/${id}`, { method: "DELETE" }),
  engines: () => json<Engine[]>("/api/engines"),
  rpcWorker: () => json<RpcWorker>("/api/rpc-worker"),
  setRpcWorker: (s: { enabled: boolean; expose: boolean; port: number }) => json<RpcWorker>("/api/rpc-worker", send("POST", s)),
  aiAccess: () => json<AiAccess>("/api/ai-access"),
  setAiAccess: (patch: Partial<Pick<AiAccess, "local_level" | "public_enabled" | "public_url">>) => json<AiAccess>("/api/ai-access", send("PATCH", patch)),
  aiNewCode: () => json<AiAccess>("/api/ai-access/pairing-code", send("POST", {})),
  aiRevoke: (id: number) => json<AiAccess>(`/api/ai-access/connections/${id}`, send("DELETE")),
  remoteAccess: () => json<{ enabled: boolean; token: string | null }>("/api/remote-access"),
  setRemoteAccess: (enabled: boolean) => json<{ enabled: boolean; token: string | null }>("/api/remote-access", send("POST", { enabled })),
  devices: () => json<Device[]>("/api/devices"),
  deviceAdd: (name: string, url: string, token: string) => json<{ id: number }>("/api/devices", send("POST", { name, url, token })),
  deviceDelete: (id: number) => json<{ ok: boolean }>(`/api/devices/${id}`, send("DELETE")),
  deviceCheck: (id: number) => json<{ ok: boolean; error?: string; hardware?: { gpu: string | null; cpu: string; ram_mb: number; vram_mb: number }; models?: string[] }>(`/api/devices/${id}/check`),
  remoteRun: (expId: number, device_id: number, run_ids: number[]) => json<{ job_id: number }>(`/api/experiments/${expId}/remote`, send("POST", { device_id, run_ids })),
  remoteJobs: (expId: number) => json<RemoteJob[]>(`/api/remote-jobs?experiment_id=${expId}`),
  remoteJobCancel: (id: number) => json<{ ok: boolean }>(`/api/remote-jobs/${id}/cancel`, send("POST")),
  agentStatus: () => json<AgentStatus>("/api/agent/status"),
  agentKey: (key: string) => json<AgentStatus>("/api/agent/key", send("POST", { key })),
  agentSessions: () => json<{ id: number; title: string; provider: string; updated_at: number }[]>("/api/agent/sessions"),
  agentSession: (id: number) => json<AgentSession>(`/api/agent/sessions/${id}`),
  agentNew: (provider: string) => json<{ id: number }>("/api/agent/sessions", send("POST", { provider })),
  agentDelete: (id: number) => json(`/api/agent/sessions/${id}`, send("DELETE")),
  agentApprove: (sid: number, id: string, name: string, allow: boolean, always: boolean) => json(`/api/agent/sessions/${sid}/approve`, send("POST", { id, name, allow, always })),
  agentStop: (sid: number) => json(`/api/agent/sessions/${sid}/stop`, send("POST")),
  devicePresets: () => json<{ presets: Record<string, DevicePreset>; cpu_threads: number; gpu_count?: number }>("/api/device-presets"),
  engineAdd: (repo: string, ref: string, name: string, backend = "auto") => json<{ id: string }>("/api/engines", send("POST", { repo, ref, name, backend })),
  engineLog: (id: string) => json<{ log: string }>(`/api/engines/${id}/log`),
  engineCancel: (id: string) => json(`/api/engines/${id}/cancel`, send("POST", {})),
  engineDelete: (id: string) => json(`/api/engines/${id}`, { method: "DELETE" }),
  weightsFamily: (model: string) => json<{ files: WeightFile[]; base_types: string[]; tensor_types: string[]; needs_imatrix: string[] }>(`/api/weights/family?model=${encodeURIComponent(model)}`),
  weightsTensors: (model: string) => json<TensorGroup[]>(`/api/weights/tensors?model=${encodeURIComponent(model)}`),
  weightsCreate: (spec: VariantSpec) => json<{ id: string }>("/api/weights/variants", send("POST", spec)),
  weightsJobs: () => json<WeightJob[]>("/api/weights/jobs"),
  weightsCancel: (id: string) => json(`/api/weights/jobs/${id}/cancel`, send("POST", {})),
  experimentImport: async (file: File, into?: number) => {
    const r = await fetch(`/api/experiments/import${into ? `?into=${into}` : ""}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: await file.text() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `Import failed (${r.status})`);
    return d as { experiment_id: number; runs: number; from: { device: string | null; gpu: string } };
  },
  experimentProtocol: (id: number, lm_eval: { tasks: string; limit: number } | null, kernels = false, mmlu_per_category = 40, ruler: { samples: number } | null = null, ppl_chunks = 32,
    more: Record<string, unknown> = {}) =>
    json(`/api/experiments/${id}/protocol`, send("PUT", { lm_eval, kernels, mmlu_per_category, ruler, ppl_chunks, ...more })),
  experimentPreregister: (id: number, plan: Record<string, unknown>) =>
    json<{ id: number; sha256: string }>(`/api/experiments/${id}/preregistrations`, send("POST", plan)),
  experimentCanvas: (id: number, canvas: CanvasState) => json(`/api/experiments/${id}/canvas`, send("PUT", canvas)),
  experimentCancelQueue: (id: number) => json(`/api/experiments/${id}/cancel-queue`, send("POST", {})),
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

export type AgentItem = { role: "user" | "assistant" | "error"; text: string }
  | { role: "tool"; id: string; name: string; input: Record<string, unknown>; status: string; result?: string };
export type AgentSession = { id: number; title: string; provider: string; model: string; display: AgentItem[]; running: boolean };
export type AgentStatus = { claude: boolean; claude_model: string; claude_models: string[] };
export type AgentEvent = { type: "text" | "thinking"; text: string } | { type: "approval" | "tool_call"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; id: string; content: string; is_error: boolean; status: string } | { type: "error"; message: string } | { type: "done" };

async function readSSE<T>(r: Response, onEvent: (e: T) => void) {
  if (!r.ok || !r.body) throw new Error(`Request failed: ${r.status} ${await r.text().catch(() => "")}`.slice(0, 300));
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

export async function streamAgent(sid: number, text: string, onEvent: (e: AgentEvent) => void) {
  const r = await fetch(`/api/agent/sessions/${sid}/message`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
  await readSSE(r, onEvent);
}

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
export type ValidationBatch = {
  id: number; created_at: number;
  plan: { pairs: number; metric: string; min_effect_pct: number; protocol: string; sources: { id: number; label: string }[] };
  summary: { metric_label: string; unit: string; conclusion: string; complete: boolean; problems: string[]; note: string;
    stats: { n: number; median_pct: number | null; sd_pct: number | null; range_pct: number[] | null; ci95_pct: number[] | null; p: number | null };
    pairs: { pair: number; run_ids: number[]; states: string[]; reference: number | null; candidate: number | null; improvement_pct: number | null }[] };
};
export type ResearchMetric = { type: string; name?: string; value?: number; unit?: string; kind?: string; protocol?: string; task?: string; dataset_sha256?: string; scope?: string; better?: string | null; tokenizer_sha256?: string | null; samples?: number[] | null; spread?: { n: number; sd: number | null }; note?: string; path?: string; message?: string };
export type ResearchProject = { id: number; name: string; path: string; created_at: number };
export type ResearchJob = { id: number; project_id: number; parent_id: number | null; status: string; error: string | null; directory: string; log: string; events: ResearchMetric[]; spec: { argv: string[]; config: Record<string, unknown>; python: string; gpu: boolean; timeout_s: number }; provenance: Record<string, unknown> };
export type ResearchComparison = { runs: { id: number | string; project_id: number | null; status: string }[]; metrics: { name: string; compatible: boolean; reasons: string[]; values: (ResearchMetric | null)[] }[]; note: string };
export type RegisteredMethod = { id: number; name: string; source_job_id: number; definition: { spec: { config: Record<string, unknown>; gpu: boolean }; capabilities: ResearchMetric[]; note: string } };
