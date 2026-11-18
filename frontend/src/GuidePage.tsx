import { useEffect, useRef, useState } from "react";
import {
  Activity, ArrowRight, BookOpen, Bot, Boxes, Brain, ClipboardCheck, Cpu, FlaskConical, Gauge, GitBranch, Hammer, LayoutDashboard,
  MessageSquare, Microscope, Network, PlugZap, Radar, Repeat, Settings as SettingsIcon, Smartphone, Swords, Target, Wand2,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { GLOSSARY } from "./ui";

/** The Blocky guide: every feature, the research workflow that ties them together, and how the numbers are made. */

type Go = (hash: string) => void;

const TOC = [
  { id: "g-overview", label: "What Blocky is" },
  { id: "g-start", label: "Quick start" },
  { id: "g-workflow", label: "The research workflow" },
  { id: "g-example", label: "Worked example" },
  { id: "g-features", label: "Every feature" },
  { id: "g-measure", label: "How runs are measured" },
  { id: "g-reading", label: "Reading the results" },
  { id: "g-canvas", label: "Working on the canvas" },
  { id: "g-python", label: "Python & notebooks" },
  { id: "g-assistants", label: "AI assistants" },
  { id: "g-limits", label: "Limits & good practice" },
  { id: "g-glossary", label: "Glossary" },
  { id: "g-hood", label: "Under the hood" },
];

const STEPS: { n: number; icon: LucideIcon; title: string; text: string; where: string; hash: string }[] = [
  { n: 1, icon: FlaskConical, title: "Run a baseline", text: "Pick a model and profile it once. This is the reference everything else is compared to.", where: "Experiments → New experiment → Run baseline", hash: "#/experiments" },
  { n: 2, icon: Target, title: "Read the profile", text: "Speed, quality, long-context recall, memory, drift, and what limits it on this hardware.", where: "Experiments → Overview / Compare results", hash: "#/experiments" },
  { n: 3, icon: GitBranch, title: "Change one thing", text: "Branch from any run: another quantization, KV cache, context, engine, or a whole different model. Or queue a sweep.", where: "Change panel, or + on a canvas card", hash: "#/experiments" },
  { n: 4, icon: Repeat, title: "Compare and validate", text: "Inspect compatible metrics side by side, then repeat both setups in alternating AB/BA order. Instrument checks confirm the measurements behave correctly. Insufficient evidence remains inconclusive.", where: "Compare results / Repeat and validate / Canvas compare", hash: "#/experiments" },
  { n: 5, icon: Gauge, title: "Find the bottleneck", text: "Memory speed, system RAM or software overhead: measured against the hardware's physical limits.", where: "What limits each run · Roofline", hash: "#/experiments" },
  { n: 6, icon: Hammer, title: "Go deeper", text: "Build your own weights, scan which layers matter, try a new engine, fine-tune, look inside. Then loop back to 3.", where: "Workshop · Engines · Internals", hash: "#/internals" },
];

type Feature = { icon: LucideIcon; title: string; what: string; can: string[]; hash: string; tag?: string };
const FEATURES: { group: string; intro: string; items: Feature[] }[] = [
  {
    group: "Research", intro: "Where experiments happen.",
    items: [
      { icon: Bot, title: "Agent", hash: "#/agent", what: "A research agent inside Blocky: give it a task in plain words and it works through it with the whole lab.",
        can: ["Uses every lab tool: experiments, runs, sweeps, weight variants, layer scans, engines, device presets, the technique playground",
          "Writes and runs Python in its own workspace (analysis with the blocky client, or PyTorch); you allow each run",
          "Every step is visible: tool calls, inputs, results", "Brain: this device's model (private) or Claude with your API key (stronger; data goes to Anthropic)",
          "Keeps working if you leave the page; stop it any time"] },
      { icon: FlaskConical, title: "Experiments", hash: "#/experiments", what: "The research hub: one experiment = a tree of profiled runs, all measured the same way.",
        can: ["Baseline profile of any model on this hardware", "Change one setting, or branch from any run", "Sweeps: queue every combination of settings (up to 32 runs), leave it overnight",
          "Repeat a setup to measure session-to-session noise", "Report view: results table, charts, long-context grid, hardware limits, telemetry",
          "Canvas view: the experiment as a tree you work in", "Reproducibility: versions, file hashes, exact commands, and the hardware each run ran on",
          "Compare devices: export JSON on one Blocky device, import it on another", "CSV / JSON export of every number"] },
      { icon: Hammer, title: "Weights workshop", hash: "#/experiments?workshop=1", tag: "Experiments → Weights → Make a variant",
        what: "Build your own versions of a model's weights with llama.cpp's tools.",
        can: ["Any quantization type (2–8 bit, K- and IQ-quants)", "Importance matrix (calibrated on separate text, so the test isn't leaked)",
          "Per-tensor precision: keep attention, embeddings or the output head at 8-bit", "Remove layers (pruning)", "Each file gets a recipe that appears in the run's provenance"] },
      { icon: Radar, title: "Layer sensitivity scan", hash: "#/experiments?workshop=1", tag: "In the weights workshop",
        what: "Finds which layers are fragile and which can be squeezed.",
        can: ["Quantizes (or removes) one layer at a time and measures the drift", "Control run gives the noise floor", "Bar chart per layer; one click protects the most sensitive ones"] },
      { icon: Cpu, title: "Engines & frameworks", hash: "#/experiments", tag: "Experiments → Inference engine",
        what: "The software that runs the model, as a variable you can test.",
        can: ["llama.cpp: build any version (upstream, a branch, a commit, your own fork) and compare it with the installed one",
          "vLLM, SGLang and TensorRT-LLM: run original Hugging Face checkpoints on the same tests (vLLM needs an RTX 20-series or newer GPU)",
          "Several machines: llama.cpp RPC workers (another Blocky lends its GPU) or vLLM pipeline stages over Ray",
          "Full training in Internals: every weight or from scratch, on one GPU, several (FSDP) or several machines",
          "Several GPUs: GPU count, llama.cpp layer or row (tensor-parallel) split, tensor-parallel size for vLLM and SGLang",
          "Serving speed is measured the same way for every engine, so frameworks compare fairly",
          "Optional GPU kernel profile: time per operation (matrix multiply, attention, …) from an Nsight trace",
          "Backends: CUDA (NVIDIA), Vulkan (AMD, Intel and most other GPUs, including integrated) or CPU only",
          "Only add repositories you trust: building runs their code"] },
      { icon: Smartphone, title: "Cheap-device simulation", hash: "#/experiments", tag: "Experiments → Target device",
        what: "Measure how a model would run on a cheaper device before buying one.",
        can: ["Presets: single-board computer, mini PC, laptop without a GPU, 4 GB and 6 GB GPUs", "CPU only (no GPU at all), fewer cores (pinned), RAM and VRAM limits: a run that wouldn't fit fails",
          "Projected decode speed at the target's memory bandwidth, because speed itself can't be simulated", "Mix freely with any other setting, or sweep across presets"] },
      { icon: Network, title: "Remote devices", hash: "#/settings", tag: "Settings → Remote access & devices · Experiments → Run on another device",
        what: "One Blocky drives others, e.g. a few cheap boxes, and collects their results in one experiment.",
        can: ["Pick setups and profile them on another device with the same protocol", "Results come back labeled with that device's hardware, next to your runs",
          "Every Blocky is local-only until its owner turns on remote access (token required)", "Over untrusted networks, connect through an SSH tunnel instead"] },
      { icon: PlugZap, title: "Energy & cost", hash: "#/settings", tag: "Every run · set up in Settings",
        what: "Energy per token and what running a model costs.",
        can: ["Dedicated measurement: idle power with the model loaded, then ~15 s of sustained generation", "Energy per token, gross and above idle, tokens per watt-hour",
          "Sources: a wall-power smart plug (Shelly, Tasmota, any JSON URL), else GPU and CPU sensors", "With prices set: electricity per 1M tokens, speed per device price"] },
      { icon: Microscope, title: "Internals", hash: "#/internals", what: "The PyTorch side: look inside a model and fine-tune it.",
        can: ["Download original Hugging Face weights", "Activations: signal size per layer and token, how much each layer changes it, attention spread",
          "Logit lens: what the model would answer after each layer", "Layer knockout: skip one layer at a time and see how the answer changes",
          "GPU time per layer, split into attention and feed-forward, for prefill and decode",
          "Technique playground: simulate GPTQ, AWQ-style, RTN, NF4, ternary or your own Python quantization (optionally Hessian-aware), plus a quantized KV cache; scored by KL divergence, perplexity, and per-layer weight and output error",
          "Export a winning method to a real GGUF (near-exact on the Q4_0 grid) and profile it against the original in Experiments",
          "LoRA or QLoRA (4-bit) fine-tuning → merged → converted to GGUF → test it in Experiments"] },
      { icon: BookOpen, title: "Notebooks", hash: "#/notebooks", what: "JupyterLab on the box and a Python client for everything above.",
        can: ["Every result as a pandas table", "Queue runs, sweeps, weight builds, scans and engine builds from code",
          "Send experiments to MLflow or Weights & Biases (local / offline by default)", "REST API reference at /docs"] },
      { icon: LayoutDashboard, title: "Overview", hash: "#/home", what: "Live dashboard of the box: model, speed, memory, recent activity.", can: ["What's loaded and how fast it is", "GPU and RAM use at a glance"] },
    ],
  },
  {
    group: "Tools", intro: "Focused tools you can use on their own.",
    items: [
      { icon: Boxes, title: "Models", hash: "#/models", what: "The models on the box and where to get more.",
        can: ["Switch model, context size, KV cache, GPU layers (with automatic rollback)", "Download from Hugging Face, with a fit estimate for this machine",
          "Inside the model: the real layer structure from the file", "Compression lab: compare quantizations of one model"] },
      { icon: Swords, title: "Arena", hash: "#/arena", what: "Two models (or settings) answer the same prompt side by side.", can: ["Blind mode and votes → leaderboard", "Time, speed and energy per answer", "Thinking vs no thinking"] },
      { icon: ClipboardCheck, title: "Quality tests", hash: "#/tests", what: "Benchmarks and freshly generated questions.",
        can: ["GSM8K, ARC, MMLU-Pro, MATH-500, AIME, HumanEval+, MBPP+, CRUXEval, LiveCodeBench", "Fresh questions that can't be memorized; a memorization check",
          "Code runs in a locked-down sandbox", "Import your own benchmark from Hugging Face"] },
      { icon: Gauge, title: "Speed tests", hash: "#/bench", what: "Raw speed with llama-bench and more.",
        can: ["Prompt reading and writing speed, at any context depth", "Many users at once (throughput)", "Draft speed-up (speculative decoding)", "Power and tokens per joule where the GPU reports power"] },
    ],
  },
  {
    group: "Everyday use & the device", intro: "Blocky as your private local AI.",
    items: [
      { icon: MessageSquare, title: "Chat", hash: "#/chat", what: "Private chat with the loaded model; nothing leaves this machine.",
        can: ["Thinking on/off and a thinking budget", "Inspector: sampling settings, live speed, context meter", "Confidence view: per-word probability and the alternatives", "Search old chats, export as Markdown"] },
      { icon: Brain, title: "Memory", hash: "#/memory", what: "Facts Blocky remembers about you across chats.", can: ["See, edit and delete memories", "Memories are added after each reply"] },
      { icon: Activity, title: "System", hash: "#/system", what: "Live CPU, GPU, RAM, VRAM and the model server's log.", can: ["Spot memory pressure or errors", "Filter the server log"] },
      { icon: SettingsIcon, title: "Settings", hash: "#/settings", what: "Look and feel, energy prices and meters, AI assistants.", can: ["Light / dark theme, accent color", "Guided / Researcher workspace (bottom-left switch); all tools remain accessible", "Wall-power meter and prices for energy and cost", "How to connect an AI assistant (MCP)"] },
    ],
  },
];

const MEASURES = [
  ["Load", "Memory layout (GPU / system RAM, per buffer), layers on GPU, load time", "llama.cpp's own load log", "—"],
  ["Quality", "MMLU-Pro: fixed questions, 5-shot; sample size set per experiment", "Scores option letters from returned top-k log probabilities; this is not a full official benchmark reproduction", "95% interval; paired McNemar test vs compatible baseline and parent"],
  ["Long context", "A code hidden at 5 depths in texts from 2k tokens up to the run's context", "Exact retrieval of the code", "Fixed seed: identical haystacks for every run"],
  ["lm-eval (optional)", "Any generative lm-evaluation-harness task you add, e.g. gsm8k, ifeval", "The field's standard harness, against this run's own server", "Its standard error"],
  ["Serving speed", "Output throughput over the full request and time to first content chunk; prompt/latency proxy", "Streaming through the chat API; server-reported token counts only", "Median of available repetitions after a warm-up; descriptive spread"],
  ["Speed (llama.cpp)", "Prompt reading and writing speed, at 0 and 4k tokens of context", "llama-bench, 5 repetitions", "Median; robust spread; only non-finite/non-positive samples dropped"],
  ["GPU kernels (optional)", "Time per operation type: matrix multiply, attention, normalization, …", "Nsight Systems trace of short llama-bench runs", "Regular Linux only; not available in WSL"],
  ["Fidelity", "Perplexity, and KL divergence + same-top-token vs the baseline's outputs", "llama-perplexity on WikiText-2 (16k tokens)", "Its ± values; only between models with the same tokenizer"],
  ["Telemetry", "GPU and memory-controller load, clocks, temperature, VRAM, CPU, RAM, power", "nvidia-smi and the OS, every 0.5 s", "Per phase"],
  ["Hardware analysis", "Estimated roofline utilization and possible bottlenecks; estimated weight-read time", "Simplified weight-traffic model vs nominal hardware limits", "Only total time is measured; a kernel profile is needed to investigate causes"],
];

const CANVAS = [
  ["Move around", "Drag the background · scroll to pan · ⌃/⌘ + scroll or pinch to zoom · Fit / Tidy up buttons"],
  ["Branch from a run", "Hover a card and click its + · or right-click → Branch from here. The new run starts from that card's exact setup."],
  ["Compare two runs", "Click one card, shift-click another → side-by-side panel with every metric"],
  ["Right-click a card", "Branch · Repeat · Compare with parent / baseline / selected · Copy settings · Stop · Delete (branches re-attach to its parent)"],
  ["Right-click empty space", "Branch from baseline · Add note here · Fit · Tidy up"],
  ["Notes", "Double-click empty space. Notes and card positions are saved on the box."],
  ["Δ vs parent / baseline", "Top-right switch: show each card's changes against the run it came from, or against the baseline"],
];

const LIMITS = [
  "Energy needs a power reading: a wall meter (most accurate: the whole device), or GPU / CPU sensors where the drivers expose them. Without one, Blocky says energy wasn't measured instead of guessing.",
  "Cheap-device presets simulate capacity (RAM, VRAM) and core count, not per-core speed or memory bandwidth: measured speeds are an upper bound, and the projection covers decode only. CPU-only runs of the full protocol can take an hour or more.",
  "Other programs using the same GPU add noise to speed measurements. Use “Repeat this setup” to measure it; Blocky then only highlights bigger changes.",
  "560 MMLU-Pro questions detect differences of roughly 3–4 points; the 95% interval of each difference shows how large a change could hide. For smaller effects use 1,400 or 3,500 questions, or KL divergence, which is far more sensitive.",
  "KL divergence and perplexity only compare models with the same tokenizer; others show “not comparable”. They're measured with llama.cpp's tools, so vLLM and SGLang runs don't have them.",
  "With several GPUs the bandwidth and compute limits assume identical GPUs. Layer split runs the GPUs one after another for one sequence, so its limit is one GPU's bandwidth; row split and tensor parallelism add them up.",
  "vLLM needs an NVIDIA GPU of the RTX 20-series or newer, and loads slowly the first time (it compiles kernels for the model).",
  "GPU kernel profiles need CUDA kernel tracing, which works on a regular Linux install but not inside WSL.",
  "lm-eval runs generative tasks only (the model writes an answer). Multiple-choice tasks scored by log-probabilities need features this server doesn't offer; MMLU-Pro covers that style.",
  "Fine-tuning at full precision needs a lot of GPU memory: as a rule of thumb, an 8 GB GPU handles models up to about 1–2B parameters; bigger models need bigger GPUs.",
  "Change one thing at a time. With several changes at once, Blocky warns you: effects can't be told apart.",
];

const EXTRA_TERMS: [string, string][] = [
  ["Baseline", "The first run of an experiment: the reference point. Always shown with its model, e.g. “Baseline · Qwen3.5-9B-Q8_0”."],
  ["Parent run", "The run a new run was branched from. Its setup + one change = the new run."],
  ["KL divergence (KLD)", "How far a run's word-by-word probabilities drift from the baseline's. 0 = identical. The most sensitive quality signal."],
  ["Perplexity", "How surprised the model is by real text; lower is better. Only comparable between models with the same tokenizer."],
  ["McNemar test", "Statistics for “same 560 questions, two setups”: is the difference in right/wrong answers bigger than chance? p < 0.05 = yes."],
  ["Roofline", "Chart of achieved compute against the GPU's two limits: memory speed and peak compute."],
  ["Importance matrix", "Measures which weights matter most on sample text, so compression keeps those precise."],
  ["LoRA", "Fine-tuning by training small add-on matrices instead of the whole model; merged back in afterwards."],
  ["Logit lens", "Reading out what the model would predict after each layer, to see where an answer forms."],
  ["Engine", "The program that runs the model (llama.cpp). Different versions can be faster or slower for the same model."],
  ["Sweep", "Queueing every combination of chosen settings as runs."],
  ["Provenance", "Everything needed to reproduce a run: versions, file fingerprints, exact commands."],
];

function Section({ id, title, kicker, children }: { id: string; title: string; kicker?: string; children: React.ReactNode }) {
  return (
    <section id={id} className="gd-sec">
      {kicker && <div className="gd-kicker">{kicker}</div>}
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export default function GuidePage() {
  const [active, setActive] = useState(TOC[0].id);
  const main = useRef<HTMLDivElement>(null);
  const go: Go = (hash) => { window.location.hash = hash; };
  const jump = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

  useEffect(() => {
    const obs = new IntersectionObserver((entries) => {
      const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (vis[0]) setActive(vis[0].target.id);
    }, { rootMargin: "-10% 0px -70% 0px" });
    TOC.forEach((t) => { const el = document.getElementById(t.id); if (el) obs.observe(el); });
    return () => obs.disconnect();
  }, []);

  return (
    <div className="page">
      <div className="page-inner wide gd">
        <div className="gd-hero">
          <div className="gd-kicker">Guide</div>
          <h1 className="page-title">Everything Blocky does, and how it fits together</h1>
          <p className="page-sub">Blocky is your private local AI and a lab for testing local models on this exact hardware. This page explains every feature, the research workflow that connects them, and how the numbers are made.</p>
          <div className="gd-paths">
            <button onClick={() => go("#/chat")}><MessageSquare /><span><b>Just chat</b><em>Private AI on this machine</em></span><ArrowRight /></button>
            <button onClick={() => jump("g-workflow")}><FlaskConical /><span><b>Test a model</b><em>The research workflow</em></span><ArrowRight /></button>
            <button onClick={() => go("#/internals")}><Microscope /><span><b>Look inside a model</b><em>Activations and fine-tuning</em></span><ArrowRight /></button>
          </div>
        </div>

        <div className="gd-layout">
          <nav className="gd-toc" aria-label="Guide contents">
            {TOC.map((t) => <button key={t.id} className={active === t.id ? "on" : ""} onClick={() => jump(t.id)}>{t.label}</button>)}
          </nav>

          <div className="gd-main" ref={main}>
            <Section id="g-overview" title="What Blocky is" kicker="Overview">
              <p>Blocky turns a computer with a GPU into two things at once:</p>
              <div className="gd-two">
                <div className="gd-card"><MessageSquare /><b>A private AI</b><span>Chat, memory and search with open models, running entirely on this machine. Nothing is sent anywhere.</span></div>
                <div className="gd-card"><FlaskConical /><b>A research lab</b><span>Measure, compare and modify local models on this exact hardware, with methods a researcher would accept.</span></div>
              </div>
              <div className="gd-stack">
                <div><b>Model file</b><span>GGUF weights, e.g. Qwen3.5-9B-Q4_K_M</span></div><ArrowRight />
                <div><b>Engine</b><span>llama.cpp runs the model</span></div><ArrowRight />
                <div><b>Hardware</b><span>GPU memory, system RAM, CPU</span></div><ArrowRight />
                <div><b>Blocky</b><span>chat on top, measurements around it</span></div>
              </div>
              <p className="row-sub">Every one of those layers is something you can change and measure: the weights (workshop, fine-tuning), the engine (engines), the settings (experiments), and how the hardware is used (hardware analysis).</p>
            </Section>

            <Section id="g-start" title="Quick start" kicker="5 minutes">
              <ol className="gd-steps">
                <li><b>Open Blocky</b> in a browser on the device: <code>http://localhost:8000</code>. It starts automatically with the device.</li>
                <li><b>Chat</b>: <em>New chat</em> at the top left. Turn thinking on for harder questions.</li>
                <li><b>Pick or get a model</b> in <a onClick={() => go("#/models")}>Models</a>: switch what's loaded, or download from Hugging Face with a fit estimate.</li>
                <li><b>Choose your detail level</b>: Guided / Researcher at the bottom left. Guided offers steps and common settings; Researcher expands advanced details. Both keep the same tools. Underlined terms explain themselves on hover.</li>
                <li><b>Run your first experiment</b>: <a onClick={() => go("#/experiments")}>Experiments</a> → New experiment → Run baseline. About 5 minutes; chat pauses meanwhile.</li>
              </ol>
            </Section>

            <Section id="g-workflow" title="The research workflow" kicker="The loop">
              <p>Every research feature plugs into one loop. You always compare against something, and you change one thing at a time.</p>
              <div className="gd-flow">
                {STEPS.map((s) => (
                  <button key={s.n} className="gd-flow-step" onClick={() => go(s.hash)}>
                    <span className="gd-n">{s.n}</span>
                    <s.icon />
                    <b>{s.title}</b>
                    <span>{s.text}</span>
                    <em>{s.where}</em>
                  </button>
                ))}
              </div>
              <div className="gd-loop"><Repeat size={15} /> Steps 3–6 repeat: each new run can branch from the best run so far, so the experiment grows into a tree. The canvas shows that tree.</div>
              <div className="gd-callout"><b>Results leave the box too:</b> CSV / JSON export on every experiment, the Python client, and notebooks for your own analysis.</div>
              <div className="gd-callout"><b>Comparing hardware:</b> run the same experiment on two Blocky devices, then use <em>Export → JSON</em> on one and <em>Import</em> on the other, either as a new experiment or into an existing one. The imported runs keep their own hardware snapshot and appear in the same table and canvas, labeled with the device they ran on.</div>
            </Section>

            <Section id="g-example" title="Worked example: does 3-bit hurt, and can we fix it?" kicker="An example study">
              <p>Use a high-precision model as the reference, then compare lower-precision variants on your device. This is a workflow template; results come from your experiment, not from preset example numbers.</p>
              <div className="table-wrap"><table className="data gd-table">
                <thead><tr><th>Step</th><th>What to run</th><th>What to check</th></tr></thead>
                <tbody>
                  <tr><td>1</td><td>Profile the high-precision baseline</td><td>Record memory placement, speed spread, quality scores, and provenance.</td></tr>
                  <tr><td>2</td><td>Sweep compatible 4-bit and 3-bit variants</td><td>Compare quality and speed with uncertainty; smaller weights do not guarantee faster inference.</td></tr>
                  <tr><td>3</td><td>Build a variant with an importance matrix</td><td>Measure whether drift changes against the same reference distribution.</td></tr>
                  <tr><td>4</td><td>Branch from one variant and change KV precision</td><td>Compare with that parent and repeat the long-context tests.</td></tr>
                  <tr><td>5</td><td>Repeat the setup with another engine build</td><td>Record the exact engine commit; check decode, prefill, and quality separately.</td></tr>
                </tbody>
              </table></div>
              <p className="row-sub">Choose a setup only after checking your measured tradeoffs. Failure to detect a quality difference does not establish that two configurations are equivalent.</p>
            </Section>

            <Section id="g-features" title="Every feature" kicker="Page by page">
              {FEATURES.map((g) => (
                <div key={g.group} className="gd-group">
                  <div className="gd-group-head"><h3>{g.group}</h3><span className="row-sub">{g.intro}</span></div>
                  <div className="gd-grid">
                    {g.items.map((f) => (
                      <div key={f.title} className="gd-feat">
                        <div className="gd-feat-head"><span className="gd-ic"><f.icon /></span><div><b>{f.title}</b>{f.tag && <em>{f.tag}</em>}</div></div>
                        <p>{f.what}</p>
                        <ul>{f.can.map((c) => <li key={c}>{c}</li>)}</ul>
                        <button className="btn small ghost" onClick={() => go(f.hash)}>Open <ArrowRight size={13} /></button>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </Section>

            <Section id="g-measure" title="How runs are measured" kicker="One profile = all of this">
              <p>Every run in an experiment goes through the same phases, in the same order, with the same data. That's what makes runs comparable. A profile takes about 4–8 minutes for a 9B model; more with lm-eval tasks.</p>
              <div className="table-wrap"><table className="data gd-table">
                <thead><tr><th>Phase</th><th>What it measures</th><th>How</th><th>Noise & statistics</th></tr></thead>
                <tbody>{MEASURES.map((m) => <tr key={m[0]}><td><b>{m[0]}</b></td><td>{m[1]}</td><td>{m[2]}</td><td>{m[3]}</td></tr>)}</tbody>
              </table></div>
              <p className="row-sub">While a profile runs, the chat model is paused so nothing else uses the GPU. Runs wait in a queue and start one after another; chat comes back after the last one.</p>
            </Section>

            <Section id="g-reading" title="Reading the results" kicker="Colors mean something">
              <div className="gd-legend">
                <div><span className="d good">+8%</span><span>Better than the comparison run, and bigger than the measurement noise.</span></div>
                <div><span className="d bad">−12%</span><span>Worse, and bigger than the noise.</span></div>
                <div><span className="d same">+1.5%</span><span>Grey: statistical significance is unavailable or not established. This does not prove equivalence.</span></div>
                <div><span className="d same">not comparable</span><span>Token-level metrics between models that split text differently.</span></div>
              </div>
              <ul className="gd-list">
                <li><b>Speeds</b> count as different only beyond twice their combined spread. If you've repeated a setup, also beyond twice the drift between the two repeats.</li>
                <li><b>MMLU-Pro</b> uses a paired test on the same 560 questions (McNemar, p &lt; 0.05), against the baseline and against the run's parent.</li>
                <li><b>KL divergence</b> is the sensitive one: small drifts show up here long before accuracy moves.</li>
                <li><b>⚠ on a run</b> means its software environment differs from the baseline's: another driver, data file or protocol. See Reproducibility.</li>
                <li><b>“on RTX … · device”</b> under a run means it ran on different hardware than the baseline. Each run stores a snapshot of its hardware (GPU, memory speed, compute peak, CPU, RAM), so its limits and bottlenecks are always judged against the machine it actually ran on.</li>
                <li><b>“What limits each run”</b> names the binding limit (GPU memory speed, system RAM, or software overhead) with the numbers behind it.</li>
              </ul>
            </Section>

            <Section id="g-canvas" title="Working on the canvas" kicker="Experiments → Canvas">
              <p>The canvas shows an experiment as a tree: baseline on the left, each branch to the right, every line labeled with what changed. You can start new runs directly from it.</p>
              <div className="table-wrap"><table className="data gd-table">
                <tbody>{CANVAS.map((c) => <tr key={c[0]}><td><b>{c[0]}</b></td><td>{c[1]}</td></tr>)}</tbody>
              </table></div>
            </Section>

            <Section id="g-python" title="Python & notebooks" kicker="For scripting and analysis">
              <p>Everything in Experiments is available from Python. In <a onClick={() => go("#/notebooks")}>Notebooks</a> it's already installed; elsewhere, download <code>blocky.py</code> from the same page.</p>
              <pre className="xcmds gd-code">{`import blocky
b = blocky.Client()                              # this box
exp = b.experiment("quantization study")
df = exp.results()                               # one row per run, every metric
exp.branch(exp.run("Q4_K_M"), kv="q4_0")         # that run's setup + one change
exp.sweep({"ctx": [4096, 16384, 32768]})         # queue a grid
exp.wait()

b.variant("Qwen3.5-9B-Q8_0.gguf", "Q3_K_M", imatrix=True).wait()
b.scan("Qwen3.5-0.8B-Q8_0.gguf", mode="quantize", qtype="q2_k").wait()
b.build_engine("https://github.com/you/llama.cpp", "my-branch").wait()

exp.to_mlflow()                                  # → ./mlflow.db   (or a tracking server URI)
exp.to_wandb(project="blocky")                   # offline; upload later with \`wandb sync\``}</pre>
              <p className="row-sub">The full REST API is documented at <a href="/docs" target="_blank" rel="noreferrer">/docs</a>.</p>
            </Section>

            <Section id="g-assistants" title="Working with an AI assistant" kicker="Claude, ChatGPT and other agents">
              <p>Blocky is an <b>MCP server</b> (Model Context Protocol, the standard way AI assistants use tools). Connect the assistant you already use, and it can see and use all of Blocky: read every result and all stored data, plan the next experiment, queue runs and sweeps, build weight variants and engines, run benchmarks and quality tests, and anything else the web page can do, while you watch it happen in the UI. You decide what each assistant may do (Settings → AI assistants: read, write, or full including deleting). Its model and API keys stay in your own app: Blocky never contacts an AI provider by itself.</p>
              <div className="table-wrap"><table className="data gd-table">
                <thead><tr><th>Assistant</th><th>How to connect</th></tr></thead>
                <tbody>
                  <tr><td>Claude Code</td><td><code>claude mcp add --transport http blocky http://localhost:8000/mcp</code></td></tr>
                  <tr><td>Claude Desktop</td><td>Settings → Developer → Edit config, add <code>{`"blocky": {"command": "npx", "args": ["mcp-remote", "http://localhost:8000/mcp"]}`}</code> under <code>mcpServers</code> (needs Node.js)</td></tr>
                  <tr><td>VS Code, Cursor, other editors</td><td>Add an MCP server of type <code>http</code> with the URL <code>http://localhost:8000/mcp</code></td></tr>
                  <tr><td>Codex and other CLIs</td><td>Any client that supports Streamable HTTP: URL <code>http://localhost:8000/mcp</code></td></tr>
                  <tr><td>ChatGPT</td><td>Publish Blocky through an HTTPS tunnel and turn on Settings → AI assistants → ChatGPT, Claude on the web and phone. In ChatGPT, turn on Developer mode (Settings → Apps &amp; Connectors → Advanced settings), then create a connector with the address shown in Blocky and OAuth sign-in. Approve it on the Blocky page with the pairing code.</td></tr>
                  <tr><td>Claude on the web and phone</td><td>Same tunnel and switch, then in Claude: Settings → Connectors → Add custom connector with the address shown in Blocky. Approve it with the pairing code.</td></tr>
                </tbody>
              </table></div>
              <p><b>Most powerful: a coding agent on the Blocky device.</b> An agent like Claude Code or Codex running on the device itself gets a shell, files, Python and PyTorch as well as the lab tools, so it can write analysis scripts or a new quantization method, run them, and iterate. Open the repository's <code>research/</code> folder with it (in VS Code: connect with Remote-SSH, or the WSL extension when Blocky runs in WSL, then start Claude Code there). Its <code>CLAUDE.md</code> / <code>AGENTS.md</code> explain how to do research on Blocky, and <code>.mcp.json</code> connects the lab tools automatically.</p>
              <p>Use the computer that runs Blocky (or one you've given access to it). Then ask in plain words, for example:</p>
              <pre className="xcmds gd-code">{`"Look at my quantization experiment. Which variant is the best trade-off,
 and is the accuracy difference significant? Queue the one run that would
 tell us the most next."

"Find the layers of Qwen3.5-0.8B that are most sensitive to quantization,
 then build a Q3_K_M variant that keeps them at Q8_0 and profile it against Q4_K_M."`}</pre>
              <p className="row-sub">What it can do depends on the level you give it. <b>Read</b>: see everything, including read-only database queries (never passwords, tokens or API keys). <b>Write</b>: also start, queue, build and change things, including research jobs that run code on this machine; nothing is deleted. <b>Full</b>: also delete. No level can change who has access or read secrets. Each run still takes minutes of your machine, and the queue runs one profile at a time. A coding agent working on Blocky's own source code should read <code>AGENTS.md</code> in the repository.</p>
            </Section>

            <Section id="g-limits" title="Limits & good practice" kicker="Honest notes">
              <ul className="gd-list">{LIMITS.map((l) => <li key={l}>{l}</li>)}</ul>
            </Section>

            <Section id="g-glossary" title="Glossary" kicker="Terms used across Blocky">
              <div className="gd-gloss">
                {EXTRA_TERMS.map(([t, d]) => <div key={t}><b>{t}</b><span>{d}</span></div>)}
                {Object.entries(GLOSSARY).map(([k, g]) => <div key={k}><b>{g.title}</b><span>{g.what} {g.why}</span></div>)}
              </div>
            </Section>

            <Section id="g-hood" title="Under the hood" kicker="For developers">
              <div className="table-wrap"><table className="data gd-table">
                <thead><tr><th>Part</th><th>Where</th></tr></thead>
                <tbody>
                  <tr><td>Web app + API</td><td>FastAPI on port 8000 (service <code>blocky-web</code>); React UI served by it</td></tr>
                  <tr><td>MCP server</td><td><code>/mcp</code> on the same port (Streamable HTTP; tools in <code>backend/app/mcp_server.py</code>)</td></tr>
                  <tr><td>Chat model server</td><td>llama-server on port 8080 (service <code>blocky-llm</code>); settings in <code>/opt/blocky/data/llm.env</code></td></tr>
                  <tr><td>Profiles, arena, tests</td><td>Temporary llama-server on port 8081, started per run</td></tr>
                  <tr><td>Notebooks</td><td>JupyterLab on 127.0.0.1:8888 (service <code>blocky-notebook</code>), token in <code>/opt/blocky/data/notebook-token</code></td></tr>
                  <tr><td>Models</td><td><code>/opt/blocky/models</code> (GGUF + <code>.recipe.json</code> for built variants)</td></tr>
                  <tr><td>Engines</td><td><code>/opt/blocky/engines/&lt;name&gt;/bin</code></td></tr>
                  <tr><td>PyTorch lab</td><td><code>/opt/blocky/ml-venv</code>; checkpoints in <code>/opt/blocky/hf</code></td></tr>
                  <tr><td>Data</td><td>SQLite + results in <code>/opt/blocky/data</code> (back up with <code>scripts/backup.sh</code>)</td></tr>
                  <tr><td>Setup</td><td><code>scripts/install.sh</code>, <code>scripts/setup-ml.sh</code>, <code>scripts/setup-notebooks.sh</code>; see docs/SETUP.md in the repository</td></tr>
                </tbody>
              </table></div>
            </Section>

            <div className="gd-end"><Wand2 size={16} /> Missing something? Everything here is in the repository, and every feature has an API, so it can be extended.</div>
          </div>
        </div>
      </div>
    </div>
  );
}
