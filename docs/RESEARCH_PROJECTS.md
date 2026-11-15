# Research projects

Research projects add custom Python programs and command-line jobs alongside Experiments, fine-tuning, the weight
workshop, and the existing model servers. Existing features and stored results keep their own workflows.

Open **Research projects** in the sidebar (`#/projects`). Create a byte baseline, conventional Transformers evaluator,
or blank project. Edit files in the page or open the displayed workspace path in your editor.

## Run an experiment

1. Implement the architecture, trainer, or evaluator in the project source directory. Other entry points, compiled
   executables and distributed launchers can be specified as an argument array; commands are executed without a shell.
2. Choose an existing Python interpreter, such as `/opt/blocky/ml-venv/bin/python` or your own project environment.
   Create a separate venv and install your requirements yourself; Blocky does not modify environments automatically.
3. Put model and dataset paths, seeds and method settings in the job's JSON configuration. Use absolute paths for
   large external assets. Source snapshots are limited to 500 files / 20 MB and exclude `.git`, `.venv`, and `__pycache__`.
4. Select GPU reservation for jobs that need it. All jobs take the lab's heavy-work lock. GPU jobs pause the chat
   model and restore it on completion, cancellation or failure. CPU jobs hide CUDA devices from cooperating libraries.
   This is scheduling, not enforcement against deliberately uncooperative research code. Existing profile queues
   have priority. Avoid GPU work in a job marked CPU.
5. Queue the job. Blocky snapshots its source, SDK, command and configuration before queueing. Editing the project
   afterward does not change queued jobs. The initial source hashes, installed package list and hardware fingerprint
   are recorded. The selected environment itself is not copied or locked; external data must be fingerprinted by the evaluator.
6. Follow logs and reported evidence. Download model checkpoints, plots and other artifacts reported by the SDK.
   “Branch settings” copies a prior job's configuration and command into the form; the next job uses the current source.

The programs run locally with the Blocky service account's permissions. They are trusted research code, not sandboxed
programs. Common Hugging Face libraries and Weights & Biases are set to offline mode; arbitrary programs can still
access files or the network. No downloads, package installation, cloud submission, or provider calls are performed
by the runner itself. Download and prepare assets separately.

Queued jobs persist across restarts. Interrupted jobs become errors. Cancellation and timeout terminate the process
group on Linux, including ordinary child processes; programs that detach themselves are outside that lifecycle.
Jobs keep their data and artifacts; no background deletion is provided.

## Report actual evidence

Every source snapshot includes the dependency-free `blocky_research.py` module from `client/`:

```python
from blocky_research import Run, fingerprint

run = Run()
settings = run.config
# Train or load your model, then evaluate it on a held-out dataset.
# 'score' and 'per_example_scores' below must come from that evaluation.
run.metric(
    "accuracy", score, unit="fraction", better="higher",
    protocol="my-exact-match-evaluator/1", task="my-held-out-task",
    dataset_sha256=fingerprint(settings["evaluation_file"]),
    scope="all held-out examples; exact match; deterministic decoding",
    samples=per_example_scores,
)
run.progress("Evaluation complete", step=100)
# Save a file under run.directory before registering it.
run.artifact(run.directory / "checkpoint.pt", "Trained model checkpoint")
```

Reporting writes JSONL on disk and makes no network calls. Metrics require finite values, explicit units, a measurement
protocol, task, dataset fingerprint and scope. Tag estimates/simulations with `kind="estimated"` or `"simulated"`.
Unavailable measurements should be omitted, not replaced with zero. Raw samples get a descriptive SD when at least
two exist; the runner does not infer independence, confidence intervals or significance for arbitrary samples.
Metric labels and contracts are declarations by the project evaluator, not certification by Blocky.

Artifacts must be inside `run.directory`; external assets can be referenced and fingerprinted instead of copied.
The UI exposes recent progress, metric evidence, the last 32 KB of stdout/stderr, source hashes and package versions.
Full stdout remains on disk as `stdout.log`; `environment.txt`, snapshot files and registered artifacts can be downloaded
through the artifact endpoint. Up to 10,000 report events are shown; save longer histories as artifacts.

## Compare architectures and conventional models

Select completed jobs from one or several projects and choose **Compare selected jobs**. You can also add existing
Experiments profile runs. Blocky checks metric name, dataset hash, task, unit, protocol, measurement scope, evidence
kind, direction and hardware metadata. Missing or mismatched contracts remain visible with a reason and do not become
an asserted improvement. Token-based quantities additionally require matching tokenizer fingerprints.

Built-in MMLU scores and llama-bench rates retain their original measurement protocols. Their token rates cannot be
converted to bytes/second from stored aggregates. Per-token perplexity and KL divergence cannot be converted into a
cross-tokenizer quality comparison from the available summaries. Run a shared evaluator when shared metrics are needed.

The byte-bigram and conventional Transformers templates demonstrate a shared text likelihood contract. They read the
same newline-delimited UTF-8 documents, score from each model's native start state, and report bits per evaluated byte.
The Transformers scorer evaluates the canonical token encoding rather than summing over every possible tokenization;
that limitation is part of its evidence. Timing includes encoding and complete scoring with the model already loaded,
and is evaluation throughput, not text generation throughput. Raw document samples and repeated timings are retained.

Included text is deliberately tiny pipeline smoke-test data. Replace it for research. The byte-bigram baseline is a
real count-trained model, not a BLT implementation. The conventional template requires a prepared local checkpoint and
its dependencies. Its checkpoint files are hashed before timing. Neither template controls the other model's pretraining
budget, and different code or kernels can affect timing. Matching evaluation contracts support comparison of these
implementations; they do not establish architectural superiority, quality equivalence or statistical significance.

For BLT or a new architecture, use a custom project with its actual model code and runtime. Implement the common
evaluation contract where appropriate, or define a new one. There is no requirement to export GGUF or expose a chat API.
BLT integration, model downloads, custom architecture correctness, distributed cluster management, resumable training,
and a universal automatic evaluator are not supplied by this first project runner.

## APIs and notebooks

- `GET/POST /api/research/projects`: list/create projects.
- `GET /api/research/projects/{id}/files` and `GET/PUT .../file`: inspect/edit source.
- `GET/POST /api/research/projects/{id}/jobs`: read/queue jobs.
- `POST /api/research/jobs/{id}/cancel`: cancel a queued or running job.
- `GET /api/research/jobs/{id}/artifact?name=...`: download a file within that run.
- `GET /api/research/compare?ids=1,2,profile:40`: compare jobs and built-in profiles.

These local HTTP APIs are usable from notebooks and custom clients. MCP exposes project/job discovery and comparisons.
Arbitrary code execution remains an explicit UI/API action rather than an automatically callable MCP execution tool.

No existing measurement procedure changed, so the built-in profile protocol is unchanged. The custom runner records
`blocky-research/1`; custom metrics version their own evaluators.
