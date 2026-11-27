# blockyos-client

A small, dependency-free Python client for a [blockyOS](https://github.com/matco01/blockyOS) device, and the
`blocky` command.

## The `blocky` command

```bash
blocky measure Qwen3.5-0.8B-F16.gguf Qwen3.5-0.8B-Q4_K_M.gguf   # baseline + variant: queue, wait, print the table
blocky measure base.gguf --repeat 1 --project "Support bot" --csv results.csv
blocky results 13            # any experiment, by id or name (--csv, --json)
blocky check 13              # instrument checks; exit code 1 if one fails
blocky status                # chat model, lab activity, GPU temperature and safety pause
blocky models | experiments | projects
```

Every number comes with its spread or interval, and accuracy differences with the paired test against the baseline.
Another device: `--url http://box:8000 --token <remote-access token>` (or `BLOCKY_URL`, `BLOCKY_TOKEN`). Exit
codes: 0 done, 1 a run or check failed, 2 usage or connection error, so it fits CI jobs and scripts.

## Python

```python
import blocky

b = blocky.Client("http://localhost:8000")
exp = b.new_experiment("Qwen3-4B-Q8_0.gguf", name="KV cache types")
base = exp.branch()                      # the baseline
exp.sweep({"kv": ["f16", "q8_0", "q4_0"]}, parent=base)
exp.wait()
df = exp.results()                       # one row per run (pandas, with the `tables` extra)
```

`blocky_research` is the reporting SDK for custom research programs run by Blocky (see the repository's
docs/RESEARCH_PROJECTS.md).

Install: `pip install blockyos-client` (add `[tables]` for pandas results, `[mlflow]` or `[wandb]` for trackers).
Licensed under Apache-2.0.
