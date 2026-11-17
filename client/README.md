# blockyos-client

A small, dependency-free Python client for a [blockyOS](https://github.com/matco01/blockyOS) device.

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
