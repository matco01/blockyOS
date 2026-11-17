"""A fake llama.cpp for end-to-end tests: llama-server, llama-bench and llama-perplexity in one script.

Installed as three executables (see install_fake_llama), it answers the same HTTP endpoints and prints the same output
formats as the real tools, so profile.run() can execute a complete experiment without a GPU or a model:

  llama-server      /health, /tokenize, /completion (option-letter log-probs), /v1/chat/completions (streaming, usage)
  llama-bench       -o jsonl lines with samples_ts
  llama-perplexity  "Final estimate: PPL" (baseline, writes the logits file) or the KL-divergence summary

The model's quality comes from its file name: the number of bits in the quantization type sets how often MMLU-Pro
answers are right and how large the KL divergence is, so a "Q2_K" file measures worse than a "Q8_0" one, consistently.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

BITS = [("F16", 16), ("BF16", 16), ("Q8_0", 8.5), ("Q6_K", 6.6), ("Q5_K", 5.5), ("Q4_K", 4.8), ("Q4_0", 4.5), ("Q3_K", 3.9), ("Q2_K", 2.6)]


def bits_of(model: str) -> float:
    name = Path(model).name.upper()
    return next((b for k, b in BITS if k in name), 16.0)


def accuracy(model: str) -> float:
    return 0.62 - 0.10 * max(0.0, (8.5 - bits_of(model))) / 6  # 0.62 at 8+ bits, ~0.52 at 2.6 bits


def kld(model: str) -> float:
    b = bits_of(model)
    return 0.0 if b >= 16 else round(0.3 * 2 ** (-(b - 2.0)), 6)


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def _h(*parts) -> float:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


# ---------------------------------------------------------------------------------------------------- server

def server(argv):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    if "--version" in argv:
        print("version: 9999 (c0ffee12)")
        print("built with fake for testing")
        return
    model, port = _arg(argv, "-m"), int(_arg(argv, "--port"))
    acc = accuracy(model)
    print("load_tensors: offloaded 33/33 layers to GPU", flush=True)
    print("load_tensors:        CUDA0 model buffer size =  3000.00 MiB", flush=True)
    print("llama_kv_cache:      CUDA0 KV buffer size =   512.00 MiB", flush=True)
    print("llama_context:      CUDA0 compute buffer size =   300.00 MiB", flush=True)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json({"status": "ok"})
            if self.path == "/props":
                return self._json({"model_path": model, "default_generation_settings": {"n_ctx": 8192}})
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/tokenize":
                text = body.get("content") or body.get("prompt") or ""
                return self._json({"tokens": [0] * max(1, len(text) // 4)})
            if self.path == "/completion":
                m = re.findall(r"\(key ([A-J])\)", body.get("prompt", ""))
                answer = m[-1] if m else "A"
                q = body.get("prompt", "")[-400:]
                pick = answer if _h(q, round(acc, 3)) < acc else "ABCDEFGHIJ"[("ABCDEFGHIJ".index(answer) + 1) % 10]
                top = [{"token": " " + pick, "logprob": -0.2}] + [{"token": " " + x, "logprob": -2.5 - i * 0.3}
                                                                   for i, x in enumerate("ABCD") if x != pick]
                return self._json({"content": " " + pick, "completion_probabilities": [{"top_logprobs": top}]})
            if self.path == "/v1/chat/completions":
                return self._chat(body)
            self._json({"error": "not found"}, 404)

        def _chat(self, body):
            text = (body.get("messages") or [{}])[-1].get("content", "")
            needle = re.search(r"The secret code for the (\w+) vault is (\d+)\.", text)
            words = [needle.group(2)] if needle else ["word"] * int(body.get("max_tokens") or 16)
            prompt_tokens = max(1, len(text) // 4)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            time.sleep(0.002 * min(prompt_tokens, 2000) / 100)  # "prefill"
            for i, w in enumerate(words):
                chunk = {"choices": [{"index": 0, "delta": {"content": w + ("" if needle else " ")}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                if i % 64 == 0:
                    self.wfile.flush()
                    time.sleep(0.001)
            final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                     "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(words)},
                     "timings": {"prompt_n": prompt_tokens, "prompt_per_second": 900.0, "predicted_n": len(words), "predicted_per_second": 40.0}}
            self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


# ---------------------------------------------------------------------------------------------------- bench

def bench(argv):
    model = _arg(argv, "-m")
    b = bits_of(model)
    tg = 600 / b  # decode is memory-bound: fewer bytes per weight, faster
    for depth in (0, 4096):
        slow = 0.9 if depth else 1.0
        pp_s = [round(1500 * slow * (1 + 0.01 * (i - 2)), 2) for i in range(5)]
        tg_s = [round(tg * slow * (1 + 0.005 * (i - 2)), 2) for i in range(5)]
        print(json.dumps({"n_prompt": 512, "n_gen": 0, "n_depth": depth, "samples_ts": pp_s, "avg_ts": sum(pp_s) / 5}))
        print(json.dumps({"n_prompt": 0, "n_gen": 128, "n_depth": depth, "samples_ts": tg_s, "avg_ts": sum(tg_s) / 5}))


# ---------------------------------------------------------------------------------------------------- perplexity

def perplexity(argv):
    model = _arg(argv, "-m")
    base = _arg(argv, "--kl-divergence-base")
    ppl = round(6.0 * (1 + 0.4 * kld(model)), 4)
    if base and "--kl-divergence" not in argv:
        Path(base).write_bytes(b"fake logits")
        print(f"Final estimate: PPL = {ppl:.4f} +/- 0.04120")
        return
    if base:
        if not Path(base).is_file():
            sys.exit("missing logits file")
        k = kld(model)
        print("====== Perplexity statistics ======")
        print(f"Mean PPL(Q)                   :   {ppl:.6f} ±   0.041200")
        print("====== KL divergence statistics ======")
        print(f"Mean    KLD:   {k:.6f} ±   {k * 0.04 + 1e-6:.6f}")
        print(f"99.0%   KLD:   {k * 9:.6f}")
        print("====== Token probability statistics ======")
        print(f"Same top p: {100 - k * 60:.3f} ±  0.120 %")
        return
    print(f"Final estimate: PPL = {ppl:.4f} +/- 0.04120")


def install_fake_llama(bin_dir: Path) -> Path:
    """Write llama-server, llama-bench and llama-perplexity into bin_dir, each running this file."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("llama-server", "llama-bench", "llama-perplexity"):
        exe = bin_dir / name
        exe.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
                       f"import fake_llama\nfake_llama.main({name!r}, sys.argv[1:])\n")
        exe.chmod(0o755)
    return bin_dir


def main(tool, argv):
    {"llama-server": server, "llama-bench": bench, "llama-perplexity": perplexity}[tool](argv)


if __name__ == "__main__":
    main(os.path.basename(sys.argv[1]), sys.argv[2:])
