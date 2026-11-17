"""Full training: request validation, the launch command (one GPU, FSDP via torchrun, several machines) and the MFU estimate."""
import json
from pathlib import Path

import pytest

from app import torchlab


@pytest.fixture
def ckpt(monkeypatch, tmp_path):
    hf = tmp_path / "hf"
    d = hf / "Org--Tiny"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2}))
    monkeypatch.setattr(torchlab, "HF", hf)
    return d


def test_defaults_depend_on_the_mode(ckpt):
    ft = torchlab.train_plan({"checkpoint": "Org--Tiny", "data": "wikitext:wikitext-2-raw-v1"}, 1)
    pt = torchlab.train_plan({"checkpoint": "Org--Tiny", "data": "wikitext:wikitext-2-raw-v1", "mode": "pretrain", "layers": 4}, 1)
    assert ft["lr"] == 2e-5 and pt["lr"] == 3e-4 and ft["eval_frac"] == 0.02
    assert ft["layers"] == 0 and pt["layers"] == 4          # layer override only applies when pretraining
    assert torchlab.train_plan({"checkpoint": "Org--Tiny", "data": "x/y", "eval_frac": 0}, 1)["eval_frac"] == 0.0


@pytest.mark.parametrize("body,msg", [
    ({"data": ""}, "Data"), ({"data": "../etc/passwd"}, "Data"), ({"data": "/nope/file.txt"}, "doesn't exist"),
    ({"data": "x/y", "mode": "rlhf"}, "Mode"), ({"data": "x/y", "gpus": 3}, "GPU"), ({"data": "x/y", "lr": 5}, "Learning rate"),
    ({"data": "x/y", "nnodes": 2}, "address"), ({"data": "x/y", "nnodes": 2, "master_addr": "127.0.0.1"}, "address"),
    ({"data": "x/y", "nnodes": 2, "master_addr": "10.0.0.1", "master_port": 80}, "Port"),
])
def test_rejects_bad_requests(ckpt, body, msg):
    with pytest.raises(ValueError, match=msg):
        torchlab.train_plan({"checkpoint": "Org--Tiny", **body}, 2)


def test_launch_commands(ckpt, tmp_path):
    out = tmp_path / "out"
    one = torchlab.train_args(torchlab.train_plan({"checkpoint": "Org--Tiny", "data": "x/y"}, 1), ckpt, out)
    assert one[0].endswith("train.py") and "--out" in one
    multi = torchlab.train_args(torchlab.train_plan({"checkpoint": "Org--Tiny", "data": "x/y", "gpus": 2}, 2), ckpt, out)
    assert multi[:2] == ["-m", "torch.distributed.run"] and multi[multi.index("--nproc_per_node") + 1] == "2" and "--standalone" in multi
    p = torchlab.train_plan({"checkpoint": "Org--Tiny", "data": "x/y", "gpus": 2, "nnodes": 3, "master_addr": "10.0.0.1"}, 2)
    nodes = torchlab.train_args(p, ckpt, out)
    assert nodes[nodes.index("--node_rank") + 1] == "0" and nodes[nodes.index("--master_addr") + 1] == "10.0.0.1"
    other = torchlab.other_node_command(p, nodes)
    assert "<node rank: 1 to 2>" in other and "--master_addr 10.0.0.1" in other and "--standalone" not in other
    assert torchlab.other_node_command(torchlab.train_plan({"checkpoint": "Org--Tiny", "data": "x/y"}, 1), one) is None


def test_mfu_estimate():
    # 1B parameters at 10k tokens/s = 6e13 FLOP/s; on 2 GPUs of 100 TFLOPS peak that's 30%.
    assert torchlab.mfu(10_000, 1_000_000_000, 2, 100) == 0.3
    assert torchlab.mfu(None, 1, 1, 1) is None and torchlab.mfu(1, 1, 1, None) is None


def test_progress_reader_skips_partial_lines(tmp_path):
    p = tmp_path / "progress.jsonl"
    p.write_text('{"event": "start", "params": 5}\n{"event": "eval", "step": 0, "eval_loss": 3.0}\n{"step": 1, "loss": 2.5}\n{"step": 2, "lo')
    steps, evals, info, end = torchlab._read_progress(p)
    assert [s["step"] for s in steps] == [1] and evals[0]["eval_loss"] == 3.0 and info["params"] == 5 and end is None
