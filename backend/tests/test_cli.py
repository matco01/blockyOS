"""The `blocky` command, against the real API (requests go through the test server instead of the network)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "client"))
import blocky  # noqa: E402
import blocky_cli  # noqa: E402

from app import db  # noqa: E402


@pytest.fixture
def cli(http_client, fresh_db, monkeypatch):
    def req(self, method, path, body=None, params=None):
        r = http_client.request(method, path, params={k: v for k, v in (params or {}).items() if v is not None}, json=body)
        if r.status_code >= 400:
            raise blocky.BlockyError(f"{method} {path}: {r.status_code} {r.json().get('detail')}")
        return r.json() if r.content else None
    monkeypatch.setattr(blocky.Client, "_req", req)

    def run(*argv):
        return blocky_cli.main(list(argv), client=blocky.Client())
    return run


def test_measure_queues_baseline_repeat_and_variants(cli, make_gguf, capsys):
    make_gguf("base.gguf")
    make_gguf("small.gguf")
    assert cli("measure", "base.gguf", "missing.gguf") == 2
    assert "Not on the device: missing.gguf" in capsys.readouterr().err
    assert cli("measure", "base.gguf", "small.gguf", "--repeat", "1", "--project", "Demo", "--ppl-chunks", "64", "--no-wait") == 0
    eid = int(capsys.readouterr().out.strip())
    with db.connect() as c:
        runs = c.execute("SELECT config, parent_id FROM exp_runs WHERE experiment_id = ? ORDER BY id", (eid,)).fetchall()
        exp = c.execute("SELECT project_id, protocol FROM experiments WHERE id = ?", (eid,)).fetchone()
    assert [json.loads(r["config"])["model"] for r in runs] == ["base.gguf", "base.gguf", "small.gguf"]
    assert runs[0]["parent_id"] is None and runs[1]["parent_id"] == runs[2]["parent_id"]
    assert exp["project_id"] and json.loads(exp["protocol"])["ppl_chunks"] == 64
    assert cli("projects") == 0 and "Demo" in capsys.readouterr().out


def test_results_table_csv_and_check(cli, make_gguf, capsys, tmp_path):
    make_gguf("r.gguf")
    assert cli("measure", "r.gguf", "--no-wait") == 0
    eid = int(capsys.readouterr().out.strip())
    res = {"mmlu": {"acc": 0.6, "ci95": [0.56, 0.64]}, "speed": {"tg": 50.0, "tg_sd": 1.25, "pp": 1500.0, "pp_sd": 20.0}, "ppl": 7.8, "ppl_sd": 0.1}
    with db.connect() as c:
        c.execute("UPDATE exp_runs SET status = 'done', results = ? WHERE experiment_id = ?", (json.dumps(res), eid))
    out_csv = tmp_path / "r.csv"
    assert cli("results", str(eid), "--csv", str(out_csv)) == 0
    out = capsys.readouterr().out
    assert "60.0% [56.0, 64.0]" in out and "50.0 ± 1.2" in out and "7.800 ± 0.100" in out
    assert "speed.tg" in out_csv.read_text().splitlines()[0]
    assert cli("results", str(eid), "--json") == 0
    assert json.loads(capsys.readouterr().out)[0]["decode_tok_s"] == "50.0 ± 1.2"
    assert cli("check", str(eid)) == 0 and "self_kld" in capsys.readouterr().out


def test_wait_reports_failures(cli, make_gguf, capsys):
    make_gguf("w.gguf")
    cli("measure", "w.gguf", "--no-wait")
    eid = int(capsys.readouterr().out.strip())
    with db.connect() as c:
        c.execute("UPDATE exp_runs SET status = 'error', error = 'boom' WHERE experiment_id = ?", (eid,))
    assert blocky_cli.wait(blocky.Client().experiment(eid), poll=0) is False


def test_status(cli, capsys):
    assert cli("status") == 0 and "GPU:" in capsys.readouterr().out


def test_unreachable_device(capsys):
    assert blocky_cli.main(["--url", "http://127.0.0.1:9", "models"]) == 2   # nothing listens on port 9
    assert "can't reach" in capsys.readouterr().err
