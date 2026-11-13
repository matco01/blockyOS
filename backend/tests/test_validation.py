"""Inputs that reach the command line are validated: repos, refs, tensor patterns, layer numbers."""
import pytest

from app import engines, scans, weights


@pytest.mark.parametrize("repo", ["https://github.com/ggml-org/llama.cpp", "https://github.com/you/llama.cpp.git", "https://gitlab.example.org:8443/a/b"])
def test_engine_repo_accepts_https_urls(repo):
    assert engines._REPO.match(repo)


@pytest.mark.parametrize("repo", ["file:///etc", "git@github.com:a/b.git", "http://github.com/a/b", "https://github.com/a/b;rm -rf /", "--upload-pack=x"])
def test_engine_repo_rejects_everything_else(repo):
    assert not engines._REPO.match(repo)


@pytest.mark.parametrize("ref,ok", [("master", True), ("b6000", True), ("feature/new-kernel", True), ("8ca5a8008b57", True),
                                    ("--upload-pack=x", False), ("a b", False), ("x;y", False)])
def test_engine_ref(ref, ok):
    # The same rule engines.add() applies: allowed characters, no leading dash (git option injection), no "..".
    valid = bool(engines._REF.match(ref)) and not ref.startswith("-") and ".." not in ref
    assert valid == ok


def test_engine_bin_dir():
    assert engines.bin_dir("stock") == engines.lab.LLAMA_BIN
    with pytest.raises(ValueError):
        engines.bin_dir("../../etc")
    with pytest.raises(ValueError):
        engines.bin_dir("not-built")


def test_weights_validation(make_gguf):
    make_gguf("Base-Q8_0.gguf", ftype=7, layers=24)
    spec = weights._validate({"source": "Base-Q8_0.gguf", "qtype": "Q3_K_M", "imatrix": True, "overrides": {r"blk\.(0|23)\.": "q8_0", "attn_v": "q6_k"}, "prune": [12]})
    assert spec["out"] == "Base-Q3_K_M-imat-custom-pruned1.gguf"
    assert list(spec["overrides"]) == [r"blk\.(0|23)\.", "attn_v"]       # order kept: first match wins in llama-quantize
    assert weights._validate({"source": "Base-Q8_0.gguf", "qtype": "IQ2_XS"})["imatrix"] is True   # required for this type
    for bad in ({"source": "Base-Q8_0.gguf", "qtype": "Q9_K"},
                {"source": "Base-Q8_0.gguf", "qtype": "Q4_K_M", "overrides": {"attn_v; rm -rf /": "q8_0"}},
                {"source": "Base-Q8_0.gguf", "qtype": "Q4_K_M", "overrides": {"attn_v": "q9_9"}},
                {"source": "Base-Q8_0.gguf", "qtype": "Q4_K_M", "overrides": {"(unclosed": "q8_0"}},
                {"source": "Base-Q8_0.gguf", "qtype": "Q4_K_M", "prune": [24]},
                {"source": "../../etc/passwd", "qtype": "Q4_K_M"}):
        with pytest.raises(ValueError):
            weights._validate(bad)


def test_scan_rejects_bad_input(make_gguf):
    make_gguf("Scan.gguf")
    with pytest.raises(ValueError):
        scans.start({"model": "Scan.gguf", "mode": "melt"})
    with pytest.raises(ValueError):
        scans.start({"model": "Scan.gguf", "mode": "quantize", "qtype": "q9"})
    with pytest.raises(ValueError):
        scans.start({"model": "missing.gguf"})
