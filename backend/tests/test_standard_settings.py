"""Settings for numbers comparable with published ones: the whole perplexity text, and recorded reference values."""
import pytest

from app import instrument, profile


@pytest.fixture
def api(http_client, fresh_db, monkeypatch, tmp_path):
    monkeypatch.setattr(instrument, "REFERENCE_FILE", tmp_path / "reference-values.json")
    return http_client


def test_whole_text_perplexity_option(api, make_gguf):
    make_gguf("std.gguf")
    eid = api.post("/api/experiments", json={"name": "Std", "model": "std.gguf"}).json()["id"]
    assert api.put(f"/api/experiments/{eid}/protocol", json={"ppl_chunks": profile.PPL_FULL}).status_code == 200
    exp = profile.get_experiment(eid)
    assert profile.ppl_chunks(exp) == profile.PPL_FULL
    assert "whole text" in api.put(f"/api/experiments/{eid}/protocol", json={"ppl_chunks": 500}).json()["detail"]


def test_reference_values_form(api):
    sha = "ab" * 32
    good = {"model_sha256": sha, "metric": "ppl", "value": 7.74, "tolerance": 0.05, "source": "Author, Model card table 1, https://example.org"}
    assert api.post("/api/reference-values", json={**good, "model_sha256": "short"}).status_code == 400
    assert api.post("/api/reference-values", json={**good, "metric": "vibes"}).status_code == 400
    assert api.post("/api/reference-values", json={**good, "source": "trust me"}).status_code == 400
    assert api.post("/api/reference-values", json={**good, "tolerance": -1}).status_code == 400
    refs = api.post("/api/reference-values", json=good).json()["references"]
    assert len(refs) == 1 and refs[0]["value"] == 7.74
    # The instrument check reads the same file.
    assert instrument.load_references()[0]["model_sha256"] == sha
    assert api.get("/api/reference-values").json()["references"][0]["metric"] == "ppl"
    assert api.delete("/api/reference-values/0").json()["references"] == []
    assert api.delete("/api/reference-values/0").status_code == 400
