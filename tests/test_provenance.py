"""Tests for result-file provenance stamping."""
import json

import pytest

from src.utils.provenance import META_KEY, describe, load_results, provenance, save_results


def test_provenance_records_what_identifies_the_code():
    m = provenance()
    assert set(m) >= {"generated_utc", "git_commit", "dirty", "script", "argv", "env"}
    assert isinstance(m["dirty"], bool)
    assert m["env"]["python"]
    # dirty must be consistent with the file list it is derived from
    assert m["dirty"] == bool(m["dirty_files"]) or len(m["dirty_files"]) == 20


def test_save_adds_meta_without_disturbing_payload(tmp_path):
    p = save_results(tmp_path / "r.json", {"acc": 0.9, "runs": [1, 2]})
    d = json.loads(p.read_text())
    assert d["acc"] == 0.9 and d["runs"] == [1, 2]      # readers by key are unaffected
    assert META_KEY in d


def test_round_trip_strips_meta(tmp_path):
    payload = {"acc": 0.9, "nested": {"a": 1}}
    p = save_results(tmp_path / "r.json", payload)
    got, meta = load_results(p)
    assert got == payload                               # exactly what went in
    assert meta["git_commit"] is None or isinstance(meta["git_commit"], str)


def test_refuses_payloads_it_cannot_stamp(tmp_path):
    with pytest.raises(TypeError):
        save_results(tmp_path / "a.json", [1, 2, 3])          # not a dict
    with pytest.raises(ValueError):
        save_results(tmp_path / "b.json", {META_KEY: {}})     # would clobber


def test_describe_flags_missing_provenance(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"acc": 0.5}))
    assert "NO PROVENANCE" in describe(p)


def test_describe_flags_legacy_unknown_marker(tmp_path):
    """Files stamped as pre-tracking must read as untrustworthy, not as blanks."""
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({META_KEY: {"provenance": "unknown"}, "acc": 0.5}))
    assert "NO PROVENANCE" in describe(p)


def test_describe_flags_a_dirty_tree(tmp_path, monkeypatch):
    import src.utils.provenance as prov
    monkeypatch.setattr(prov, "_git", lambda *a: "deadbeef" if a[0] == "rev-parse" else " M x.py")
    p = save_results(tmp_path / "r.json", {"acc": 1.0})
    assert "DIRTY" in describe(p)
