"""``hbench check``: the registry of gates and sanity checks, and its plug-in point."""

import importlib.util
import json

import pytest

from hallucination_bench import DATA, checks, cli


def passing(colour: str = "green"):
    """A plug-in check that always passes."""
    return True, f"all {colour}"


def failing(needed: str):
    """A plug-in check that always fails."""
    return False, f"saw {needed}"


@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(checks.REGISTRY)
    yield
    checks.REGISTRY.clear()
    checks.REGISTRY.update(saved)


def test_list_prints_every_check(capsys):
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    for check in checks.REGISTRY.values():
        assert callable(check.function())
        assert check.name in out and check.guards in out
    assert "readouts_contract readouts=" in out
    assert "readouts_contract" not in checks.offline()


@pytest.mark.parametrize("name", ["bank_manifests", "index_manifest", "calibration_sets"])
def test_offline_checks_pass_on_the_repo(name, capsys):
    assert cli.main(["check", name]) == 0
    assert capsys.readouterr().out.startswith(f"PASS  {name}: ")


def test_bank_manifests_fail_on_a_changed_file(tmp_path, monkeypatch, capsys):
    bank = tmp_path / "toy"
    bank.mkdir()
    (bank / "items.json").write_text("[]")
    (bank / "MANIFEST.sha256").write_text(f"{'0' * 64}  items.json\n")
    monkeypatch.setattr(checks, "DATA", tmp_path)
    assert cli.main(["check", "bank_manifests"]) == 1
    assert "toy/items.json differs from its MANIFEST.sha256" in capsys.readouterr().out
    (bank / "MANIFEST.sha256").write_text(f"{'0' * 64}  gone.json\n")
    assert cli.main(["check", "bank_manifests"]) == 1
    assert "listed but missing" in capsys.readouterr().out


def test_calibration_sets_fail_on_a_missing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(checks, "DATA", tmp_path)
    assert cli.main(["check", "calibration_sets"]) == 1
    assert "data/calibration/dev.jsonl is missing" in capsys.readouterr().out


def test_readouts_contract(tmp_path, capsys):
    good = tmp_path / "good.jsonl"
    row = {"id": "a", "layer": 44, "pos": 1, "samples": ["x"]}
    good.write_text(json.dumps(row) + "\n")
    assert cli.main(["check", "readouts_contract", "--readouts", str(good)]) == 0
    assert "1 cells of 1 items, kind prose" in capsys.readouterr().out
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(row) + "\nnot json\n")
    assert cli.main(["check", "readouts_contract", "--readouts", str(bad)]) == 1
    assert '"malformed": 1' in capsys.readouterr().out


def test_pytest_outcome_reports_the_last_line():
    ok, why = checks.pytest_outcome("tests/test_registry_pins.py")
    assert ok and "passed" in why
    ok, _why = checks.pytest_outcome("tests/no_such_test_file.py")
    assert not ok


def test_dry_run_is_listed_only_with_the_try_command(capsys):
    if importlib.util.find_spec("hallucination_bench.tryout") is None:
        assert "dry_run" not in checks.REGISTRY
        return
    assert cli.main(["check", "dry_run", "--family", "story"]) == 0
    assert "story: prompts built" in capsys.readouterr().out


def test_usage_errors():
    with pytest.raises(SystemExit, match="unknown check"):
        cli.main(["check", "no_such_check"])
    with pytest.raises(SystemExit, match="needs readouts="):
        cli.main(["check", "readouts_contract"])
    with pytest.raises(SystemExit, match="does not take"):
        cli.main(["check", "bank_manifests", "--family", "story"])
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        cli.main(["check", "bank_manifests", "--option", "novalue"])


def test_plugin_checks_by_dotted_path(monkeypatch, capsys):
    here = "test_check_command"
    monkeypatch.setenv(checks.PLUGIN_ENV, f"ok={here}:passing, bad={here}:failing")
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "ok [colour=]\n    A plug-in check that always passes." in out
    assert cli.main(["check", "ok", "--option", "colour=blue"]) == 0
    assert capsys.readouterr().out == "PASS  ok: all blue\n"
    assert cli.main(["check", "bad", "--option", "needed=red"]) == 1
    assert capsys.readouterr().out == "FAIL  bad: saw red\n"
    monkeypatch.setenv(checks.PLUGIN_ENV, "nodots")
    with pytest.raises(SystemExit, match="expected name="):
        cli.main(["check"])


def test_check_all_runs_only_optionless_checks(capsys):
    checks.REGISTRY.clear()
    here = "test_check_command"
    checks.register_check("ok", "Always passes.", f"{here}:passing")
    checks.register_check("bad", "Always fails.", f"{here}:failing")
    assert cli.main(["check", "all"]) == 0
    assert capsys.readouterr().out == (
        "PASS  ok: all green\n"
        "SKIP  bad: skipped (needs needed=...); run: hbench check bad needed=...\n"
    )
    with pytest.raises(ValueError, match="module:function"):
        checks.register_check("x", "y", "no_colon")
    assert (DATA / "index" / "v8_manifest.json").is_file()


def test_check_all_names_what_it_skips(monkeypatch, capsys):
    monkeypatch.setattr(checks, "offline", lambda: [])  # run nothing: only the skipped lines
    assert cli.main(["check", "all"]) == 0
    assert capsys.readouterr().out == (
        "SKIP  readouts_contract: skipped (needs readouts=...); "
        "run: hbench check readouts_contract readouts=...\n"
    )


def test_a_skipped_test_is_named_in_the_check_line(tmp_path, monkeypatch):
    test = tmp_path / "test_skips.py"
    test.write_text(
        "import pytest\n\ndef test_ok():\n    pass\n\n"
        "def test_needs_a_clone():\n    pytest.skip('no clone here')\n"
    )
    ok, why = checks.pytest_outcome(str(test))
    assert ok and "1 passed, 1 skipped" in why
    assert "NOT checked (skipped)" in why and "no clone here" in why
