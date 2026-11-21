"""``hbench try``: the two-question try-out on the bundled example readouts (dry run)."""

import json

import pytest

from hallucination_bench import cli, tryout
from hallucination_bench.evals import bank_for


@pytest.mark.parametrize("family", tryout.FAMILIES)
def test_example_readouts_follow_the_bank(family):
    by = {it["id"]: it for it in bank_for(family).load_items()}
    rows = [json.loads(x) for x in tryout.example_readouts(family).read_text().splitlines()]
    assert len({r["id"] for r in rows}) == 2
    for r in rows:
        assert r["pos"] in {int(s["pos"]) for s in by[r["id"]]["sites"]}
        assert r["samples"] and all(isinstance(s, str) and s for s in r["samples"])


def test_pick_keeps_bank_order_and_only_the_picked_rows(tmp_path):
    items, rows = tryout.pick("wsbench_chat", tryout.example_readouts("wsbench_chat"), 1)
    assert [it["id"] for it in items] == ["chat-dailydialog-0000"]
    assert {r["id"] for r in rows} == {"chat-dailydialog-0000"}
    stranger = tmp_path / "x.jsonl"
    stranger.write_text('{"id": "nope", "layer": 44, "pos": 1, "samples": ["x"]}\nnot json\n')
    with pytest.raises(SystemExit, match="no readouts for any wsbench_chat item"):
        tryout.pick("wsbench_chat", stranger, 2)


@pytest.mark.parametrize("family", tryout.FAMILIES)
def test_try_dry_run_end_to_end(family, tmp_path, capsys):
    out = tmp_path / "try"
    assert cli.main(["try", "--family", family, "--dry-run", "--out", str(out)]) == 0
    shown = capsys.readouterr().out
    assert f"== hbench try: {family}, 2 item(s)" in shown
    assert "question:" in shown and "expected:" in shown
    assert "== the first judge prompt" in shown and "== report" in shown
    arm = out / "judged" / family / "claude-opus-5-5" / "try"
    picked = out / "readouts" / "try" / "hallucination.jsonl"
    assert len(picked.read_text().splitlines()) == 6
    for path in (picked, arm / "prompts_sample.txt", arm / "cache.jsonl", arm / "summary.json"):
        assert path.is_file() and str(path) in shown
    assert "(not written)" not in shown
    summary = json.loads((arm / "summary.json").read_text())
    assert summary["health"]["n_judged_cells"] == 0  # a dry run answers nothing
    assert summary["spend"]["total_usd"] == 0
    # nothing that reads like a failed run, and the claim-map prompt of record
    assert "failed" not in shown and "preflight" not in shown
    assert "6 cell(s) would be judged with claimmap-v2.9" in shown
    assert json.loads((arm / "arm.json").read_text())["map_prompt"] == "claimmap-v2.9"


def test_try_fewer_items_and_usage_errors(tmp_path, capsys):
    assert cli.main(["try", "-n", "5", "--dry-run", "--out", str(tmp_path)]) == 0
    assert "readouts for 2 item(s) only, asked for 5" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no readouts file"):
        cli.main(["try", "--readouts", str(tmp_path / "missing.jsonl")])
    with pytest.raises(SystemExit, match="-n must be >= 1"):
        cli.main(["try", "-n", "0"])


def _dry_then(codes, monkeypatch):
    """``cli.cmd_run`` replaced: the first call builds the arm directory with the dry-run client
    (no API), every call returns the next of ``codes``."""
    real_run, calls = cli.cmd_run, []

    def fake_run(args, llm=None):
        if not calls:
            real_run(args, llm=tryout.DryRunLLM())
        calls.append(args.map_prompt)
        return codes[len(calls) - 1]

    monkeypatch.setattr(cli, "cmd_run", fake_run)
    return calls


def test_live_try_asks_failed_calls_again(tmp_path, monkeypatch, capsys):
    # exit 4 = failed calls remain: the try-out repeats the run, as the full-run driver does
    calls = _dry_then([4, 0], monkeypatch)
    assert cli.main(["try", f"out={tmp_path}"]) == 0
    assert calls == ["claimmap-v2.9"] * 2
    assert "attempt 2 of 3" in capsys.readouterr().out


def test_live_try_still_failing_shows_the_report_and_exits_4(tmp_path, monkeypatch, capsys):
    calls = _dry_then([4, 4, 4], monkeypatch)
    assert cli.main(["try", f"out={tmp_path}"]) == 4
    out = capsys.readouterr().out
    assert len(calls) == 3 and "run the same command again" in out and "== report" in out


def test_a_live_try_builds_the_index_the_viewer_reads(tmp_path, monkeypatch, capsys):
    built = []
    monkeypatch.setattr(tryout, "_build_index", lambda judged, model: built.append(judged))
    _dry_then([0], monkeypatch)
    assert cli.main(["try", f"out={tmp_path}"]) == 0
    assert built == [tmp_path / "judged"]
    # the index of a root with nothing judged cannot be built: one line, not an exit code
    monkeypatch.undo()
    _dry_then([0], monkeypatch)
    assert cli.main(["try", f"out={tmp_path / 'b'}"]) == 0
    assert "(not written)" not in capsys.readouterr().out
