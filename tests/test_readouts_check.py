"""``hbench readouts``: a readouts file against a family's read plan, before any judge call."""

import json

import pytest

from hallucination_bench import cli, readouts_check, readplan
from hallucination_bench.evals.agentic_misalignment import bank as am_bank

LAYERS = [44, 52]


def _rows(family, layers=LAYERS, *, samples=("a readout",), token=True):
    tokens = readouts_check.site_tokens(family)
    out = []
    for r in readplan.plan(family, layers):
        for layer in r["layers"]:
            for pos in r["positions"]["positions"]:
                row = {"id": r["id"], "layer": layer, "pos": pos, "samples": list(samples)}
                if token and (r["id"], pos) in tokens:
                    row["token"] = tokens[(r["id"], pos)]
                out.append(row)
    return out


def _write(path, rows, meta=None):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if meta is not None:
        path.with_name(path.name + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def _check(tmp_path, family, rows, meta=None, layers=LAYERS):
    return readouts_check.check(family, _write(tmp_path / "r.jsonl", rows, meta), layers)


@pytest.mark.parametrize("family", ["story", "overrefusal", "benign_roleplay", "main"])
def test_a_complete_file_passes(tmp_path, family):
    rows = _rows(family, [44])
    res = _check(tmp_path, family, rows, layers=[44])
    assert res.ok and res.summary["missing"] == 0
    assert res.summary["wanted"] == res.summary["cells"] == len(rows)
    assert res.summary["tokens_checked"] == len(rows)


def test_the_default_is_every_layer_of_the_index(tmp_path):
    res = _check(tmp_path, "story", _rows("story"), layers=None)
    assert not res.ok and res.summary["missing"] == 160 * 9
    assert "per layer {20: 160," in res.problems[0]
    full = _check(tmp_path, "story", _rows("story", readplan.LAYERS), layers=None)
    assert full.ok and full.summary["wanted"] == 1760


def test_a_missing_cell_fails_and_is_named(tmp_path):
    rows = _rows("story")
    gone = rows.pop(3)
    res = _check(tmp_path, "story", rows)
    assert not res.ok and res.summary["missing"] == 1
    assert f"{gone['id']} L{gone['layer']} p{gone['pos']}" in res.problems[0]


def test_rows_the_loader_drops_fail(tmp_path):
    rows = _rows("story")
    path = _write(tmp_path / "r.jsonl", rows)
    # a raw line separator inside a string ends the line for the loader: both halves malformed
    line = json.dumps({**rows[0], "samples": ["one two"]}, ensure_ascii=False)
    path.write_text(line + "\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    res = readouts_check.check("story", path, LAYERS)
    assert not res.ok and "'malformed': 2" in res.problems[0]
    dup = _check(tmp_path, "story", [*rows, rows[0]])
    assert not dup.ok and "'duplicate': 1" in dup.problems[0]
    mixed = _check(tmp_path, "story", [*rows[1:], {**rows[0], "samples": None}])
    assert not mixed.ok


def test_an_empty_file_fails(tmp_path):
    res = _check(tmp_path, "story", [])
    assert res.problems == ["the file has no rows"]


def test_rows_outside_the_plan_fail_only_where_the_judge_refuses_them(tmp_path):
    stray = {"id": "not-an-item", "layer": 44, "pos": 0, "samples": ["x"]}
    res = _check(tmp_path, "story", [*_rows("story"), stray])
    assert res.ok and any("unknown_id" in n for n in res.notes)
    am = _check(
        tmp_path, "agentic_misalignment", [*_rows("agentic_misalignment", [44]), stray], layers=[44]
    )
    assert not am.ok and "unknown_id" in am.problems[0]


def test_agentic_misalignment_complete_layer_passes_and_split_sites_are_not_token_checked(
    tmp_path,
):
    rows = _rows("agentic_misalignment", [44])
    assert len(rows) == am_bank.N_SITES
    res = _check(tmp_path, "agentic_misalignment", rows, layers=[44])
    assert res.ok
    assert (
        res.summary["tokens_checked"]
        == am_bank.N_SITES - am_bank.load_bank()["meta"]["n_split_char_sites"]
    )


def test_a_wrong_token_fails(tmp_path):
    rows = _rows("overrefusal", [44])
    rows[5]["token"] = rows[5]["token"] + "?"
    res = _check(tmp_path, "overrefusal", rows, layers=[44])
    assert not res.ok and "1 of 11325 row tokens" in res.problems[0]


def test_rows_without_tokens_pass_with_a_note(tmp_path):
    res = _check(tmp_path, "story", _rows("story", token=False))
    assert res.ok and any("not cross-checked" in n for n in res.notes)


def test_sidecar(tmp_path):
    rows = _rows("story")
    sha = readplan.plan_sha(readplan.plan("story"))
    good = {"reads": "full", "sampling": "prose1", "layers": LAYERS, "plan_sha": sha}
    assert _check(tmp_path, "story", rows, good).ok
    other = _check(tmp_path, "story", rows, {**good, "plan_sha": "0" * 64})
    assert not other.ok and "not this family's plan" in other.problems[0]
    short = _check(tmp_path, "story", rows, {**good, "layers": [44]})
    assert not short.ok and "without [52]" in short.problems[0]
    four = {**good, "sampling": "bullet4"}
    bad = _check(tmp_path, "story", rows, four)
    assert not bad.ok and "bullet4 (4 per cell)" in bad.problems[0]
    assert _check(tmp_path, "story", _rows("story", samples=("a", "b", "c", "d")), four).ok
    tok = _check(tmp_path, "story", rows, {**good, "sampling": "tokens10"})
    assert not tok.ok and "the rows hold text" in tok.problems[0]
    bare = _check(tmp_path, "story", rows, {"reads": "full"})
    assert bare.ok and len(bare.notes) == 2


def test_roleplay_sidecar_plan_is_the_pinned_workspacebench_plan(tmp_path):
    rows = _rows("benign_roleplay", [44])
    sha = "49a70f2877c0b5d35ea952c617b07a152af184878545506522cf4868d33f9dd6"
    meta = {"reads": "full", "sampling": "prose1", "layers": [44], "plan_sha": sha}
    assert _check(tmp_path, "benign_roleplay", rows, meta, layers=[44]).ok


def test_no_sidecar_is_a_note(tmp_path):
    res = _check(tmp_path, "benign_roleplay", _rows("benign_roleplay", [44]), layers=[44])
    assert res.ok and any("without enforcing" in n for n in res.notes)


def test_command(tmp_path, capsys):
    path = _write(tmp_path / "r.jsonl", _rows("story"))
    assert cli.main(["readouts", str(path), "family=story", "layers=44,52"]) == 0
    out = capsys.readouterr().out
    assert out.endswith("PASS\n") and "cells the judge reads: 320, missing 0" in out
    assert cli.main(["readouts", f"readouts={path}", "family=story"]) == 2
    assert "FAIL (1)" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="readouts file not found"):
        cli.main(["readouts", str(tmp_path / "none.jsonl"), "family=story"])
    for bad in (["readouts", str(path)], ["readouts", str(path), "family=nope"]):
        with pytest.raises(SystemExit) as e:
            cli.command_for(bad)
        assert e.value.code == 2


def test_the_check_reads_the_file_with_the_judges_loader(tmp_path, monkeypatch):
    seen = {}
    real = readouts_check.load_readouts

    def spy(path, **kw):
        seen.update(kw)
        return real(path, **kw)

    monkeypatch.setattr(readouts_check, "load_readouts", spy)
    assert _check(tmp_path, "story", _rows("story")).ok
    assert set(seen) == {"positions"}
