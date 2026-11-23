"""The command line names its options by what they select: ``--view`` / ``--taxonomy`` take
``spans`` or ``claim_map``, and the claim-map commands are ``consistency``, ``pilot-dump-units``
and ``gold-units``. The earlier spellings still resolve to the same thing."""

import pytest

from hallucination_bench import cli


def _parse(*argv: str):
    return cli.build_parser().parse_args(list(argv))


def test_view_and_taxonomy_values():
    assert _parse("report").view == "spans"
    assert _parse("report", "--view", "claim_map").view == "claim_map"
    assert _parse("calibrate", "--split", "dev").taxonomy == "spans"
    assert _parse("calibrate", "--split", "dev", "--taxonomy", "claim_map").taxonomy == "claim_map"


@pytest.mark.parametrize(("old", "new"), sorted(cli.PIPELINE_ALIASES.items()))
def test_earlier_view_and_taxonomy_values_still_resolve(old, new):
    assert _parse("report", "--view", old).view == new
    assert _parse("report", f"--view={old}").view == new
    assert _parse("report", f"view={old}").view == new
    assert _parse("calibrate", "split=dev", f"taxonomy={old}").taxonomy == new
    assert _parse("calibrate", "--split", "dev", "--taxonomy", old).taxonomy == new


def test_an_unknown_value_is_still_refused(capsys):
    with pytest.raises(SystemExit) as e:
        _parse("report", "--view", "v3")
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'v3'" in err and "spans" in err and "claim_map" in err


@pytest.mark.parametrize(
    ("old", "new", "rest"),
    [
        ("consistency-v2", "consistency", ["--rescore"]),
        ("pilot-dump-v2", "pilot-dump-units", ["--arms", "a", "--dest", "x.jsonl"]),
        ("units-v2", "gold-units", ["--cases", "in.jsonl", "--dest", "out.jsonl"]),
    ],
)
def test_earlier_command_names_still_resolve(old, new, rest):
    assert cli.COMMAND_ALIASES[old] == new
    a, b = _parse(old, *rest), _parse(new, *rest)
    assert a.prog == b.prog == f"hbench {new}"
    assert type(a) is type(b) and a.to_dict() == b.to_dict()


def test_earlier_command_name_runs_the_command(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "_write_jsonl", lambda dest, rows: seen.append((dest, rows)))
    src = tmp_path / "in.jsonl"
    src.write_text("", encoding="utf-8")
    assert cli.main(["units-v2", "--cases", str(src), "--dest", str(tmp_path / "out.jsonl")]) == 0
    assert seen == [(tmp_path / "out.jsonl", [])]


def test_help_shows_the_descriptive_names_only(capsys):
    with pytest.raises(SystemExit):
        _parse("--help")
    top = capsys.readouterr().out
    for new in cli.COMMAND_ALIASES.values():
        assert new in top
    for old in cli.COMMAND_ALIASES:
        assert old not in top
    with pytest.raises(SystemExit):
        _parse("report", "--help")
    text = capsys.readouterr().out
    assert "view='spans'" in text and "claim_map" in text
    assert "v1" not in text and "v2" not in text
