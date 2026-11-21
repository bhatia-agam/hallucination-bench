"""The command line is pydra-config: every subcommand is a ``command.Command``. These tests pin
it to the argparse command line it replaced, through ``fixtures/cli_argparse_snapshot.json``
(recorded from the argparse parser on its last commit): the same defaults, and the same values
for old-spelling command lines, the live drivers' among them."""

import json
from pathlib import Path

import pydra
import pytest

from hallucination_bench import cli, command, paths
from hallucination_bench.evals.benign_roleplay import bank as br_bank

SNAPSHOT = json.loads(
    (Path(__file__).parent / "fixtures" / "cli_argparse_snapshot.json").read_text(encoding="utf-8")
)
TOKENS = {
    "<paths.READOUTS_ROOT>": str(paths.READOUTS_ROOT),
    "<paths.X3_READOUTS>": str(paths.X3_READOUTS),
    "<paths.BR_CAMPAIGN>": str(paths.BR_CAMPAIGN),
    "<paths.BR_READOUTS>": str(paths.BR_READOUTS),
    "<paths.WSBENCH_PUB>": str(paths.WSBENCH_PUB),
    "<cli.INDEX_METHODS>": str(cli.INDEX_METHODS),
    "<benign_roleplay.bank.DATA_DIR>": str(br_bank.DATA_DIR),
    "<required>": pydra.REQUIRED,
}


def _expected(values: dict) -> dict:
    return {k: TOKENS.get(v, v) if isinstance(v, str) else v for k, v in values.items()}


def _leaves(table: dict | None = None, path: tuple[str, ...] = ()):
    for name, entry in (cli.commands() if table is None else table).items():
        if isinstance(entry, tuple):
            yield from _leaves(entry[1], (*path, name))
        else:
            yield " ".join((*path, name)), entry


def _values(argv: list[str]) -> dict:
    return cli.command_for(argv)[0].to_dict()


def _new_spelling(argv: list[str], n_command: int) -> list[str]:
    """An old-spelling command line in the ``key=value`` / ``--key`` form (a repeated boolean
    becomes one comma list)."""
    kinds = cli.command_for(argv)[0]._kinds
    out: dict[str, str] = {}
    bare = []
    rest = argv[n_command:]
    i = 0
    while i < len(rest):
        arg = rest[i]
        i += 1
        if not arg.startswith("-"):
            bare.append(arg)
            continue
        name, eq, value = arg.lstrip("-").partition("=")
        key = name.replace("-", "_")
        if kinds[key] is command.boolean:
            out[key] = f"--{key}"
            continue
        if not eq:
            value = rest[i]
            i += 1
        out[key] = f"{out[key]},{value}" if key in out else f"{key}={value}"
    return [*argv[:n_command], *bare, *out.values()]


def test_every_subcommand_is_a_pydra_config():
    leaves = dict(_leaves())
    assert set(leaves) == set(SNAPSHOT["defaults"])
    for kind in leaves.values():
        assert issubclass(kind, pydra.Config) and issubclass(kind, command.Command)


@pytest.mark.parametrize("name", sorted(SNAPSHOT["defaults"]))
def test_defaults_equal_the_argparse_defaults(name):
    cmd = dict(_leaves())[name]()
    got = {k: getattr(cmd, k) for k in cmd._kinds}
    assert got == _expected(SNAPSHOT["defaults"][name])
    for key in got:  # --help has a meaning for every key
        assert cmd._meanings[key].strip()


@pytest.mark.parametrize("case", SNAPSHOT["cases"], ids=lambda c: " ".join(c["argv"])[:70])
def test_old_spelling_gives_the_argparse_values(case):
    assert _values(case["argv"]) == _expected(case["values"])


@pytest.mark.parametrize("case", SNAPSHOT["cases"], ids=lambda c: " ".join(c["argv"])[:70])
def test_key_value_spelling_gives_the_same_values(case):
    argv = case["argv"]
    if any(a.startswith("b=/x/b,c") for a in argv):  # a comma inside one repeated value:
        pytest.skip("only the old spelling can carry it")  # key=a,b always means two values
    n = 2 if isinstance(cli.commands()[argv[0]], tuple) else 1
    new = _new_spelling(argv, n)
    assert _values(new) == _expected(case["values"])


def test_translation_forms():
    run = lambda *a: _values(["run", *a])  # noqa: E731
    base = run("judge_model=m", "limit=3", "--rebuild")
    assert (base["judge_model"], base["limit"], base["rebuild"]) == ("m", 3, True)
    assert run("--judge-model", "m", "--limit", "3", "--rebuild") == base
    assert run("--judge-model=m", "--limit=3", "--rebuild") == base
    assert run("--judge_model", "m", "limit=3", "rebuild=True") == base
    assert run("--judge-mod", "m", "--lim", "3", "--reb") == base  # a unique prefix, as before
    # a value is the text that was typed: no literal evaluation
    typed = run("out=1e3", "arms=T", "layers=42,44", "judge_model=(1+1)", "items=007")
    assert typed["out"] == "1e3" and typed["arms"] == "T" and typed["layers"] == "42,44"
    assert typed["judge_model"] == "(1+1)" and typed["items"] == "007"
    assert run("limit=None")["limit"] is None and run("--arms", "None")["arms"] == "None"
    # a repeatable key: repeated, or one comma list; the old spelling keeps a value whole
    assert run("arm=a=1", "arm=b=2")["arm"] == ["a=1", "b=2"] == run("arm=a=1,b=2")["arm"]
    assert run("--arm", "a=1,x", "--arm=b=2")["arm"] == ["a=1,x", "b=2"]
    assert run("--rebuild", "rebuild=False")["rebuild"] is False  # the last one wins
    assert _values(["try", "-n", "5"])["n"] == 5 == _values(["try", "n=5"])["n"]
    assert _values(["check", "all"])["name"] == "all" == _values(["check", "name=all"])["name"]
    assert _values(["check", "--family", "story", "dry_run"])["name"] == "dry_run"
    assert _values(["overrefusal", "grow", "pairs"])["step"] == "pairs"


def test_show_prints_the_resolved_config_and_runs_nothing(capsys, monkeypatch):
    monkeypatch.setattr(cli, "cmd_run", lambda args: pytest.fail("--show must not run"))
    assert cli.main(["run", "--judge-model", "m", "limit=3", "--rebuild", "--show"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown == _values(["run", "judge_model=m", "limit=3", "--rebuild"])
    assert shown["judge_model"] == "m" and shown["limit"] == 3 and shown["rebuild"] is True
    assert cli.main(["overrefusal", "offtopic-export", "--joint", "--show"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "out": "outputs",
        "judge_model": "claude-opus-5-5",
        "joint": True,
    }


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "no_such_key=1"], "hbench run: error: unknown key 'no_such_key'"),
        (["run", "--no-such-flag", "1"], "hbench run: error: unknown key 'no_such_flag'"),
        (["run", "--no-such-flag"], "hbench run: error: unknown key 'no_such_flag'"),
        (["overrefusal", "run", "judge=1"], "hbench overrefusal run: error: unknown key 'judge'"),
        (["run", "stray"], "hbench run: error: unrecognized arguments: stray"),
        (["run", "--s", "x"], "ambiguous option: --s could match"),
        (["run", "limit=x"], "argument limit: invalid int value: 'x'"),
        (["run", "rpm=fast"], "argument rpm: invalid float value: 'fast'"),
        (["run", "rebuild=maybe"], "argument rebuild: expected True or False"),
        (["run", "eval=nope"], "argument eval: invalid choice: 'nope' (choose from "),
        (["run", "--limit"], "argument --limit: expected one argument"),
        (["calibrate"], "Missing required config value: split"),
        (["nope"], "hbench: error: unknown command 'nope'"),
        ([], "hbench: error: a command is required"),
        (["overrefusal", "nope"], "hbench overrefusal: error: unknown command 'nope'"),
        (["run", ".execute"], "unrecognized arguments: .execute"),
    ],
)
def test_usage_errors_exit_2(argv, message, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == command.EXIT_USAGE == 2
    err = capsys.readouterr().err
    assert message in err and err.startswith("usage: hbench")


def test_help_lists_every_key_with_default_and_meaning(capsys):
    for name, kind in _leaves():
        with pytest.raises(SystemExit) as e:
            cli.main([*name.split(), "--help"])
        assert e.value.code == 0
        out = capsys.readouterr().out
        cmd = kind()
        assert out.startswith(f"usage: hbench {name}")
        assert "The old spelling still works" in out
        for key in cmd._kinds:
            default = getattr(cmd, key)
            shown = "<required>" if default is pydra.REQUIRED else repr(default)
            assert f"\n  {key}={shown}\n" in out
    for argv in (["--help"], ["-h"], ["overrefusal", "--help"]):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 0
    assert "offtopic-export" in capsys.readouterr().out


def test_one_command_runs_another_through_pydra():
    run = cli.Run.of(out="o", arm=["try=/x/a,b.jsonl"], allow_missing=True, show_prompts=1)
    assert run.arm == ["try=/x/a,b.jsonl"] and run.allow_missing is True and run.show_prompts == 1
    assert run.to_dict() == _values(
        ["run", "--out", "o", "--arm", "try=/x/a,b.jsonl", "--allow-missing", "--show-prompts", "1"]
    )
    with pytest.raises(command.UsageError, match="unknown key 'nope'"):
        cli.Run.of(nope=1)
