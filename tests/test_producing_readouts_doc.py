"""docs/producing_readouts.md states the size of every family's plan and names commands and
keys; these keep the page equal to the code."""

import re
from pathlib import Path

import pytest

from hallucination_bench import REPO_ROOT, cli, readplan

DOC = (REPO_ROOT / "docs" / "producing_readouts.md").read_text(encoding="utf-8")


def _n(x: int) -> str:
    return f"{x:,}"


@pytest.mark.parametrize("family", list(readplan.FAMILIES))
def test_the_family_table_is_the_plan(family):
    rows = readplan.plan(family)
    sites = sum(len(r["positions"]["positions"]) for r in rows)
    render = rows[0]["render"]
    cells = _n(readplan.n_cells(rows))
    assert f"| `{family}` | {_n(len(rows))} | `{render}` | {_n(sites)} | {cells} |" in DOC


def test_the_total_is_the_sum_of_the_plans():
    total = sum(readplan.n_cells(readplan.plan(f)) for f in readplan.FAMILIES)
    assert f"reads {_n(total)} cells over its five plans" in DOC


def test_the_row_example_is_a_real_row():
    row = readplan.plan("story")[0]
    assert f'"id": "{row["id"]}"' in DOC
    assert str(row["positions"]["positions"]) in DOC


def test_every_hbench_command_line_on_the_page_parses():
    block = re.compile(r"```bash\n(.*?)```", re.DOTALL)
    lines = []
    for body in block.findall(DOC):
        body = body.replace("\\\n", " ")
        variables = dict(re.findall(r'^([AB])="(.*?)"$', body, re.DOTALL | re.MULTILINE))
        for line in body.splitlines():
            line = line.split("  #")[0].strip()
            if line.startswith("uv run hbench "):
                for name, value in variables.items():
                    line = line.replace(f"${name}", value)
                lines.append(line.removeprefix("uv run hbench ").split())
    assert len(lines) >= 14
    for argv in lines:
        cli.command_for(argv)  # exits 2 on an unknown command, key or value


def test_the_files_the_page_names_exist():
    for rel in ("scripts/overrefusal/sj_run.py", "src/hallucination_bench/readplan.py"):
        assert (REPO_ROOT / rel).is_file()
    assert Path(REPO_ROOT / "docs" / "producing_readouts.md").is_file()
