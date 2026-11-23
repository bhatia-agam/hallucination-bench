"""``hbench readouts``: does a readouts file hold every cell a family's judge reads? Run it
before any judge call. The file is loaded with the loader the judges use
(``readouts.load_readouts``), so a row it drops here is a row they would drop.

A problem fails the check (exit 2); a note is printed and does not. Problems: no rows, rows
the loader drops, a missing cell, a row token that is not the bank's token at that position,
and a sidecar (``<file>.meta.json``) that names another plan, lacks a layer or declares a
sampling the rows do not have.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pydra

from hallucination_bench import readplan
from hallucination_bench.command import Command, choice, optional, text
from hallucination_bench.readouts import load_readouts

EXIT_PROBLEM = 2  # as `hbench run` exits on missing cells
SIDECAR_SUFFIX = ".meta.json"
SAMPLES_PER_CELL = {"prose1": 1, "bullet4": 4}  # the sidecar's `sampling` for sampled text
TOKEN_SAMPLING = "tokens10"
# rows the loader drops for a reason other than "not in this family's plan"
BROKEN = ("malformed", "duplicate", "mixed_kind")
OUTSIDE = ("unknown_id", "pos_not_selected")
# the judge of this family refuses a file with rows outside its plan; the others ignore them
REFUSES_OUTSIDE = ("agentic_misalignment",)
SHOWN = 5


@dataclass
class Result:
    family: str
    path: Path
    summary: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def site_tokens(family: str) -> dict[tuple[str, int], str]:
    """The bank's token string at each read position, where the bank records one that a
    producer can reproduce (``tokenizer.decode([id])``)."""
    if family == "benign_roleplay":
        from hallucination_bench.evals.benign_roleplay import bank

        items = json.loads(bank.HB_ONLY_BANK.read_text(encoding="utf-8"))["items"]
        return {(it["id"], int(p)): t for it in items for p, t in it["read"]["tokens"].items()}
    if family == "overrefusal":
        from hallucination_bench.evals.overrefusal import data

        items = data.load_items()
    elif family == "agentic_misalignment":
        from hallucination_bench.evals.agentic_misalignment import bank as am

        items = am.load_items()
    elif family == "story":
        from hallucination_bench.evals.story import bank as story

        items = story.load_items()
    else:
        from hallucination_bench.evals.wsbench_chat import bank as chat

        items = json.loads(chat.X3_ITEMS_PATH.read_text(encoding="utf-8"))["items"]
    return {
        (it["id"], int(s["pos"])): s["token"]
        for it in items
        for s in it["sites"]
        # a site whose token the bank could not place whole in the text has no usable string
        if "token" in s and "split_char" not in s
    }


def _some(cells: Sequence[tuple[str, int, int]]) -> str:
    shown = ", ".join(f"{i} L{la} p{p}" for i, la, p in cells[:SHOWN])
    return shown + (f", and {len(cells) - SHOWN} more" if len(cells) > SHOWN else "")


def _sidecar(res: Result, meta: Mapping[str, Any], sha: str, layers, kind, counts) -> None:
    have = meta.get("plan_sha")
    if have and have != sha:
        res.problems.append(
            f"the sidecar's plan_sha {have} is not this family's plan ({sha}): the readouts "
            "were produced for other prompts or positions"
        )
    elif not have:
        res.notes.append("the sidecar has no plan_sha")
    declared = meta.get("layers")
    if isinstance(declared, list) and (gone := sorted(set(layers) - set(declared))):
        res.problems.append(f"the sidecar declares layers {declared}, without {gone}")
    sampling = meta.get("sampling")
    if sampling == TOKEN_SAMPLING and kind != "tokens":
        res.problems.append(f"the sidecar declares sampling {sampling}, the rows hold text")
    elif sampling in SAMPLES_PER_CELL:
        n = SAMPLES_PER_CELL[sampling]
        if kind != "prose" or set(counts) != {n}:
            res.problems.append(
                f"the sidecar declares sampling {sampling} ({n} per cell); the rows have "
                f"{dict(sorted(counts.items())) if kind == 'prose' else 'token lists'}"
            )
    elif sampling is None:
        res.notes.append("the sidecar declares no sampling")
    else:
        res.notes.append(f"the sidecar's sampling {sampling!r} is not one this check knows")


def check(family: str, path: Path, layers: Sequence[int] | None = None) -> Result:
    """Check ``path`` against ``family``'s read plan at ``layers`` (default: every layer the
    index has a row for)."""
    path = Path(path)
    res = Result(family, path)
    want_layers = sorted(set(layers)) if layers else list(readplan.LAYERS)
    rows = readplan.plan(family, want_layers)
    positions = {r["id"]: r["positions"]["positions"] for r in rows}
    cells, rep = load_readouts(path, positions=positions)
    res.summary = {
        "rows": rep.n_rows,
        "kind": rep.kind,
        "layers": rep.layers,
        "cells": len(cells),
        "empty": rep.n_empty,
        "skipped": {k: v for k, v in rep.skipped.items() if v},
    }
    if rep.n_rows == 0:
        res.problems.append("the file has no rows")
        return res
    if broken := {k: rep.skipped[k] for k in BROKEN if rep.skipped[k]}:
        res.problems.append(
            f"rows the loader drops: {broken} (malformed: not the row contract, or a raw "
            "U+0085 / U+2028 / U+2029 inside a string, which ends a line for the loader: "
            "write them escaped)"
        )
    if outside := {k: rep.skipped[k] for k in OUTSIDE if rep.skipped[k]}:
        line = f"rows outside the family's plan: {outside}"
        (res.problems if family in REFUSES_OUTSIDE else res.notes).append(line)

    have = {(c.id, c.layer, c.pos) for c in cells}
    wanted = [(i, la, p) for la in want_layers for i, ps in positions.items() for p in ps]
    missing = [c for c in wanted if c not in have]
    res.summary["wanted"] = len(wanted)
    res.summary["missing"] = len(missing)
    if missing:
        per_layer = dict(sorted(Counter(la for _, la, _ in missing).items()))
        res.problems.append(
            f"{len(missing)} of {len(wanted)} cells missing (per layer {per_layer}): "
            f"{_some(missing)}"
        )
    if extra := sorted(set(rep.layers) - set(want_layers)):
        res.notes.append(f"layers not asked for: {extra}")
    if rep.n_empty:
        res.notes.append(f"{rep.n_empty} cells with no text or tokens (never judged)")

    tokens = site_tokens(family)
    with_token = [c for c in cells if c.token is not None and (c.id, c.pos) in tokens]
    wrong = [c for c in with_token if c.token != tokens[(c.id, c.pos)]]
    res.summary["tokens_checked"] = len(with_token)
    if wrong:
        c = wrong[0]
        res.problems.append(
            f"{len(wrong)} of {len(with_token)} row tokens are not the bank's token at that "
            f"position, e.g. {c.id} p{c.pos}: {c.token!r}, bank {tokens[(c.id, c.pos)]!r} "
            "(another tokenization or render reads other positions; `token` is "
            "tokenizer.decode([id]))"
        )
    elif not with_token:
        res.notes.append("no row carries a `token`: the read positions were not cross-checked")

    counts = Counter(len(c.samples) for c in cells if c.samples is not None)
    if rep.kind == "prose":
        res.summary["samples_per_cell"] = dict(sorted(counts.items()))
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        _sidecar(res, meta, readplan.plan_sha(rows), want_layers, rep.kind, counts)
    elif family == "benign_roleplay":
        res.notes.append(
            f"no sidecar {sidecar.name}: WorkspaceBench's judge then reports its own checks "
            "without enforcing them"
        )
    else:
        res.notes.append(f"no sidecar {sidecar.name} (optional for this family)")
    return res


def render(res: Result) -> str:
    s = res.summary
    lines = [
        f"{res.family}: {res.path}",
        f"  {s.get('rows', 0)} rows, {s.get('cells', 0)} kept ({s.get('kind') or 'no kind'}), "
        f"layers {s.get('layers', [])}",
    ]
    if "wanted" in s:
        lines.append(f"  cells the judge reads: {s['wanted']}, missing {s['missing']}")
    if "samples_per_cell" in s:
        lines.append(f"  samples per cell: {s['samples_per_cell']}")
    lines += [f"  note: {n}" for n in res.notes]
    lines += [f"  PROBLEM: {p}" for p in res.problems]
    lines.append("PASS" if res.ok else f"FAIL ({len(res.problems)})")
    return "\n".join(lines)


class Readouts(Command):
    about = (
        "check, before any judge call, that a readouts file holds every cell a family's judge "
        "reads: rows in the contract, every (item, layer, position) of the read plan, the "
        "bank's token at each position, and what its sidecar declares (exit 2 on a problem)"
    )
    positional = "readouts"

    def __init__(self) -> None:
        super().__init__()
        self.key("readouts", pydra.REQUIRED, text, "the readouts file (JSONL)")
        self.key(
            "family",
            pydra.REQUIRED,
            choice(*readplan.FAMILIES),
            "the index family the file was produced for",
        )
        self.key(
            "layers",
            None,
            optional(readplan.csv_ints),
            "comma list of layers that must be complete (default: the index's "
            f"{len(readplan.LAYERS)} layers)",
        )

    def execute(self) -> int:
        path = Path(self.readouts)
        if not path.is_file():
            raise SystemExit(f"readouts file not found: {path}")
        res = check(self.family, path, self.layers)
        print(render(res))
        return 0 if res.ok else EXIT_PROBLEM


COMMANDS = {"readouts": Readouts}
