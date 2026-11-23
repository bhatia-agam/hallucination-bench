"""Where a judged run's outputs live and how they are read: the per-model root, its arm
directories, an arm's ``cells.jsonl`` and ``arm.json``, and the prompt / response context of one
cell. Shared by the span report and the claim-map report."""

import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

from hallucination_bench.evals.wsbench_chat import NAME, bank
from hallucination_bench.evals.wsbench_chat.run import model_slug


def model_root(out_root: Path, judge_model: str, eval_name: str = NAME) -> Path:
    return Path(out_root) / eval_name / model_slug(judge_model)


def arm_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/arm.json"))


def iter_cells(d: Path) -> Iterator[dict[str, Any]]:
    p = d / "cells.jsonl"
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _context(rec: dict[str, Any], width: int = 400, bank_mod: ModuleType = bank) -> tuple[str, str]:
    item = next(it for it in bank_mod.load_items() if it["id"] == rec["id"])
    site = next(s for s in item["sites"] if int(s["pos"]) == rec["pos"])
    mr = bank_mod.marked_response(item, site)
    i = mr.index("⟦READ⟧")
    return item["prompt"], mr[max(0, i - width) : i + width]


def load_arm_json(d: Path) -> dict[str, Any]:
    return json.loads((d / "arm.json").read_text())
