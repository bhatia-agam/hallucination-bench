"""The read plan of the index families: for every item, the prompt to feed the subject model,
the token positions to read and the layers to read them at. ``hbench plan`` writes it, one
JSON line per item, in the row format of WorkspaceBench's ``wsbench plan``, so a producer
that reads WorkspaceBench plans reads these too.

Everything comes from the banks under ``data/``; nothing is tokenized here. A family whose
bank ships the captured token ids is planned as ``render: "captured"`` (``extra.input_ids``);
the others as the exact text the model read (``plain``) or its chat messages
(``chat_context``), with ``extra.n_tokens`` so a producer can check its own tokenization.
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hallucination_bench.command import Command, optional, text
from hallucination_bench.evals._bank_common import positions as site_positions
from hallucination_bench.evals.benign_roleplay.wsb import GRID

LAYERS: tuple[int, ...] = tuple(GRID)  # the layers the index has a row for
SAME = {"kind": "same"}
# WorkspaceBench hashes a plan without these keys (its ``reads.PLAN_SHA_EXCLUDE``)
SHA_EXCLUDE = ("lite_positions", "lite_layers", "reads", "note", "layers")

Row = dict[str, Any]


def _row(
    family: str,
    id_: str,
    render: str,
    positions: Sequence[int],
    layers: Sequence[int],
    *,
    text: str | None = None,
    messages: list[dict[str, str]] | None = None,
    extra: Mapping[str, Any],
    lite: Mapping[str, Any] = SAME,
) -> Row:
    """One plan row, keys in the order ``wsbench plan`` writes them."""
    row: Row = {
        "family": family,
        "id": id_,
        "render": render,
        "positions": {"kind": "positions", "positions": [int(p) for p in positions]},
        "layers": [int(x) for x in layers],
    }
    if text is not None:
        row["text"] = text
    if messages is not None:
        row["messages"] = messages
    row["extra"] = dict(extra)
    row["lite_positions"] = dict(lite)
    return row


def _load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _captured(family: str, bank: Any, capture: Path, layers: Sequence[int]) -> list[Row]:
    """A chat-style bank whose capture file ships the token ids the model read. The capture
    file's read positions must be the bank's sites, item by item."""
    want = site_positions(bank.load_items())
    rows = []
    for r in _load(capture):
        id_ = r.get("id") or r["label"]
        pos = [int(p) for p in r["eval_positions"]]
        if want.get(id_) != pos:
            raise ValueError(f"{capture}: {id_}: read positions are not the bank's sites")
        ids = [int(x) for x in r["input_ids"]]
        if not all(0 <= p < len(ids) for p in pos):
            raise ValueError(f"{capture}: {id_}: a read position is outside the prompt")
        rows.append(_row(family, id_, "captured", pos, layers, extra={"input_ids": ids}))
    if {r["id"] for r in rows} != set(want) or len(rows) != len(want):
        raise ValueError(f"{capture}: its items are not the bank's items")
    return rows


def _main(layers: Sequence[int]) -> list[Row]:
    from hallucination_bench.evals.wsbench_chat import bank

    saved = bank.current_bank()
    bank.use_bank("x3", explicit=False)  # the bank the index's main family runs on
    try:
        return _captured("main", bank, bank.X3_BANK_DIR / "capture_rows.json", layers)
    finally:
        bank.use_bank(saved, explicit=False)


def _story(layers: Sequence[int]) -> list[Row]:
    from hallucination_bench.evals.story import bank

    return _captured("story", bank, bank.BANK_DIR / "capture_rows.json", layers)


def _every_position(family: str, items: Iterable[Mapping[str, Any]], layers: Sequence[int]) -> list:
    """A bank read at every token of ``prompt_render`` (the text the model read, chat markup
    included): one site per token, so the site count is the prompt's length in tokens."""
    rows = []
    for it in items:
        pos = [int(s["pos"]) for s in it["sites"]]
        if pos != list(range(len(pos))):
            raise ValueError(f"{family}: {it['id']}: sites are not every position in order")
        extra = {"n_tokens": len(pos)}
        rows.append(
            _row(family, it["id"], "plain", pos, layers, text=it["prompt_render"], extra=extra)
        )
    return rows


def _agentic_misalignment(layers: Sequence[int]) -> list[Row]:
    from hallucination_bench.evals.agentic_misalignment import bank

    return _every_position("agentic_misalignment", bank.load_items(), layers)


def _overrefusal(layers: Sequence[int]) -> list[Row]:
    from hallucination_bench.evals.overrefusal import data

    return _every_position("overrefusal", data.load_items(), layers)


def _benign_roleplay(layers: Sequence[int]) -> list[Row]:
    """The bench's own role-play items, row for row what the pinned WorkspaceBench plans for
    them (its ``jailbreak_recognition`` family over ``wsb_bank_hb_only_v1.json``), so the plan's
    hash is the one that judge expects beside the readouts. The items shared with
    WorkspaceBench are read with WorkspaceBench's own plan of that family."""
    from hallucination_bench.evals.benign_roleplay import bank

    rows = []
    for it in _load(bank.HB_ONLY_BANK)["items"]:
        messages = it["messages"]
        last = max(i for i, m in enumerate(messages) if m["role"] == "user")
        first, end = (int(x) for x in it["read"]["span"])
        extra = {"n_tokens": it["read"]["n_tokens"]}
        if it.get("kind") == "control":
            extra["ati_scored"] = False
        rows.append(
            _row(
                bank.FAMILY,
                it["id"],
                "chat_context",
                range(first, end + 1),
                layers,
                messages=messages[: last + 1],
                extra=extra,
                lite={"kind": "final_token"},
            )
        )
    return rows


@dataclass(frozen=True)
class Family:
    """One index family's plan: how its rows are built and where its judge reads the
    readouts a producer writes for them."""

    rows: Callable[[Sequence[int]], list[Row]]
    readouts: str


FAMILIES: dict[str, Family] = {
    "main": Family(_main, "any path: hbench run bank=x3 arm=<arm>=<file>"),
    "story": Family(_story, "any path: hbench run eval=story arm=<arm>=<file>"),
    "agentic_misalignment": Family(
        _agentic_misalignment, "<readouts_root>/<arm>/agentic_misalignment.jsonl"
    ),
    "overrefusal": Family(_overrefusal, "$HBENCH_OR_READOUTS_V3/<arm>/overrefusal.jsonl"),
    "benign_roleplay": Family(
        _benign_roleplay, "<readouts_root>/<arm>/hb_only/jailbreak_recognition.jsonl"
    ),
}


def plan(family: str, layers: Sequence[int] | None = None) -> list[Row]:
    """The plan rows of one index family, in bank order, at ``layers`` (default: every layer
    the index has a row for)."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; known: {', '.join(FAMILIES)}")
    return FAMILIES[family].rows(list(layers) if layers else LAYERS)


def plan_sha(rows: Iterable[Mapping[str, Any]]) -> str:
    """The hash WorkspaceBench gives a plan (its ``reads.plan_sha_rows``) and its producer
    writes beside the readouts: sha256 over the rows with their keys sorted, without the
    layers, the lite site and the note. Two plans with the same prompts and positions have
    the same hash."""
    h = hashlib.sha256()
    for row in rows:
        kept = {k: v for k, v in row.items() if k not in SHA_EXCLUDE}
        h.update(json.dumps(kept, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def n_cells(rows: Iterable[Mapping[str, Any]]) -> int:
    return sum(len(r["positions"]["positions"]) * len(r["layers"]) for r in rows)


def write(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    """One JSON object per line, ASCII only, so no character inside a prompt can end a line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def csv_ints(v: object) -> list[int]:
    try:
        return [int(x) for x in str(v).split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"expected a comma list of layer numbers, got {v!r}") from None


def family_names(v: object) -> list[str]:
    names = [x.strip() for x in str(v).split(",") if x.strip()]
    if unknown := [n for n in names if n not in FAMILIES]:
        raise ValueError(f"unknown family {unknown[0]!r} (choose from {', '.join(FAMILIES)})")
    return names


class Plan(Command):
    about = (
        "write the read plan of the index families: one JSON row per item (prompt, token "
        "positions, layers) in the row format of `wsbench plan`, for any producer of readouts "
        "(docs/producing_readouts.md)"
    )

    def __init__(self) -> None:
        super().__init__()
        self.key("out", "outputs/plan", text, "the folder that gets <family>.jsonl")
        self.key(
            "families",
            ",".join(FAMILIES),
            family_names,
            f"comma list of {', '.join(FAMILIES)} (default: all)",
        )
        self.key(
            "layers",
            None,
            optional(csv_ints),
            f"comma list of layers to plan (default: the index's {len(LAYERS)} layers, "
            f"{LAYERS[0]} to {LAYERS[-1]} in steps of {LAYERS[1] - LAYERS[0]})",
        )

    def execute(self) -> int:
        out = Path(self.out)
        for name in self.families:
            rows = plan(name, self.layers)
            path = out / f"{name}.jsonl"
            write(rows, path)
            sites = sum(len(r["positions"]["positions"]) for r in rows)
            print(
                f"{name}: {len(rows)} items, {sites} positions, {n_cells(rows)} cells at "
                f"{len(rows[0]['layers'])} layers -> {path}\n"
                f"  plan_sha {plan_sha(rows)}\n"
                f"  readouts go to {FAMILIES[name].readouts}"
            )
        if "benign_roleplay" in self.families:
            print(
                "benign_roleplay: these are the bench's own items; the items shared with "
                "WorkspaceBench are read with its plan of jailbreak_recognition (`wsbench plan` "
                "in the pinned checkout)"
            )
        return 0


COMMANDS = {"plan": Plan}
