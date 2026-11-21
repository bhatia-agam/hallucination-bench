"""``$HBENCH_EXTRA_READERS`` / ``readers.add``: extra reader names on every allow-list, none by
default (the lists a monkeypatch used to extend: index, both safety exports, benign_roleplay)."""

import json
from pathlib import Path
from typing import Any

import pytest

from hallucination_bench import index, readers
from hallucination_bench.evals.benign_roleplay import cli as br_cli
from hallucination_bench.evals.benign_roleplay import index_export as br_export
from hallucination_bench.evals.benign_roleplay import wsb
from hallucination_bench.evals.overrefusal import index_export as or_export

NEW = "my-new-lens"
BASE_V8 = frozenset({"s3d-rl600", "mlnla-rl600-all11", "jlens"})


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(readers.ENV, raising=False)
    readers.clear()
    yield
    readers.clear()


def test_default_is_unchanged() -> None:
    assert readers.extra() == ()
    for fam in index.SAFETY_READERS_V8:
        assert index.safety_readers(fam, "v8") == BASE_V8
    assert or_export.export_readers() == or_export.EXPORT_READERS
    assert wsb.known_arms() == wsb.KNOWN_ARMS and wsb.campaign_nested() == wsb.CAMPAIGN_NESTED
    assert br_export._readers() == index.SAFETY_READERS_V7_ADDED[index.BENIGN_ROLEPLAY]


def test_env_names_reach_every_allow_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(readers.ENV, f"{NEW}, other-lens,{NEW},,jlens")
    assert readers.extra() == (NEW, "other-lens", "jlens")
    for fam in index.SAFETY_READERS_V8:
        assert index.safety_readers(fam, "v8") == BASE_V8 | {NEW, "other-lens"}
    assert or_export.export_readers() == (*or_export.EXPORT_READERS, NEW, "other-lens")
    assert wsb.known_arms() == (*wsb.KNOWN_ARMS, NEW, "other-lens")
    assert NEW in wsb.campaign_nested() and NEW in br_export._readers()
    # the module constants themselves never move
    assert NEW not in index.SAFETY_READERS_V8[index.OVERREFUSAL]
    assert NEW not in or_export.EXPORT_READERS and NEW not in wsb.KNOWN_ARMS


def test_add_is_the_code_side_twin() -> None:
    readers.add(NEW, NEW, "")
    assert readers.extra() == (NEW,)
    assert NEW in index.safety_readers(index.OVERREFUSAL, "v8")
    readers.clear()
    assert readers.extra() == ()


def _row(arm: str, item: str) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(index.SAFETY_COLS, 0)
    row.update(arm=arm, layer=44, item=item, cluster=item, n_claims=3, n_false=1, n_contradicted=1)
    return row


def test_safety_rows_keep_an_extra_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(NEW, "i1"), _row("jlens", "i1")]
    (tmp_path / index.SAFETY_COUNTS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    got, ignored = index.safety_rows(tmp_path, index.OVERREFUSAL, "v8")
    assert set(got) == {"jlens"} and ignored == {NEW}  # silently dropped without the setting
    monkeypatch.setenv(readers.ENV, NEW)
    got, ignored = index.safety_rows(tmp_path, index.OVERREFUSAL, "v8")
    assert set(got) == {"jlens", NEW} and not ignored


def test_benign_roleplay_accepts_an_extra_arm(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(wsb.WsbError, match="unknown arms"):
        br_cli._arms(f"s3d-rl600,{NEW}")
    assert wsb.campaign_dir(tmp_path, "final-olens-rl600-x4", "all") == (
        tmp_path / "final-olens-rl600-x4" / "all"
    )
    monkeypatch.setenv(readers.ENV, NEW)
    assert br_cli._arms(f"s3d-rl600,{NEW}") == ["s3d-rl600", NEW]
    assert wsb.campaign_dir(tmp_path, NEW, "L44") == tmp_path / NEW / "L44"
