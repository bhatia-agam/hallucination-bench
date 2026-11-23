"""``hbench plan``: the index families' read plan, from the banks, in WorkspaceBench's plan
row format. Offline: nothing is tokenized and nothing is called."""

import json

import pytest

from hallucination_bench import cli, readplan
from hallucination_bench.evals.agentic_misalignment import bank as am_bank
from hallucination_bench.evals.benign_roleplay import bank as br_bank
from hallucination_bench.evals.benign_roleplay import wsb
from hallucination_bench.evals.overrefusal import data as or_data
from hallucination_bench.evals.story import bank as story_bank
from hallucination_bench.evals.wsbench_chat import bank as chat_bank

# what the pinned WorkspaceBench (wsb.WSB_SHA) hashes its own plan of the bench's role-play
# items to, with the bench's bank as its jailbreak_recognition bank: the value its judge wants
# in the readouts' sidecar
ROLEPLAY_PLAN_SHA = "49a70f2877c0b5d35ea952c617b07a152af184878545506522cf4868d33f9dd6"
ROW_KEYS = {"family", "id", "render", "positions", "layers", "extra", "lite_positions"}
SIZES = {  # family -> (items, positions per layer)
    "main": (446, 5_924),
    "story": (20, 160),
    "agentic_misalignment": (32, 50_452),
    "overrefusal": (427, 11_325),
    "benign_roleplay": (64, 47_179),
}


def _positions(rows):
    return {r["id"]: r["positions"]["positions"] for r in rows}


def test_layers_are_the_grid_of_the_index():
    assert list(readplan.LAYERS) == wsb.GRID == list(range(20, 61, 4))


@pytest.mark.parametrize("family", sorted(SIZES))
def test_every_family_has_its_whole_bank(family):
    rows = readplan.plan(family)
    items, sites = SIZES[family]
    assert len(rows) == items == len({r["id"] for r in rows})
    assert sum(len(p) for p in _positions(rows).values()) == sites
    assert readplan.n_cells(rows) == sites * len(readplan.LAYERS)
    for r in rows:
        assert ROW_KEYS <= set(r) <= ROW_KEYS | {"text", "messages"}
        assert r["positions"]["kind"] == "positions"
        assert r["layers"] == list(readplan.LAYERS)
        pos = r["positions"]["positions"]
        assert pos == sorted(set(pos)) and pos[0] >= 0


def test_main_is_the_bank_the_index_runs_on_and_leaves_the_selected_bank_alone():
    before = (chat_bank.ITEMS_PATH, chat_bank.BANK_SOURCE, chat_bank.SELECTED)
    rows = readplan.plan("main")
    assert (chat_bank.ITEMS_PATH, chat_bank.BANK_SOURCE, chat_bank.SELECTED) == before
    items = json.loads(chat_bank.X3_ITEMS_PATH.read_text(encoding="utf-8"))["items"]
    assert _positions(rows) == chat_bank.positions(items)
    for r in rows:
        assert r["family"] == "main" and r["render"] == "captured"
        assert max(r["positions"]["positions"]) < len(r["extra"]["input_ids"])


def test_story_positions_are_the_bank_sites():
    rows = readplan.plan("story")
    assert _positions(rows) == story_bank.positions(story_bank.load_items())
    assert {r["render"] for r in rows} == {"captured"}


@pytest.mark.parametrize(
    "family,items",
    [("agentic_misalignment", am_bank.load_items), ("overrefusal", or_data.load_items)],
)
def test_prompt_families_read_every_token_of_the_text_the_model_read(family, items):
    by_id = {it["id"]: it for it in items()}
    rows = readplan.plan(family)
    assert set(by_id) == {r["id"] for r in rows}
    for r in rows:
        it = by_id[r["id"]]
        assert r["render"] == "plain" and r["text"] == it["prompt_render"]
        assert r["positions"]["positions"] == [s["pos"] for s in it["sites"]]
        assert r["extra"] == {"n_tokens": len(it["sites"])}
        assert r["text"].endswith("<|im_start|>assistant")


def test_roleplay_plan_is_the_pinned_workspacebench_plan():
    rows = readplan.plan("benign_roleplay")
    assert readplan.plan_sha(rows) == ROLEPLAY_PLAN_SHA
    bank_rows = json.loads(br_bank.HB_ONLY_BANK.read_text(encoding="utf-8"))["items"]
    assert [r["id"] for r in rows] == [it["id"] for it in bank_rows]
    for r, it in zip(rows, bank_rows, strict=True):
        assert r["family"] == br_bank.FAMILY and r["render"] == "chat_context"
        first, end = it["read"]["span"]
        assert r["positions"]["positions"] == list(range(first, end + 1))
        assert r["messages"][-1]["role"] == "user"


def test_plan_sha_ignores_layers_lite_site_and_note_only():
    rows = readplan.plan("story")
    sha = readplan.plan_sha(rows)
    assert readplan.plan_sha(readplan.plan("story", [44])) == sha
    assert readplan.plan_sha([{**r, "note": "x", "lite_positions": {}} for r in rows]) == sha
    moved = [{**rows[0], "positions": {"kind": "positions", "positions": [0]}}, *rows[1:]]
    assert readplan.plan_sha(moved) != sha
    assert readplan.plan_sha(rows[1:]) != sha


def test_unknown_family_is_an_error():
    with pytest.raises(KeyError, match="unknown family 'nope'"):
        readplan.plan("nope")


def test_capture_file_that_disagrees_with_the_bank_is_refused(tmp_path, monkeypatch):
    rows = json.loads((story_bank.BANK_DIR / "capture_rows.json").read_text(encoding="utf-8"))
    rows[0]["eval_positions"] = rows[0]["eval_positions"][:-1]
    (tmp_path / "capture_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(story_bank, "BANK_DIR", tmp_path)
    with pytest.raises(ValueError, match="not the bank's sites"):
        readplan.plan("story")


def test_command_writes_one_file_per_family(tmp_path, capsys):
    assert cli.main(["plan", f"out={tmp_path}", "families=story,overrefusal", "layers=44,52"]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["overrefusal.jsonl", "story.jsonl"]
    raw = (tmp_path / "overrefusal.jsonl").read_bytes()
    assert raw.isascii() and raw.count(b"\n") == 427
    rows = [json.loads(x) for x in raw.decode().splitlines()]
    assert rows == readplan.plan("overrefusal", [44, 52])
    keys = ["family", "id", "render", "positions", "layers", "text", "extra", "lite_positions"]
    assert list(rows[0]) == keys
    out = capsys.readouterr().out
    assert "story: 20 items, 160 positions, 320 cells at 2 layers" in out
    assert f"plan_sha {readplan.plan_sha(rows)}" in out
    # the same bytes on a second run
    assert cli.main(["plan", f"out={tmp_path}", "families=overrefusal", "layers=44,52"]) == 0
    assert (tmp_path / "overrefusal.jsonl").read_bytes() == raw


def test_command_keys(capsys):
    cmd = cli.command_for(["plan"])[0]
    assert cmd.families == list(readplan.FAMILIES) and cmd.layers is None
    assert cli.command_for(["plan", "layers=20,44"])[0].layers == [20, 44]
    for bad in (["plan", "families=nope"], ["plan", "layers=x"], ["plan", "nope=1"]):
        with pytest.raises(SystemExit) as e:
            cli.command_for(bad)
        assert e.value.code == 2
    assert "unknown family 'nope'" in capsys.readouterr().err
