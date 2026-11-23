"""The sibling ladder's method list and readout paths are configurable (``--methods-config``,
``--out``, ``--probes-root``); with none of them given everything is as before."""

import importlib.util
import json
import sys

import pytest

from hallucination_bench import REPO_ROOT
from hallucination_bench.evals.sibling_probe import um_ladder as U

SCRIPT = REPO_ROOT / "scripts" / "sibling_probe" / "um_ladder.py"
BUILT_IN = ("s3d", "mlnla-rl600", "jlens", "nla-rl-L42")


# what ``U.configure(domain)`` rebinds (it is one-way: country is the import-time state)
DOMAIN_STATE = (
    "CATEGORY",
    "ATTRIBUTE_QUESTION",
    "COMMIT_SYSTEM",
    "COMMIT_PROMPT_VERSION",
    "GOLD_MAP",
    "GOLD_EQUIV",
    "_CAP_ALL",
    "_COMPATIBLE",
    "MARK_NOUN",
)


@pytest.fixture(autouse=True)
def _clean():
    saved = {k: getattr(U, k) for k in DOMAIN_STATE}
    U.reset_methods()
    yield
    U.reset_methods()
    for k, v in saved.items():
        setattr(U, k, v)


@pytest.fixture()
def driver():
    """A fresh copy of the driver module (its paths are module globals)."""
    spec = importlib.util.spec_from_file_location("um_ladder_driver", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fm():
    return U.FastMatcher(U.load_lexicon())


def _ctx(fm, gold="ukraine"):
    item = {
        "gold": gold,
        "messages": [{"role": "user", "content": "I grew up eating my grandmother's borscht."}],
        "response": "",
    }
    return U.item_context(fm, item)


def _strip(units):
    return [{k: v for k, v in u.items() if k not in ("method", "uid")} for u in units]


def test_defaults_are_the_four_built_in_methods():
    assert U.METHODS == BUILT_IN and U.methods() == BUILT_IN
    assert [U.method_kind(m) for m in BUILT_IN] == ["packed", "prose", "tokens", "prose"]
    assert [U.eff_layer(m, 28) for m in BUILT_IN] == [28, 28, 28, 42]
    assert U.method_label("s3d") == "s3d O-lens"
    # a method named nowhere reads as prose at its own layer, as before
    assert U.method_kind("other") == "prose" and U.eff_layer("other", 28) == 28


def test_built_in_methods_cannot_be_redeclared():
    U.add_method("jlens", "tokens")  # the same settings: accepted, nothing changes
    with pytest.raises(ValueError, match="built in"):
        U.add_method("jlens", "prose")
    with pytest.raises(ValueError, match="built in"):
        U.add_method("s3d", "packed", fixed_layer=42)
    with pytest.raises(ValueError, match="not one of"):
        U.add_method("new", "bags")
    with pytest.raises(ValueError, match="unknown or repeated"):
        U.select_methods(["jlens", "never-declared"])
    assert U.methods() == BUILT_IN


def test_added_methods_read_like_the_built_in_one_of_their_kind(fm):
    U.add_method("logit_lens", "tokens", label="Logit lens")
    U.add_method("olens-new", "packed")
    U.add_method("nla", "prose")
    U.add_method("nla-L30", "prose", fixed_layer=30)
    U.select_methods(["logit_lens", "olens-new", "nla"])
    assert U.methods() == ("logit_lens", "olens-new", "nla") and U.METHODS == BUILT_IN
    assert U.method_label("logit_lens") == "Logit lens" and U.method_label("nla") == "nla"
    assert U.eff_layer("nla", 28) == 28 and U.eff_layer("nla-L30", 28) == 30
    ctx = _ctx(fm)
    tok = {"id": "x", "layer": 44, "pos": 3, "tokens": [" Chinese", " food"], "scores": [2.0, 1.0]}
    packed = {"id": "x", "layer": 44, "pos": 3, "samples": ["- a Chinese user\n- from Japan"]}
    prose = {"id": "x", "layer": 44, "pos": 3, "samples": ["a Chinese user", "from Japan"]}
    for new, old, row, kw in (
        ("logit_lens", "jlens", tok, {"summary": "Chinese or Japanese food"}),
        ("olens-new", "s3d", packed, {}),
        ("nla", "mlnla-rl600", prose, {}),
    ):
        got, got_strata = U.build_mentions(fm, new, row, ctx, **kw)
        want, want_strata = U.build_mentions(fm, old, row, ctx, **kw)
        assert got and _strip(got) == _strip(want)
        assert {u["method"] for u in got} == {new}
        assert _strip(got_strata) == _strip(want_strata)
    # a prose method does not split a packed string; a packed method does
    assert len(U.cell_samples("nla", packed)) == 1 and len(U.cell_samples("olens-new", packed)) == 2


def test_driver_paths_are_unchanged_without_options(driver):
    um = REPO_ROOT / "outputs" / "sibling" / "um"
    src = REPO_ROOT / "somewhere"
    driver.setup_domain("country")
    assert (driver.OUT, driver.LABEL_DIR) == (um, um / "label")
    assert (driver.PROBES_DIR, driver.PROBES_ADJ_DIR) == (um / "probes", um / "probes_adj")
    assert driver.readout_path(src, "s3d") == (
        src / "outputs" / "sibling" / "readouts" / "s3d-rl600-all11" / "sibling_eval.jsonl"
    )
    assert driver.token_methods() == ["jlens"]
    driver.setup_domain("holiday")
    assert (driver.OUT, driver.LABEL_DIR) == (um / "holiday", um / "holiday" / "label")
    assert driver.PROBES_DIR == um / "holiday" / "probes"
    assert driver.PROBES_ADJ_DIR == um / "holiday" / "probes_adj"
    assert driver.readout_path(src, "jlens") == (
        src / "outputs" / "sibling" / "readouts" / "jlens-all11" / "sibling_holiday.jsonl"
    )


def test_out_and_probes_root_move_the_dirs(driver, tmp_path):
    driver.setup_domain("capital_city", tmp_path / "run", tmp_path / "frozen")
    assert driver.OUT == tmp_path / "run" / "capital_city"
    assert driver.LABEL_DIR == tmp_path / "run" / "capital_city" / "label"
    assert driver.PROBES_DIR == tmp_path / "frozen" / "capital_city" / "probes"
    assert driver.PROBES_ADJ_DIR == tmp_path / "frozen" / "capital_city" / "probes_adj"


def test_methods_config_names_methods_and_files(driver, tmp_path):
    cfg = tmp_path / "methods.json"
    cfg.write_text(
        json.dumps(
            {
                "methods": [
                    {"name": "s3d", "readouts": str(tmp_path / "s3d.jsonl")},
                    {
                        "name": "logit_lens",
                        "kind": "tokens",
                        "readouts": "ll/sibling_{category}.jsonl",
                    },
                    {"name": "jlens"},
                ]
            }
        )
    )
    driver.METHOD_FILES.update(driver.load_methods_config(cfg))
    driver.setup_domain("holiday")
    src = tmp_path / "src"
    assert U.methods() == ("s3d", "logit_lens", "jlens")
    assert driver.token_methods() == ["logit_lens", "jlens"]
    assert driver.readout_path(src, "s3d") == tmp_path / "s3d.jsonl"
    assert driver.readout_path(src, "logit_lens") == (
        src / "outputs" / "sibling" / "readouts" / "ll" / "sibling_holiday.jsonl"
    )
    assert driver.readout_path(src, "jlens") == (
        src / "outputs" / "sibling" / "readouts" / "jlens-all11" / "sibling_holiday.jsonl"
    )
    for bad in (
        [{"name": "new"}],  # no readout file
        [{"name": "jlens", "kind": "prose"}],  # a built-in kind is fixed
        [{"name": "new", "readouts": "x.jsonl", "path": "y"}],  # unknown key
    ):
        cfg.write_text(json.dumps({"methods": bad}))
        with pytest.raises(SystemExit):
            driver.load_methods_config(cfg)


def test_mentions_runs_for_configured_methods(driver, tmp_path, monkeypatch):
    """`mentions` end to end (no API): two new methods, their own files, their own out dir."""
    bank = json.loads((REPO_ROOT / "data" / "sibling_probe" / "items.json").read_text())["items"]
    item = next(it for it in bank if it["category"] == "country" and it["gold"] == "paraguay")
    other = next(it for it in bank if it["category"] == "holiday")
    cell = {"id": item["id"], "layer": "44", "pos": "30"}  # a reader writes strings
    rows = {
        "nla": [
            {**cell, "samples": ["The user is from Japan.", "A Japanese user."]},
            {**cell, "id": other["id"], "samples": ["The user is from Japan."]},  # not country
        ],
        "logit_lens": [{**cell, "tokens": [" Japan", " the"], "scores": [3.0, 1.0]}],
    }
    for name, rs in rows.items():
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    cfg = tmp_path / "methods.json"
    cfg.write_text(
        json.dumps(
            [
                {"name": "nla", "readouts": str(tmp_path / "nla.jsonl")},
                {
                    "name": "logit_lens",
                    "kind": "tokens",
                    "readouts": str(tmp_path / "logit_lens.jsonl"),
                },
            ]
        )
    )
    out = tmp_path / "um"
    argv = ["um_ladder.py", "--src-root", str(REPO_ROOT), "--methods-config", str(cfg)]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(out), "mentions"])
    driver.main()
    units = [json.loads(x) for x in (out / "mentions.jsonl").read_text().splitlines()]
    assert [(u["method"], u["concept"], u["layer"], u["pos"]) for u in units] == [
        ("nla", "japan", 44, 30),
        ("logit_lens", "japan", 44, 30),
    ]
    assert [len(u["samples"]) for u in units] == [2, 1]
    assert units[1]["samples"][0]["summary_status"] == "missing"  # `summarize` not run
    counts = json.loads((out / "counts.json").read_text())
    assert counts["cells"] == {"nla": 1, "logit_lens": 1}
    assert counts["units"] == {"nla": 1, "logit_lens": 1}
