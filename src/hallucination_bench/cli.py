"""``hbench <command> key=value ...``: ``run``, ``calibrate``, ``report``, ``consistency-v2``,
``pilot-dump``, ``pilot-dump-v2``, ``units-v2``, ``index``, ``final`` and the evals' own commands.
Every command is a ``pydra.Config`` (``command.Command``): keys and defaults in ``__init__``,
types in ``finalize()``; ``--show`` prints the resolved config, ``--help`` lists the keys."""

# The shared commands and helpers live here. An eval with its own subcommands keeps them in
# ``evals/<eval>/cli.py`` (a ``COMMANDS`` table, listed in EVAL_CLIS); an eval with its own driver
# behind ``run|report --eval`` keeps that in ``evals/<eval>/cli.py`` too (plan 0012 D2).

import copy
import importlib
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pydra

from hallucination_bench import DATA, paths
from hallucination_bench.command import (
    EXIT_USAGE,
    HELP_FLAGS,
    Command,
    UsageError,
    aliased,
    boolean,
    choice,
    help_text,
    integer,
    many,
    number,
    optional,
    parse,
    text,
    usage,
)
from hallucination_bench.command import show as show_text
from hallucination_bench.llm import DEFAULT_EFFORT, JudgeConfigError

DEFAULT_MODEL = "claude-opus-5-5"
INDEX_METHODS = DATA / "index" / "methods.json"  # plan 0013: method <-> arm mapping
READOUTS_FILE = "hallucination.jsonl"
# Descriptive names for the values the index is built on. Each is replaced by the value it
# stands for when the command line is parsed, so arm.json, caches, manifests and reports record
# the value itself, exactly as a run started with the value does.
MAP_PROMPT_NAMES = {"current": "claimmap-v2.9"}  # the claim-map prompt the index uses
BANK_NAMES = {"main": "x3"}  # the chat question bank the index's main family runs on
INDEX_MANIFEST = DATA / "index" / "v8_manifest.json"  # the manifest of the results of record
AM_EVAL = "agentic_misalignment"  # plan 0008: its own driver (dispatched first in run / report)
# modules exposing ``COMMANDS``: their subcommands follow the shared ones, in this order
EVAL_CLIS: list[str] = [
    "hallucination_bench.tryout",  # the two-question try-out (`hbench try`)
    "hallucination_bench.evals.overrefusal.cli",  # plan 0015
    "hallucination_bench.evals.benign_roleplay.cli",  # plan 0023
]


def _layers(s: str | None) -> list[int] | None:
    return [int(x) for x in s.split(",") if x.strip()] if s else None


def _effort(s: str) -> str | None:
    return None if s in ("", "none", "default") else s


def resolve_arms(args: Command) -> list[tuple[str, Path]]:
    arms: list[tuple[str, Path]] = []
    for a in args.arm or []:
        if "=" not in a:
            raise SystemExit(f"--arm expects name=path, got {a!r}")
        name, path = a.split("=", 1)
        arms.append((name, Path(path)))
    if args.readouts_root:
        root = Path(args.readouts_root)
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            f = d / READOUTS_FILE
            if f.exists():
                arms.append((d.name, f))
    if args.arms:
        keep = set(args.arms.split(","))
        unknown = keep - {n for n, _ in arms}
        if unknown:
            raise SystemExit(f"--arms names not found: {sorted(unknown)}")
        arms = [(n, p) for n, p in arms if n in keep]
    names = [n for n, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate arm names: {names}")
    if not arms:
        raise SystemExit("no arms: pass --arm name=path and/or --readouts-root DIR")
    for n, p in arms:
        if not p.exists():
            raise SystemExit(f"[{n}] readouts file not found: {p}")
    return arms


def _excuse_verifier(args: Command, map_prompt: str) -> str | None:
    """Plan 0029 Amendment 2: ``--excuse-verifier`` (default: ``cverify-v1`` for single-call
    claimmap-v2.9 roots, off otherwise, including any ``--anchor v5c`` run; ``off`` disables it
    for diagnostics). An explicit verifier with ``--anchor v5c`` is refused by
    ``check_verifier`` (exit 3). Rebuild / report replay the arm dir's recorded setting whatever
    this says; a judge run that differs exits 3."""
    from hallucination_bench.judges import claimmap_v2 as cm
    from hallucination_bench.judges import continuation_verifier as cv

    flag = getattr(args, "excuse_verifier", None)
    if flag == "off":
        return None
    v29 = cm.is_v29(map_prompt) if map_prompt in cm.VERSIONS else False
    anchored = (getattr(args, "anchor", None) or "none") != "none"
    if flag is None:
        return cv.VERSION if v29 and not anchored else None
    if not v29:
        raise SystemExit(f"--excuse-verifier {flag} needs --map-prompt claimmap-v2.9")
    return flag


def _models(args: Command):
    from hallucination_bench.evals.wsbench_chat.run import Models

    map_prompt = getattr(args, "map_prompt", None) or "claimmap-v2.7"
    return Models(
        judge=args.judge_model,
        summarizer=args.summarizer_model,
        classifier=args.classifier_model,
        effort=_effort(args.effort),
        map_model=getattr(args, "map_model", None),
        map_anchor=getattr(args, "anchor", None) or "none",
        eval=getattr(args, "eval", None) or "wsbench_chat",
        map_prompt=map_prompt,
        situation_model=getattr(args, "situation_model", None),
        excuse_verifier=_excuse_verifier(args, map_prompt),
    )


def _stages(args: Command) -> list[str]:
    """``--stages`` (default: every stage); ``--skip-classify`` keeps its plan-0001 meaning
    (summary, span and verify only) when ``--stages`` is not given."""
    from hallucination_bench.evals.wsbench_chat.run import EXTRA_STAGES, STAGES, expand_stages

    if args.stages:
        sel = [x.strip() for x in args.stages.split(",") if x.strip()]
    elif args.skip_classify:
        sel = ["summary", "span", "verify"]
    else:
        sel = [*STAGES, *EXTRA_STAGES]  # cverify: a no-op unless the verifier is on
    try:
        expand_stages(sel, anchor=getattr(args, "anchor", None) or "none")
    except ValueError as e:
        raise SystemExit(str(e)) from None
    return sel


def _items(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _apply_bank(args: Command, *, readouts: bool = False) -> None:
    """``--bank`` (plans 0028 / 0029): point the wsbench_chat bank at a named bank; with
    ``readouts`` (``run``), ``--bank x3`` and neither ``--arm`` nor ``--readouts-root``, the
    readouts default to ``paths.X3_READOUTS`` ($HBENCH_X3_READOUTS). An x3 run records the bank
    in ``arm.json``; judging into an arm dir recorded on another bank exits 3, and rebuild /
    report replay the recorded bank when ``--bank`` is omitted (``run.check_bank`` /
    ``run.replay_bank``)."""
    name = getattr(args, "bank", None)
    if not name:
        return
    if (getattr(args, "eval", None) or "wsbench_chat") != "wsbench_chat":
        raise SystemExit("--bank is a wsbench_chat (main eval) option")
    from hallucination_bench.evals.wsbench_chat import bank

    bank.use_bank(name)
    if readouts and name == "x3" and not args.readouts_root and not args.arm:
        args.readouts_root = str(paths.X3_READOUTS)


def cmd_run(args: Command, llm: Any = None) -> int:
    """``llm`` replaces the judge client of the main driver (``hbench try --dry-run``)."""
    _apply_bank(args, readouts=True)
    if getattr(args, "eval", None) == AM_EVAL:
        from hallucination_bench.evals.agentic_misalignment import cli as amcli

        return amcli.run(args)
    if getattr(args, "items", None):
        raise SystemExit("--items is an agentic_misalignment option")
    from hallucination_bench.evals.wsbench_chat import run as wrun

    arms = resolve_arms(args)
    models = _models(args)
    out = Path(args.out)
    if getattr(args, "joint", False):
        return _cmd_run_joint(args, arms, models, out)
    if args.rebuild:
        for n, p in arms:
            s = wrun.rebuild_arm(n, p, out, models)
            _print_summary(n, s)
        return 0
    summaries = wrun.run_arms(
        arms,
        out,
        models,
        layers=_layers(args.layers),
        limit=args.limit,
        rpm=args.rpm,
        concurrency=args.concurrency,
        allow_missing=args.allow_missing,
        classify=not args.skip_classify,
        show_prompts=args.show_prompts,
        stages=_stages(args),
        sample_seed=args.sample_seed,
        llm=llm,
    )
    for n, s in summaries.items():
        _print_summary(n, s)
    # Failed calls are cached as failures and re-queued by the next run; a non-zero exit lets
    # the driver re-run the stage instead of moving on.
    n_failed = sum(
        v for s in summaries.values() for k, v in s["health"].items() if k.startswith("n_failed_")
    )
    if n_failed:
        print(f"{n_failed} failed calls remain; re-run to retry them", file=sys.stderr)
        return 4
    return 0


def _cmd_run_joint(args: Command, arms: list, models, out: Path) -> int:
    """Plan 0029 §3.7: ``hbench run --joint``: the joint 11-layer cells of each ``--arms``
    target (sources: ``--joint-from`` in precedence order, default the arm itself)."""
    from hallucination_bench.evals.wsbench_chat import run as wrun
    from hallucination_bench.judges.claimmap_v2 import JOINT_LAYERS

    paths = dict(arms)
    if args.joint_from:
        if len(arms) != 1:
            raise SystemExit("--joint-from needs exactly one target arm (--arms)")
        src_names = [x.strip() for x in args.joint_from.split(",") if x.strip()]
        args2 = copy.copy(args)
        args2.arms = ",".join(src_names)
        paths |= dict(resolve_arms(args2))
    rc = 0
    for name, _p in arms:
        names = [x.strip() for x in args.joint_from.split(",")] if args.joint_from else [name]
        sources = [(n, paths[n]) for n in names]
        if args.rebuild:
            import asyncio

            s2 = asyncio.run(wrun.rebuild_joint_async(name, out, models))
        else:
            s2 = wrun.run_joint(
                name,
                sources,
                out,
                models,
                layers=_layers(args.layers) or JOINT_LAYERS,
                limit=args.limit,
                rpm=args.rpm,
                concurrency=args.concurrency,
                show_prompts=args.show_prompts,
                sample_seed=args.sample_seed,
            )
        h = s2["health"]
        print(
            f"[{name}] joint map ok={h['n_map_ok']} failed={h['n_map_failed']} "
            f"pending={h['n_map_pending']}",
            flush=True,
        )
        # plan 0029 Amendment 2: failed verifier calls are failed rows too (re-asked next run)
        if (h["n_map_failed"] or h.get("n_verifier_failed", 0)) and not args.rebuild:
            rc = 4
    return rc


def _print_summary(name: str, s: dict) -> None:
    r = s["overall"]["rates"]
    h = s["health"]

    def f(k: str) -> str:
        v = r.get(k)
        return "—" if v is None else f"{v:.3f}"

    failed = {k: h[f"n_failed_{k}"] for k in ("summary", "span", "verify", "classify", "map")}
    print(
        f"[{name}] layers={s['config']['judged_layers']} judged={h['n_judged_cells']} "
        f"pending={h['n_pending_cells']} failed={failed} "
        f"| v5c={f('hallucination_rate')} concept={f('concept_rate')} detail={f('detail_rate')} "
        f"strict={f('strict_hallucination_rate')} unsupported={f('unsupported_rate')} "
        f"unclassified_spans={h['n_unclassified']} spend_total=${s['spend']['total_usd']:.2f}",
        flush=True,
    )


def cmd_calibrate(args: Command) -> int:
    from hallucination_bench import calibrate

    if args.taxonomy == "v2":
        return _calibrate_v2(args)
    model = args.classifier_model or args.judge_model
    res = calibrate.run(
        args.split,
        model=model,
        effort=_effort(args.effort),
        out_root=Path(args.out),
        repeats=args.repeats,
        rpm=args.rpm,
        concurrency=args.concurrency,
        cases_path=Path(args.cases) if args.cases else None,
    )
    print(calibrate.render_markdown(res))
    sp = res["spend_this_run"]
    print(f"\nspend this run: ${sp['usd']:.4f} ({sp['calls']} calls)")
    if args.split == "heldout":
        print(f"GATE {'PASSED' if res['gate_passed'] else 'FAILED'}")
    return 0


def _calibrate_v2(args: Command) -> int:
    from hallucination_bench import calibrate_v2

    res = calibrate_v2.run(
        args.split,
        model=args.map_model or args.judge_model,
        effort=_effort(args.effort),
        out_root=Path(args.out),
        repeats=args.repeats,
        rpm=args.rpm,
        concurrency=args.concurrency,
        cases_file=Path(args.cases) if args.cases else None,
        anchor=args.anchor,
        prompt=args.prompt,
    )
    print(calibrate_v2.render_markdown(res))
    sp = res["spend_this_run"]
    print(f"\ncoverage: {json.dumps(res['coverage'])}")
    print(f"spend this run: ${sp['usd']:.4f} ({sp['calls']} calls)")
    if args.split == "heldout":
        print(f"GATE {'PASSED' if res['gate_passed'] else 'FAILED'}")
    return 0


def _index_dest(out: Path, judge_model: str) -> Path:
    from hallucination_bench.evals.wsbench_chat.run import model_slug

    return out / "index" / model_slug(judge_model)


def cmd_index(args: Command) -> int:
    from hallucination_bench import index

    out = Path(args.out)
    sources = index.default_sources(out, args.judge_model, args.version)
    for fam, flag, roots in (
        (index.MAIN, "--main-root", args.main_root),
        (index.AM_CLAIMS, "--am-root", args.am_root),
        (index.AM_SITUATION, "--am-situation-root", args.am_situation_root),
        (index.SIBLING, "--um-root", args.um_root),
        (index.STORY, "--story-root", args.story_root),
        (index.OVERREFUSAL, "--overrefusal-root", args.overrefusal_root),
        (index.BENIGN_ROLEPLAY, "--benign-roleplay-root", args.benign_roleplay_root),
    ):
        if roots:
            if missing := [r for r in roots if not Path(r).is_dir()]:
                raise SystemExit(f"{flag}: not a directory: {missing}")
            sources[fam] = [Path(r) for r in roots]
    if args.am_root and not args.am_situation_root:  # default: the AM root
        sources[index.AM_SITUATION] = list(sources[index.AM_CLAIMS])
    # an opted-in safety root must hold its counts, else the family (and full / matched)
    # silently vanishes
    for fam in index.safety_families(args.version):
        if gone := [
            str(r) for r in sources.get(fam, []) if not (r / index.SAFETY_COUNTS).is_file()
        ]:
            raise SystemExit(f"{fam}: no {index.SAFETY_COUNTS} in {gone}")
    if not sources.get(index.BENIGN_ROLEPLAY):
        print(
            f"warning: hindex-{args.version} without a benign_roleplay root "
            "(--benign-roleplay-root or $HBENCH_SAFETY_BR_OUT): the family, and so full / "
            "matched, are absent",
            file=sys.stderr,
        )
    try:  # plan 0022's unit guard, after the checks above (build_index repeats it)
        index.check_safety_units(sources, args.version)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    try:
        built = index.build_index(
            sources, index.load_methods(Path(args.methods)), args.workers, version=args.version
        )
    except (ValueError, OSError) as e:  # bad config (incl. JSONDecodeError) / unreadable file
        raise SystemExit(str(e)) from None
    dest = Path(args.dest) if args.dest else _index_dest(out, args.judge_model)
    js, md = index.write_index(built, dest)
    print(index.render_md(built))
    print(f"wrote {js}, {md}")
    return 0


def _cmd_report(args: Command) -> int:
    _apply_bank(args)
    if getattr(args, "eval", None) == AM_EVAL:
        from hallucination_bench.evals.agentic_misalignment import cli as amcli

        return amcli.report(args)
    from hallucination_bench import report

    if args.view == "v2":
        from hallucination_bench import report_v2

        root = report_v2.report(
            Path(args.out),
            args.judge_model,
            rebuild=not args.no_rebuild,
            audit_n=args.audit_n if args.audit_n is not None else 10,
            seed=args.seed,
            verdict_n=args.audit_verdict_n,
            eval_name=args.eval,
        )
        print(
            f"wrote {root / 'report_v2.md'}, {root / 'report_v2.json'}, "
            f"{root / 'probe_candidates.jsonl'}, {root / 'audit_samples_v2.md'}"
        )
        return 0

    root = report.report(
        Path(args.out),
        args.judge_model,
        rebuild=not args.no_rebuild,
        audit_n=args.audit_n if args.audit_n is not None else 15,
        seed=args.seed,
        eval_name=args.eval,
    )
    print(f"wrote {root / 'report.md'}, {root / 'report.json'}, {root / 'audit_samples.md'}")
    return 0


def cmd_consistency_v2(args: Command) -> int:
    from hallucination_bench import consistency_v2

    if args.rescore:
        res = consistency_v2.rescore(Path(args.out), args.map_model or args.judge_model)
        print(consistency_v2.render_markdown(res))
        print("\nre-scored from the cached r0/r1 cells: no API calls")
        return 0
    if not args.arms:
        raise SystemExit("--arms is required unless --rescore")
    res = consistency_v2.run(
        args.arms,
        readouts_root=Path(args.readouts_root),
        out_root=Path(args.out),
        models=_models(args),
        n=args.n,
        seed=args.seed,
        rpm=args.rpm,
        concurrency=args.concurrency,
    )
    print(consistency_v2.render_markdown(res))
    sp = res["spend_this_run"]
    print(f"\nspend this run: ${sp['usd']:.4f} ({sp['calls']} calls)")
    return 0


def cmd_pilot_dump(args: Command) -> int:
    from hallucination_bench import report
    from hallucination_bench.calibrate import validate_case

    cases = report.pilot_dump(
        Path(args.out),
        args.judge_model,
        args.arms.split(","),
        per_stratum=args.per_stratum,
        seed=args.seed,
        layers=_layers(args.layers),
    )
    errs = [e for c in cases for e in validate_case(c, require_gold=False)]
    if errs:
        print("warning: " + "; ".join(errs[:10]), file=sys.stderr)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_sp = sum(len(c["spans"]) for c in cases)
    n_sampled = sum(s["sampled"] for c in cases for s in c["spans"])
    print(f"wrote {len(cases)} cases ({n_sampled} sampled spans, {n_sp} spans total) to {dest}")
    return 0


def cmd_pilot_dump_v2(args: Command) -> int:
    from hallucination_bench import report_v2
    from hallucination_bench.calibrate_v2 import validate_case

    cases = report_v2.pilot_dump_v2(
        Path(args.out),
        args.judge_model,
        args.arms.split(","),
        per_stratum=args.per_stratum,
        seed=args.seed,
        layers=_layers(args.layers),
    )
    errs = [e for c in cases for e in validate_case(c, require_gold=False)]
    if errs:
        print("warning: " + "; ".join(errs[:10]), file=sys.stderr)
    _write_jsonl(Path(args.dest), cases)
    n_units = sum(len(c["gold"]["units"]) for c in cases)
    print(f"wrote {len(cases)} cells ({n_units} units to label) to {args.dest}")
    return 0


def cmd_units_v2(args: Command) -> int:
    """Fill each case's ``gold.units`` with the production units (leaf null) for labelling."""
    from hallucination_bench.calibrate_v2 import skeleton

    src = Path(args.cases)
    cases = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = [skeleton(c) for c in cases]
    _write_jsonl(Path(args.dest), out)
    print(f"wrote {len(out)} cases ({sum(len(c['gold']['units']) for c in out)} units)")
    return 0


def _write_jsonl(dest: Path, rows: list[dict]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for c in rows:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------- key groups
# Keys several commands share, added in ``__init__`` (the names are the old flags' names with
# dashes turned to underscores).


def judge_keys(c: Command) -> None:
    """Where the outputs go and how the judge is called."""
    c.key("out", "outputs", text, "output root")
    c.key("judge_model", DEFAULT_MODEL, text, "the judge model; it also names <out>/<eval>/<model>")
    c.key("effort", DEFAULT_EFFORT, text, "Anthropic output_config.effort ('none' omits it)")
    c.key("rpm", 1500.0, number, "requests per minute, one pacer for the whole process")
    c.key("concurrency", 64, integer, "judge calls in flight at once")


def eval_key(c: Command) -> None:
    c.key(
        "eval",
        "wsbench_chat",
        choice(_eval_names),
        "which bank and output root <out>/<eval>/ (story: plan 0007; agentic_misalignment: plan "
        "0008, its own driver)",
    )


def bank_key(c: Command) -> None:
    c.key(
        "bank",
        None,
        optional(aliased(BANK_NAMES, choice("wsb", "x3"))),
        "wsbench_chat only: wsb = the WSB 149 x 8 bank; x3 = plan 0028's 3x bank "
        "($HBENCH_X3_BANK/items.json, default data/wsbench_chat_x3; run: readouts default "
        f"{paths.X3_READOUTS} = $HBENCH_X3_READOUTS). Not given: the WSB bank, or "
        "$HBENCH_WSBENCH_CHAT_ITEMS when set. main = the bank the index's main family runs on "
        "(it stands for x3)",
    )


def anchor_key(c: Command) -> None:
    c.key(
        "anchor",
        "none",
        choice("none", "v5c"),
        "claim map: none = single-call (no established spans, no verify); v5c = plan rev 12 "
        "(v5c wrong spans passed as ESTABLISHED units)",
    )


def _eval_names() -> list[str]:
    from hallucination_bench.evals import EVALS

    return sorted(EVALS)


def _prompt_choices() -> list[str]:
    from hallucination_bench.judges.claimmap_v2 import VERSIONS

    return list(VERSIONS)


def _verifier_choices() -> list[str]:
    from hallucination_bench.judges import continuation_verifier as cv

    return [*cv.VERSIONS, "off"]


# ------------------------------------------------------------------------------------ commands


class Run(Command):
    about = "judge arms (summarise -> span -> verify -> map -> classify)"

    def __init__(self) -> None:
        super().__init__()
        judge_keys(self)
        self.key("arm", None, many, "name=path/to/hallucination.jsonl (repeatable)")
        self.key(
            "readouts_root",
            None,
            optional(text),
            "DIR with <arm>/hallucination.jsonl subdirectories (agentic_misalignment: "
            f"<arm>/agentic_misalignment.jsonl, default {paths.READOUTS_ROOT})",
        )
        self.key("arms", None, optional(text), "comma-separated arm names to keep")
        self.key(
            "layers", None, optional(text), "comma-separated layers (each arm judges those it has)"
        )
        self.key(
            "limit",
            None,
            optional(integer),
            "judge only the first N non-empty cells per arm x layer (smoke); agentic_misalignment "
            "map: N mapped cells per layer, drawn among the detection positives",
        )
        self.key(
            "sample_seed",
            None,
            optional(integer),
            "with limit=: draw the N cells per arm x layer by a seeded shuffle instead of the "
            "first N (plan 0029 pilots: seeded, stratified by layer)",
        )
        self.key(
            "items",
            None,
            optional(text),
            "agentic_misalignment only: comma-separated item ids to judge (pilot)",
        )
        self.key("summarizer_model", None, optional(text), "the summary model (default: judge)")
        self.key("classifier_model", None, optional(text), "the classifier model (default: judge)")
        self.key("map_model", None, optional(text), "claim-map (:M) judge (default: judge)")
        self.key(
            "map_prompt",
            "claimmap-v2.7",
            aliased(MAP_PROMPT_NAMES, choice(_prompt_choices)),
            "claim-map prompt version (plan 0029; recorded in arm.json, a run that differs from "
            "an arm dir's record is refused; agentic_misalignment maps claimmap-v2.8 / -v2.9 to "
            "their +am3 versions; v8 roots use claimmap-v2.9, plan 0029 amendment). current = the "
            "prompt the index uses (it stands for claimmap-v2.9)",
        )
        self.key(
            "excuse_verifier",
            None,
            optional(choice(_verifier_choices)),
            "plan 0029 amendment 2: the continuation verifier that decides the continuation_miss "
            "excuse (not given: cverify-v1 for claimmap-v2.9 roots, off otherwise and under "
            "anchor=v5c, which refuses an explicit verifier with exit 3; 'off' = the v2.9 "
            "judge-tag rule, diagnostics only; recorded in arm.json, a run that differs from an "
            "arm dir's record exits 3; hbench final requires cverify-v1)",
        )
        self.key(
            "situation_model",
            None,
            optional(text),
            "agentic_misalignment, v8 roots: the amsit-v2 situation judge (stages=situation; "
            "default claude-sonnet-5-5)",
        )
        self.key(
            "joint",
            False,
            boolean,
            "plan 0029: judge the joint 11-layer cells (sample 0 of every layer in one call; "
            "needs map_prompt=claimmap-v2.8 or later (v8: claimmap-v2.9); agentic_misalignment: "
            "same as stages=joint)",
        )
        self.key(
            "joint_from",
            None,
            optional(text),
            "with --joint (chat / story): the source arms in precedence order (each layer from "
            "the first that has it, the index's rule); default the target arm itself",
        )
        anchor_key(self)
        eval_key(self)
        bank_key(self)
        self.key(
            "stages",
            None,
            optional(text),
            "comma-separated stages that may call the API: summary,span,verify,classify,map,"
            "cverify (default: all; dependencies are added: map runs summary first if missing, "
            "and span only under anchor=v5c; cverify = the continuation verifier after map, v2.9 "
            "verifier roots only; single-call map also calls span on a seeded 10% item subsample "
            "for the v5c comparison). agentic_misalignment: detect,summ,verify,map,intent,cverify "
            "(verify = Stage H, the headline verifier; default all; no dependencies are added, "
            "unselected stages are cache-only)",
        )
        self.key("allow_missing", False, boolean, "judge a readouts file that lacks bank cells")
        self.key(
            "rebuild",
            False,
            boolean,
            "regenerate cells.jsonl/summary.json from the cache (no API calls)",
        )
        self.key(
            "skip_classify",
            False,
            boolean,
            "summary + span + verify only (pilot); ignored when stages= is given",
        )
        self.key(
            "show_prompts",
            0,
            integer,
            "write the prompts of the first K cells per arm x layer to <arm>/prompts_sample.txt",
        )

    def execute(self) -> int:
        return cmd_run(self)


class Calibrate(Command):
    about = "score the classifier on calibration cases"

    def __init__(self) -> None:
        super().__init__()
        judge_keys(self)
        self.key("split", pydra.REQUIRED, choice("dev", "heldout"), "the calibration split")
        self.key(
            "taxonomy",
            "v1",
            choice("v1", "v2"),
            "v1: the taxonomy-v1 classifier on {split}.jsonl; v2: the claim map on "
            "{split}_v2.jsonl",
        )
        self.key(
            "classifier_model", None, optional(text), "taxonomy=v1: the model (default: judge)"
        )
        self.key("map_model", None, optional(text), "taxonomy=v2: the model (default: judge)")
        self.key(
            "prompt",
            "claimmap-v2.7",
            aliased(MAP_PROMPT_NAMES, choice(_prompt_choices)),
            "taxonomy=v2: the claim-map prompt version (plan 0029)",
        )
        anchor_key(self)
        self.key("repeats", 2, integer, "calls per case")
        self.key("cases", None, optional(text), "override the cases file")

    def execute(self) -> int:
        return cmd_calibrate(self)


class Report(Command):
    about = "cross-arm report + audit samples"

    def __init__(self) -> None:
        super().__init__()
        self.key("out", "outputs", text, "output root")
        self.key("judge_model", DEFAULT_MODEL, text, "the judge model of the run to report")
        self.key("no_rebuild", False, boolean, "read the existing summaries instead of rebuilding")
        self.key("view", "v1", choice("v1", "v2"), "v2: the claim-map report (plan 0002)")
        self.key(
            "audit_n", None, optional(integer), "audit samples per leaf; default 15 (v1) / 10 (v2)"
        )
        self.key("audit_verdict_n", 20, integer, "v2: audit samples per verdict")
        self.key("seed", 0, integer, "seed of the audit sample")
        self.key(
            "readouts_root",
            None,
            optional(text),
            "agentic_misalignment only: DIR with <arm>/agentic_misalignment.jsonl (default "
            f"{paths.READOUTS_ROOT}); ignored by the other evals",
        )
        eval_key(self)
        bank_key(self)

    def execute(self) -> int:
        return _cmd_report(self)


class ConsistencyV2(Command):
    about = "claim-map judge consistency: two fresh :M calls per sampled cell"

    def __init__(self) -> None:
        super().__init__()
        judge_keys(self)
        self.key("arms", None, optional(text), "arm:layer,... e.g. s3d-rl600:44,jlens:44")
        self.key(
            "rescore",
            False,
            boolean,
            "re-score the cached r0/r1 cells of this prompt version (no API calls)",
        )
        self.key("n", 50, integer, "cells per arm:layer")
        self.key("seed", 0, integer, "seed of the cell sample")
        self.key(
            "readouts_root", str(paths.READOUTS_ROOT), text, "DIR with <arm>/hallucination.jsonl"
        )
        self.key("summarizer_model", None, optional(text), "the summary model (default: judge)")
        self.key("classifier_model", None, optional(text), "the classifier model (default: judge)")
        self.key("map_model", None, optional(text), "the claim-map model (default: judge)")

    def execute(self) -> int:
        return cmd_consistency_v2(self)


class PilotDump(Command):
    about = "sample pilot spans as unlabelled calibration cases"
    stratum = (5, "spans per arm x grounding")

    def __init__(self) -> None:
        super().__init__()
        n, meaning = self.stratum
        self.key("out", "outputs", text, "output root of the run to sample")
        self.key("judge_model", DEFAULT_MODEL, text, "the judge model of that run")
        self.key("arms", pydra.REQUIRED, text, "comma-separated arm names")
        self.key("layers", None, optional(text), "comma-separated layers (default: all)")
        self.key("per_stratum", n, integer, meaning)
        self.key("seed", 0, integer, "seed of the sample")
        self.key("dest", pydra.REQUIRED, text, "the JSONL file to write")

    def execute(self) -> int:
        return cmd_pilot_dump(self)


class PilotDumpV2(PilotDump):
    about = "dump real cells with production units for v2 hand-labelling"
    stratum = (3, "cells per arm x v5c cell class")

    def execute(self) -> int:
        return cmd_pilot_dump_v2(self)


class UnitsV2(Command):
    about = "fill v2 cases' gold.units with the production unit split (leaf null)"

    def __init__(self) -> None:
        super().__init__()
        self.key("cases", pydra.REQUIRED, text, "the v2 cases file to read")
        self.key("dest", pydra.REQUIRED, text, "the JSONL file to write")

    def execute(self) -> int:
        return cmd_units_v2(self)


class Index(Command):
    about = (
        "the hallucination index (hindex-v8, plan 0029; the only version): one row per "
        "(method, layer)"
    )

    def __init__(self) -> None:
        super().__init__()
        self.key("out", "outputs", text, "output root of the default sources")
        self.key("judge_model", DEFAULT_MODEL, text, "the judge model of the default sources")
        for name, default in (
            ("main_root", "<out>/wsbench_chat/<model>"),
            ("story_root", "<out>/story/<model>"),
            ("am_root", "<out>/agentic_misalignment/<model>"),
            ("um_root", "<out>/sibling/um (country at label/, others at <domain>/label/)"),
        ):
            self.key(name, None, many, f"repeatable; replaces the default {default}")
        self.key(
            "am_situation_root",
            None,
            many,
            "repeatable; AM v8 root(s) whose arm dirs hold situation.jsonl "
            "(default: the am_root / <out>/agentic_misalignment/<model>)",
        )
        for name, var in (
            ("overrefusal_root", "HBENCH_SAFETY_OR_OUT"),
            ("benign_roleplay_root", "HBENCH_SAFETY_BR_OUT"),
        ):
            self.key(
                name,
                None,
                many,
                "repeatable; a dir holding index_counts.jsonl (+ index_meta.json); default: "
                f"${var} only when set (opt-in, plan 0019), else the family is absent",
            )
        self.key(
            "version",
            "v8",
            choice("v8"),
            "v8 (the only version): hindex-v8 (plan 0029: main 9, am_claims 3, am_situation 3, "
            "sibling 1, story 1, overrefusal 1, benign_roleplay 2 (no cheating: archived, plan "
            "0029 amendment 3); per-layer rows + a joint row; claimmap-v2.8+ roots, "
            "claimmap-v2.9 for the final run; needs a `refusal_claims` overrefusal_root "
            "(offtopic_index/, plan 0022) and a `prompts` benign_roleplay_root (`hbench "
            "benign-roleplay index-export`, plan 0027))",
        )
        self.key("methods", str(INDEX_METHODS), text, "method <-> arm mapping")
        self.key(
            "dest",
            None,
            optional(text),
            "where index.json / index.md go (default <out>/index/<model>/)",
        )
        self.key("workers", 8, integer, "processes over arm dirs")

    def execute(self) -> int:
        return cmd_index(self)


class Final(Command):
    about = (
        "plan 0029: check the v8 manifest (sha256s, completeness, one prompt version), build "
        "hindex-v8 from exactly its roots and draw the final plots"
    )

    def __init__(self) -> None:
        super().__init__()
        self.key(
            "manifest",
            str(INDEX_MANIFEST),
            text,
            "the frozen manifest to check and build from (default: the shipped manifest of the "
            "results of record)",
        )
        self.key("out", "outputs/final", text, "index.json / index.md destination")
        self.key("plots", "plots/final", text, "plot destination")
        self.key("methods", str(INDEX_METHODS), text, "method <-> arm mapping")
        self.key("workers", 8, integer, "processes over arm dirs")

    def execute(self) -> int:
        return cmd_final(self)


def cmd_final(args: Command) -> int:
    from hallucination_bench import final

    return final.main(
        Path(args.manifest),
        out=Path(args.out),
        plots=Path(args.plots),
        methods=Path(args.methods),
        workers=args.workers,
    )


# a name -> a command, or (for an eval's own CLI) a group: its one-line meaning and its commands
Group = tuple[str, dict[str, type[Command]]]
COMMANDS: dict[str, type[Command] | Group] = {
    "run": Run,
    "calibrate": Calibrate,
    "report": Report,
    "consistency-v2": ConsistencyV2,
    "pilot-dump": PilotDump,
    "pilot-dump-v2": PilotDumpV2,
    "units-v2": UnitsV2,
    "index": Index,
    "final": Final,
}


def commands() -> dict[str, type[Command] | Group]:
    """Every subcommand: the shared ones, then each ``EVAL_CLIS`` module's ``COMMANDS``."""
    out = dict(COMMANDS)
    for mod in EVAL_CLIS:
        out.update(importlib.import_module(mod).COMMANDS)
    return out


def _usage(prog: str, names: Any) -> str:
    return f"usage: {prog} <{'|'.join(names)}> [key=value ...] [--show | --help]"


def _listing(prog: str, table: dict[str, Any]) -> str:
    lines = [_usage(prog, table), ""]
    for name, entry in table.items():
        about = entry[0] if isinstance(entry, tuple) else entry.about
        first, *rest = textwrap.wrap(about, 76) or [""]
        lines.append(f"  {name:<16} {first}")
        lines += [f"  {'':<16} {x}" for x in rest]
    return "\n".join([*lines, "", f"{prog} <command> --help lists a command's keys."])


class _Help(Exception):
    """``--help`` was asked for; ``str()`` is the text to print."""


def _fail(prog: str, message: str, usage: str) -> SystemExit:
    print(f"{usage}\n{prog}: error: {message}", file=sys.stderr)
    return SystemExit(EXIT_USAGE)


def _find(argv: list[str]) -> tuple[Command, list[str]]:
    """The command ``argv`` names and the arguments left for it."""
    prog, table = "hbench", commands()
    while True:
        if argv and argv[0] in HELP_FLAGS:
            raise _Help(_listing(prog, table))
        if not argv or argv[0] not in table:
            what = f"unknown command {argv[0]!r}" if argv else "a command is required"
            raise _fail(prog, what, _usage(prog, table))
        prog, entry, argv = f"{prog} {argv[0]}", table[argv[0]], argv[1:]
        if not isinstance(entry, tuple):
            command = entry()
            command.prog = prog
            return command, argv
        table = entry[1]


def command_for(argv: list[str]) -> tuple[Command, bool]:
    """``argv`` as a finalized command, and whether ``--show`` was given. A command line that
    does not parse prints the reason and exits 2; ``--help`` prints the keys and exits 0."""
    try:
        command, rest = _find(list(argv))
        if any(a in HELP_FLAGS for a in rest):
            raise _Help(help_text(command))
    except _Help as h:
        print(h)
        raise SystemExit(0) from None
    try:
        return command, parse(command, rest)
    except UsageError as e:
        raise _fail(command.prog, str(e), usage(command)) from None


class _Parser:
    def parse_args(self, argv: list[str]) -> Command:
        return command_for(argv)[0]


def build_parser() -> _Parser:
    """For in-process callers: ``build_parser().parse_args(argv)`` is the finalized command."""
    return _Parser()


def main(argv: list[str] | None = None) -> int:
    command, show = command_for(list(sys.argv[1:] if argv is None else argv))
    if show:
        print(show_text(command))
        return 0
    try:
        return command.execute()
    except JudgeConfigError as e:
        print(f"JUDGE CONFIG ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
