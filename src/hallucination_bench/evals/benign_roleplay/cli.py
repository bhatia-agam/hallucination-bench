"""``hbench benign-roleplay import|run|report`` (plan 0023), registered through
``cli.EVAL_CLIS``; each subcommand is a pydra config (``command.Command``). Exit codes: 0 ok,
2 pin / layout / refused input, else ``wsbench``'s code."""

import json
import sys
from pathlib import Path

import pydra

from hallucination_bench import paths
from hallucination_bench.command import Command, boolean, choice, integer, many, optional, text

from . import bank, metrics, wsb

# the jailbreak family's registered judge pin (WSB evals/jailbreak_recognition/judge_smoke_v2.json,
# plan 0037 J8.5 / Q5): HB passes no judge_model=, so this only names the output dir and is
# checked against every input's config.judge_model
DEFAULT_JUDGE = "claude-sonnet-5"


def _layers(s: str | None) -> list[int] | None:
    return [int(x) for x in s.split(",") if x.strip()] if s else None


def _arms(s: str) -> list[str]:
    arms = [a for a in s.split(",") if a]
    bad = set(arms) - set(wsb.KNOWN_ARMS)
    if bad:
        raise wsb.WsbError(f"unknown arms {sorted(bad)}; known {list(wsb.KNOWN_ARMS)}")
    return arms


def cmd_import(args: Command) -> int:
    root = Path(args.wsb)
    wsb.check_pin(root, args.wsb_sha)
    bp = wsb.bank_path(root)
    rel = str(bp.resolve().relative_to(root.resolve()))
    build_p = root / bank.BUILD_REL
    items_doc, hb_doc, gt = bank.build(
        json.loads(bp.read_text(encoding="utf-8")),
        json.loads(build_p.read_text(encoding="utf-8")),
        wsb_sha=args.wsb_sha,
        bank_rel=rel,
        bank_sha256=bank.sha256_bytes(bp.read_bytes()),
        build_sha256=bank.sha256_bytes(build_p.read_bytes()),
        n_target=args.n_target,
    )
    bank.write(items_doc, hb_doc, gt, Path(args.dest))
    print(
        f"[import] {items_doc['n_items']} items ({items_doc['n_shared']} shared, "
        f"{items_doc['n_hb_only']} hb_only), kind mix {items_doc['kind_mix']} -> {args.dest}"
    )
    return 0


def _out(args: Command) -> Path:
    return Path(args.out).resolve() / "benign_roleplay" / args.judge


def cmd_run(args: Command) -> int:
    calls = wsb.plan_calls(
        campaign=Path(args.campaign),
        readouts=Path(args.readouts_root),
        out=_out(args),
        layers=_layers(args.layers),
        arms=_arms(args.arms),
        nested=args.layout == "nested",
    )
    return wsb.run_calls(Path(args.wsb), calls, dry_run=args.dry_run)


def cmd_report(args: Command) -> int:
    out = _out(args)
    floor_res = metrics.read_result(Path(args.floor)) if args.floor else None
    s = metrics.summarize(
        bank.load_items(),
        out=out,
        campaign=Path(args.campaign),
        arms=_arms(args.arms),
        layers=_layers(args.layers),
        floor_res=floor_res,
        accept_refused_as_flagged=args.accept_refused_as_flagged,
        nested=args.layout == "nested",
    )
    for arm, a in s["rates"].items():
        if a["instrument"].get("judge_model") != args.judge:
            raise wsb.WsbError(
                f"{arm}: inputs judged by {a['instrument'].get('judge_model')}, not {args.judge}"
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(metrics.render_md(s), encoding="utf-8")
    print(metrics.render_md(s), end="")
    print(f"wrote {out / 'summary.json'} and {out / 'summary.md'}")
    return 0


def cmd_index_export(args: Command) -> int:
    from . import index_export

    pairs = [kv for kv in args.arm_map.split(",") if kv]
    if bad := [kv for kv in pairs if kv.count("=") != 1 or not all(kv.split("="))]:
        raise wsb.WsbError(f"--arm-map entries must be from=to: {bad}")
    arm_map = dict(kv.split("=") for kv in pairs)
    meta = index_export.export(
        [Path(p) for p in args.summary],
        Path(args.dest),
        bank.load_items(),
        arm_map=arm_map,
        joint=args.joint,
    )
    print(
        f"[index-export] {meta['n_rows']} rows, arms {meta['arms']}, layers {meta['layers']} "
        f"-> {args.dest}"
    )
    return 0


def _guarded(fn, args: Command) -> int:  # type: ignore[no-untyped-def]
    """Run ``fn``: a refused pin, layout or input is exit 2 with one line."""
    try:
        return int(fn(args))
    except (wsb.WsbError, ValueError) as e:  # ValueError: bank.build's fail-closed gate
        print(f"benign-roleplay: {e}", file=sys.stderr)
        return 2


def unit_keys(c: Command) -> None:
    """Which judged results a command reads or writes."""
    c.key("out", "outputs", text, "output root")
    c.key("judge", DEFAULT_JUDGE, text, "the family pin (output dir name)")
    c.key("campaign", str(paths.BR_CAMPAIGN), text, "campaign judged dir")
    c.key("arms", ",".join(wsb.ARMS), text, "comma-separated arm names")
    c.key("layers", None, optional(text), "csv subset of readplan.GRID (C3)")
    c.key(
        "layout",
        "flat",
        choice("flat", "nested"),
        "flat: plan 0023's campaign dirs; nested: every arm's shared ids at "
        "<campaign>/<arm>/{all,L<k>} (plan 0027, the index arms)",
    )


class Import(Command):
    about = "copy the gated controls from a pinned WSB checkout"

    def __init__(self) -> None:
        super().__init__()
        self.key("wsb", str(paths.WSBENCH_PUB), text, "the WorkspaceBench checkout")
        self.key("wsb_sha", wsb.WSB_SHA, text, "the commit that checkout must be at")
        self.key("n_target", bank.N_TARGET, integer, "items to keep")
        self.key("dest", str(bank.DATA_DIR), text, "the data dir to write")

    def execute(self) -> int:
        return _guarded(cmd_import, self)


class Run(Command):
    about = "judge via the pinned `wsbench run` (no GPU)"

    def __init__(self) -> None:
        super().__init__()
        unit_keys(self)
        self.key("wsb", str(paths.WSBENCH_PUB), text, "the WorkspaceBench checkout")
        self.key("readouts_root", str(paths.BR_READOUTS), text, "DIR with the arms' readouts")
        self.key("dry_run", False, boolean, "print the wsbench calls only")

    def execute(self) -> int:
        return _guarded(cmd_run, self)


class Report(Command):
    about = "summary.{json,md} from the judged results"

    def __init__(self) -> None:
        super().__init__()
        unit_keys(self)
        self.key("floor", None, optional(text), "po-opus-v1 jailbreak_recognition results.json")
        self.key(
            "accept_refused_as_flagged",
            False,
            boolean,
            "admit WSB results incomplete ONLY through judge-refused (unjudged) cells; those "
            "items count as flagged (conservative) and the count is shown beside the rate",
        )

    def execute(self) -> int:
        return _guarded(cmd_report, self)


class IndexExport(Command):
    about = "plan 0027: summary.json -> index_counts.jsonl + index_meta.json (hindex-v7 root)"

    def __init__(self) -> None:
        super().__init__()
        self.key("summary", pydra.REQUIRED, many, "repeatable summary.json")
        self.key("dest", pydra.REQUIRED, text, "the benign_roleplay_root to write")
        self.key(
            "arm_map",
            "",
            text,
            "from=to,... : rename an eval arm to an index reader name (no-op when they match)",
        )
        self.key(
            "joint",
            False,
            boolean,
            "plan 0029: also write the eval's `all` unit to index_counts_joint.jsonl (the "
            "hindex-v8 joint rows; layer 'all')",
        )

    def execute(self) -> int:
        return _guarded(cmd_index_export, self)


COMMANDS = {
    "benign-roleplay": (
        "plan 0023: false jailbreak recognition (HB eval)",
        {"import": Import, "run": Run, "report": Report, "index-export": IndexExport},
    )
}
