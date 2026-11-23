"""``hbench overrefusal candidates|judge-gt|bank|run|report|...|offtopic-export`` (plan 0015;
plan 0022: ``offtopic-export``), registered through
``hallucination_bench.cli.EVAL_CLIS`` (plan 0012 D2). Each subcommand is a pydra config
(``command.Command``)."""

import json
import os
import sys
from pathlib import Path

import pydra

from hallucination_bench.command import Command, boolean, choice, integer, number, optional, text

DEFAULT_MODEL = "claude-opus-5-5"


def _readouts_root(args: Command) -> Path:
    return Path(
        args.readouts_root
        or os.environ.get("HBENCH_OR_READOUTS")
        or "/workspace/hb-readouts-overrefusal-v2"  # rev 7: old + v2add readouts
    )


def cmd_candidates(args: Command) -> int:
    from . import data

    print(json.dumps(data.build_candidates(args.n_hard, args.n_toxic, args.seed), indent=1))
    return 0


def cmd_judge_gt(args: Command) -> int:
    from . import run

    out = Path(args.out)
    res = run.run_gt(
        out,
        run.Judge(model=args.judge_model),
        samples_path=Path(args.samples) if args.samples else out / "overrefusal" / "samples.jsonl",
        greedy_path=Path(args.greedy)
        if args.greedy
        else out / "overrefusal" / "samples_greedy.jsonl",
        candidates_path=Path(args.candidates) if args.candidates else None,
        labels_out=Path(args.labels_out) if args.labels_out else None,
        rpm=args.rpm,
        concurrency=args.concurrency,
        limit=args.limit,
        write=not args.limit,
    )
    print(json.dumps(res, indent=1))
    return 4 if res["n_failed"] else 0


def cmd_bank(args: Command) -> int:
    from transformers import AutoTokenizer

    from . import data

    gt = data.read_jsonl(data.GT_LABELS)
    pop, eligible = data.select_population(
        gt, n_never=args.n_never, n_hard_refused=args.n_hard_refused, n_always=args.n_always
    )
    tok = AutoTokenizer.from_pretrained(data.SUBJECT)
    bank = data.build_bank(pop, data.load_candidates(), tok)
    meta = {
        "eligible": eligible,
        "selected": {k: len(v) for k, v in pop.items()},
        "n_sites": sum(len(it["sites"]) for it in bank["items"]),
        "subject": data.SUBJECT,
    }
    data.ITEMS.write_text(
        json.dumps({"meta": meta, "items": bank["items"]}, ensure_ascii=False, indent=1) + "\n"
    )
    data.CAPTURE_ROWS.write_text(json.dumps(bank["capture_rows"], ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


def cmd_run(args: Command) -> int:
    from hallucination_bench.llm import JudgeConfigError

    from . import run

    stages = [s for s in args.stages.split(",") if s]
    bad = set(stages) - set(run.STAGES)
    if bad:
        raise SystemExit(f"unknown stages {sorted(bad)}; known {run.STAGES}")
    arms = run.resolve_arms(_readouts_root(args), args.arms.split(",") if args.arms else None)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    try:
        res = run.run(
            arms,
            Path(args.out),
            run.Judge(model=args.judge_model),
            stages=stages,
            layers=layers,
            limit=args.limit,
            rpm=args.rpm,
            concurrency=args.concurrency,
        )
    except JudgeConfigError as e:
        print(f"JUDGE CONFIG ERROR: {e}", file=sys.stderr)
        return 3
    nf = 0
    for name, s in res.items():
        print(
            f"[{name}] cells={s['n_cells']} status={s['status']} "
            f"refusal_cand={s['n_refusal_candidates']} rejected={s['n_rejected']} "
            f"unverified={s['n_unverified']} api_refused={s['n_api_refused']} "
            f"spend=${s['spend']['usd']:.2f}",
            flush=True,
        )
        nf += run.n_failed(s) if not args.limit else 0
    return 4 if nf else 0


def cmd_report(args: Command) -> int:
    from . import report

    root = report.report(
        Path(args.out),
        args.judge_model,
        _readouts_root(args),
        args.arms.split(",") if args.arms else None,
    )
    print(f"wrote {root / 'report.md'}, {root / 'report.json'}, {root / 'examples.jsonl'}")
    hb = json.loads((root / "report.json").read_text()).get("hb_scores")
    if hb:
        print("HB score (never_refused, tail): probe_validated [headline] / raw, % readouts")
        for arm, by in hb["arms"].items():
            cells = [
                f"L{la} {100 * d['tail']['probe_validated']['mean']:.1f} / "
                f"{100 * d['tail']['raw']['mean']:.1f}"
                for la, d in by.items()
            ]
            print(f"  {arm}: " + "  ".join(cells))
    return 0


def cmd_probe_data(args: Command) -> int:
    from . import probe_data

    if args.step == "80k":
        res: object = probe_data.build_80k_candidates()
    elif args.step == "calib":
        res = {"n_calib_texts": probe_data.build_calib_texts()}
    else:
        res = probe_data.build_all()
    print(json.dumps(res, indent=1, default=str))
    return 0


def cmd_probe_report(args: Command) -> int:
    from . import probe_report

    arms = args.arms.split(",") if args.arms else ["s3d-rl600", "mlnla-rl600-all11", "jlens"]
    root = probe_report.report(Path(args.out), args.judge_model, arms)
    print(f"wrote {root / 'probe_report.md'}, {root / 'probe_report.json'}")
    return 0


def cmd_grow(args: Command) -> int:
    from . import grow

    if args.step == "candidates":
        res = grow.build_80k2_candidates()
    elif args.step == "pairs":
        from transformers import AutoTokenizer

        from . import data

        res = grow.grow_pair_controls(AutoTokenizer.from_pretrained(data.SUBJECT))
    else:
        from transformers import AutoTokenizer

        from . import data

        res = grow.grow_bank(AutoTokenizer.from_pretrained(data.SUBJECT))
    print(json.dumps(res, indent=1, default=str))
    return 0


def cmd_freeze(args: Command) -> int:
    from . import data

    doc = json.loads(data.ITEMS.read_text())
    doc["meta"]["frozen"] = args.date
    doc["meta"]["hb_score"] = (
        "verified refusal_intent readout rate on never_refused (tail sites headline, all sites "
        "reported); hard_refused / always_refused are diagnostics only"
    )
    data.ITEMS.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {data.write_manifest()}")
    return 0


def cmd_wsb_export(args: Command) -> int:
    from . import wsb_export

    res = wsb_export.export(Path(args.out), args.judge_model, Path(args.family_dir))
    print(json.dumps(res, indent=1))
    return 0


def cmd_offtopic_export(args: Command) -> int:
    from . import offtopic_export

    if getattr(args, "joint", False):  # plan 0029: the joint rows beside the per-layer ones
        res = offtopic_export.export_joint(Path(args.out), args.judge_model)
    else:
        res = offtopic_export.export(Path(args.out), args.judge_model)
    print(json.dumps(res, indent=1))
    return 0


def judge_keys(c: Command) -> None:
    c.key("out", "outputs", text, "output root")
    c.key("judge_model", DEFAULT_MODEL, text, "the judge model; it also names the output dir")
    c.key("rpm", 1000.0, number, "requests per minute")
    c.key("concurrency", 96, integer, "judge calls in flight at once")
    c.key("limit", None, optional(integer), "smoke: at most N calls / cells")


def report_keys(c: Command, judge_model: str = DEFAULT_MODEL, out: str = "output root") -> None:
    c.key("out", "outputs", text, out)
    c.key("judge_model", judge_model, text, "the judge model whose outputs are read")


class Candidates(Command):
    about = "Stage 0: pinned OR-Bench -> data/overrefusal/candidates.jsonl"

    def __init__(self) -> None:
        super().__init__()
        self.key("n_hard", 400, integer, "candidates from the hard config")
        self.key("n_toxic", 150, integer, "candidates from the toxic config")
        self.key("seed", 0, integer, "seed of the draw")

    def execute(self) -> int:
        return cmd_candidates(self)


class JudgeGt(Command):
    about = "Stage 2: refusal labels (orgt-v1) -> gt_labels.jsonl"

    def __init__(self) -> None:
        super().__init__()
        judge_keys(self)
        self.key("candidates", "", text, "rev 5: another candidates file (80k sample)")
        self.key("samples", "", text, "rev 5: its samples file")
        self.key("labels_out", "", text, "rev 5: where its labels go")
        self.key("greedy", "", text, "rev 7: its greedy samples file")

    def execute(self) -> int:
        return cmd_judge_gt(self)


class Bank(Command):
    about = "Stage 3: populations + items.json + capture_rows.json"

    def __init__(self) -> None:
        super().__init__()
        self.key("n_never", 150, integer, "never_refused prompts to keep")
        self.key("n_hard_refused", 50, integer, "hard_refused prompts to keep")
        self.key("n_always", 50, integer, "always_refused prompts to keep")

    def execute(self) -> int:
        return cmd_bank(self)


class Run(Command):
    about = "Stages 5-6: detector + verifier"

    def __init__(self) -> None:
        super().__init__()
        judge_keys(self)
        self.key("readouts_root", None, optional(text), "default $HBENCH_OR_READOUTS")
        self.key("arms", "", text, "comma-separated arm names (default: every arm in the root)")
        self.key("layers", "", text, "comma-separated layers (default: all)")
        self.key("stages", "detect,verify", text, "comma-separated stages to run")

    def execute(self) -> int:
        return cmd_run(self)


class Grow(Command):
    about = "rev 7: 80k2 candidates | append new items to the bank"
    positional = "step"

    def __init__(self) -> None:
        super().__init__()
        self.key("step", pydra.REQUIRED, choice("candidates", "bank", "pairs"), "which step")

    def execute(self) -> int:
        return cmd_grow(self)


class WsbExport(Command):
    about = "rev 8: WorkspaceBench refusal-prediction family"

    def __init__(self) -> None:
        super().__init__()
        self.key("family_dir", pydra.REQUIRED, text, "the WorkspaceBench family dir to write")
        report_keys(self, "claude-sonnet-5-5")

    def execute(self) -> int:
        return cmd_wsb_export(self)


class OfftopicExport(Command):
    about = "plan 0022: wrong-topic refusal claims -> offtopic_index/ (hindex-v6 overrefusal)"

    def __init__(self) -> None:
        super().__init__()
        report_keys(self, out="reads <out>/overrefusal/<model>/")
        self.key(
            "joint",
            False,
            boolean,
            "plan 0029: sj_records_joint.jsonl (sj_run.py --joint) -> "
            "offtopic_index/index_counts_joint.jsonl (the hindex-v8 joint overrefusal rows)",
        )

    def execute(self) -> int:
        return cmd_offtopic_export(self)


class Freeze(Command):
    about = "rev 8: mark items.json frozen + MANIFEST.sha256"

    def __init__(self) -> None:
        super().__init__()
        self.key("date", "2026-10-28", text, "the freeze date written to items.json")

    def execute(self) -> int:
        return cmd_freeze(self)


class ProbeData(Command):
    about = "rev 5: probe training data (80k sample | texts)"
    positional = "step"

    def __init__(self) -> None:
        super().__init__()
        self.key("step", pydra.REQUIRED, choice("80k", "texts", "calib"), "which step")

    def execute(self) -> int:
        return cmd_probe_data(self)


class ProbeReport(Command):
    about = "rev 5: probe validity, premise, tagging, misses"

    def __init__(self) -> None:
        super().__init__()
        report_keys(self, "claude-sonnet-5-5")
        self.key("arms", "", text, "comma-separated arm names (default: the three paper arms)")

    def execute(self) -> int:
        return cmd_probe_report(self)


class Report(Command):
    about = "Stage 7: report.md / report.json / examples.jsonl"

    def __init__(self) -> None:
        super().__init__()
        report_keys(self)
        self.key("readouts_root", None, optional(text), "default $HBENCH_OR_READOUTS")
        self.key("arms", "", text, "comma-separated arm names (default: every arm in the root)")

    def execute(self) -> int:
        return cmd_report(self)


COMMANDS = {
    "overrefusal": (
        "plan 0015: OR-Bench over-refusal eval",
        {
            "candidates": Candidates,
            "judge-gt": JudgeGt,
            "bank": Bank,
            "run": Run,
            "grow": Grow,
            "wsb-export": WsbExport,
            "offtopic-export": OfftopicExport,
            "freeze": Freeze,
            "probe-data": ProbeData,
            "probe-report": ProbeReport,
            "report": Report,
        },
    )
}
