"""``hbench try``: the two-question try-out. Picks a few items of one family, shows what the
model said at each, then runs the ordinary ``hbench run`` and ``hbench report`` on them."""

import contextlib
import io
import json
import re
from pathlib import Path

from hallucination_bench import REPO_ROOT
from hallucination_bench.command import Command, boolean, choice, integer, optional, text
from hallucination_bench.llm import CallResult

EXAMPLES = REPO_ROOT / "examples" / "readouts"
FAMILIES = ("wsbench_chat", "story")  # the families `hbench run --eval` judges with one driver
ARM = "try"
MAP_PROMPT = "claimmap-v2.9"  # the claim-map prompt of record (hindex-v8), not `run`'s default
LIVE_ATTEMPTS = 3  # a live try re-asks failed calls (exit 4) this many passes in all
TEXT_CHARS = 240
PROMPT_LINES = 12


class DryRunLLM:
    """Stands in for the judge client under ``--dry-run``: answers nothing and spends nothing,
    so every call is recorded as unanswered (re-asked by a later live run in the same dir)."""

    async def call(self, *, model: str, system: str, user: str, schema: dict) -> CallResult:
        return CallResult(None)

    async def preflight(self, model: str) -> None:
        return None

    async def aclose(self) -> None:
        return None


def example_readouts(family: str) -> Path:
    """The toy readouts file shipped for ``family``."""
    return EXAMPLES / f"{family}.jsonl"


def pick(family: str, readouts: Path, n: int) -> tuple[list[dict], list[dict]]:
    """``(items, rows)``: the first ``n`` bank items of ``family`` with readouts in the file
    (bank order; fewer when the file covers fewer) and those items' readout rows."""
    from hallucination_bench.evals import bank_for

    rows = []
    for line in readouts.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            row["id"] = str(row["id"])
        except (ValueError, KeyError, TypeError):
            continue
        rows.append(row)
    have = {r["id"] for r in rows}
    items = [it for it in bank_for(family).load_items() if it["id"] in have][:n]
    if not items:
        raise SystemExit(f"try: {readouts} has no readouts for any {family} item")
    keep = {it["id"] for it in items}
    return items, [r for r in rows if r["id"] in keep]


def _clip(text: str, limit: int = TEXT_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def describe(items: list[dict], rows: list[dict]) -> str:
    """One block per picked item: the user's question, the model's reply and what was read."""
    lines = []
    for k, it in enumerate(items, 1):
        mine = [r for r in rows if r["id"] == it["id"]]
        layers = sorted({r["layer"] for r in mine})
        lines += [
            f"[{k}] {it['id']}",
            f"    question: {_clip(it['prompt'])}",
            f"    model's reply: {_clip(it['response'])}",
            f"    expected: every claim a readout makes is supported by the conversation up to "
            f"the token it was read at ({len(it['sites'])} read sites in the bank)",
            f"    readouts in the file: {len(mine)} cells at layer(s) {layers}",
        ]
    return "\n".join(lines)


def _first_prompt(path: Path) -> str:
    """The first prompt block of ``prompts_sample.txt`` (the file ``--show-prompts`` writes)."""
    if not path.is_file():
        return ""
    blocks = path.read_text(encoding="utf-8").split("=====")
    return ("=====" + blocks[1]).rstrip() if len(blocks) > 1 else ""


def _first_table(path: Path) -> str:
    """The first table of the report (per-arm rates), with its heading."""
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, x in enumerate(lines) if x.startswith("|")), None)
    if start is None:
        return ""
    end = next((i for i in range(start, len(lines)) if not lines[i].startswith("|")), len(lines))
    head = next((x for x in reversed(lines[:start]) if x.startswith("#")), "")
    return "\n".join([head.lstrip("# "), *lines[start:end]])


def _build_index(judged: Path, judge_model: str) -> Path | None:
    """``hbench index`` on the try-out's output (``report`` no longer builds it): the index of
    the one family that was judged, which is also what the results viewer reads. Its long notes
    are not shown; a failure is one line, never the try-out's exit code."""
    from hallucination_bench import cli

    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            code = cli.main(["index", f"out={judged}", f"judge_model={judge_model}"])
    except (SystemExit, Exception) as e:  # the index must never fail the try-out
        code, note = 1, f"{type(e).__name__}: {e}"
    else:
        note = (log.getvalue().strip().splitlines() or ["no output"])[-1]
    path = judged / "index" / judge_model / "index.md"
    if code or not path.is_file():
        print(f"try: hallucination index not built ({note[:200]})")
        return None
    return path


def cmd_try(args: Command) -> int:
    from hallucination_bench import cli

    toy = args.readouts is None
    readouts = example_readouts(args.family) if toy else Path(args.readouts)
    if not readouts.is_file():
        raise SystemExit(f"try: no readouts file {readouts}")
    if args.n < 1:
        raise SystemExit("try: -n must be >= 1")
    items, rows = pick(args.family, readouts, args.n)
    out = Path(args.out or Path("outputs") / "try" / args.family)
    picked = out / "readouts" / ARM / cli.READOUTS_FILE
    picked.parent.mkdir(parents=True, exist_ok=True)
    picked.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    judged = out / "judged"
    failed = 0  # 4 when a live run left failed calls in the cache

    print(f"== hbench try: {args.family}, {len(items)} item(s), readouts {readouts}")
    if len(items) < args.n:
        print(f"   (the file has readouts for {len(items)} item(s) only, asked for {args.n})")
    print(describe(items, rows))

    mode = "dry run: prompts are built, no call is made" if args.dry_run else "live judge"
    print(f"\n== judge ({mode})")
    run_args = cli.Run.of(
        out=str(judged),
        arm=[f"{ARM}={picked}"],
        eval=args.family,
        allow_missing=True,
        show_prompts=1,
        judge_model=args.judge_model,
        map_prompt=MAP_PROMPT,
    )
    if args.dry_run:
        # the ordinary run with a client that answers nothing: every call is "failed" by design,
        # so its progress lines (failed=..., "failed calls remain") would read like a broken run
        log = io.StringIO()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            code = cli.cmd_run(run_args, llm=DryRunLLM())
        if code not in (0, 4):
            print(log.getvalue(), end="")
            return code
        cells = re.search(r"cells_to_judge=(\d+)", log.getvalue())
        print(
            f"{cells.group(1) if cells else '?'} cell(s) would be judged with {MAP_PROMPT}: "
            "prompts built, 0 calls made, nothing spent"
        )
    else:
        if toy:
            print(
                "(the example file holds a few cells of the bank: `missing=` below counts the "
                "rest and is expected)"
            )
        code = cli.cmd_run(run_args)
        for attempt in range(2, LIVE_ATTEMPTS + 1):
            if code != 4:  # 4 = some answers stayed invalid after the in-run retries
                break
            print(
                f"\ntry: asking only the failed calls again (attempt {attempt} of {LIVE_ATTEMPTS})"
            )
            code = cli.cmd_run(run_args)
        if code == 4:  # still failed: the rest is usable, so show the report and exit 4
            failed = 4
            print("try: run the same command again; it re-asks only the failed calls")
        elif code:
            return code
    from hallucination_bench.evals.wsbench_chat.run import arm_dir

    arm = arm_dir(judged, cli._models(run_args), ARM)
    if args.dry_run and (first := _first_prompt(arm / "prompts_sample.txt")):
        lines = first.splitlines()
        more = len(lines) - PROMPT_LINES
        print(f"\n== the first judge prompt (first {PROMPT_LINES} lines)")
        print("\n".join(_clip(x) for x in lines[:PROMPT_LINES]))
        if more > 0:
            print(f"[... {more} more lines in {arm / 'prompts_sample.txt'}]")

    print("\n== report" + (" (empty in a dry run: nothing was judged)" if args.dry_run else ""))
    code = cli.main(
        ["report", f"out={judged}", f"eval={args.family}", f"judge_model={args.judge_model}"]
    )
    print(_first_table(arm.parent / "report.md"))
    index_md = None if args.dry_run else _build_index(judged, args.judge_model)
    print("\n== outputs")
    for label, path in (
        ("readouts that were judged (the picked items' rows)", picked),
        ("judge prompts of the first cell", arm / "prompts_sample.txt"),
        ("judge cache (one row per call; resumable)", arm / "cache.jsonl"),
        ("per-cell verdicts", arm / "cells.jsonl"),
        ("arm summary (rates, health, spend)", arm / "summary.json"),
        ("report", arm.parent / "report.md"),
        *([("hallucination index (one family of seven)", index_md)] if index_md else []),
    ):
        print(f"  {label}: {path}" + ("" if path.exists() else "  (not written)"))
    return code or failed


class Try(Command):
    about = (
        "the two-question try-out: 2 items of one family through run -> report (--dry_run needs "
        "no API access)"
    )

    def __init__(self) -> None:
        super().__init__()
        self.key("family", "wsbench_chat", choice(*FAMILIES), "the family to try")
        self.key("n", 2, integer, "how many items")
        self.key("readouts", None, optional(text), "default: examples/readouts/<family>.jsonl")
        self.key("out", None, optional(text), "default: outputs/try/<family>")
        self.key("dry_run", False, boolean, "build the prompts, make no call")
        self.key("judge_model", "claude-opus-5-5", text, "the judge model")

    def execute(self) -> int:
        return cmd_try(self)


COMMANDS = {"try": Try}
