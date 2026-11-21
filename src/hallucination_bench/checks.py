"""``hbench check``: a registry of named gates and sanity checks. Each entry is a name, one line
on what it guards and the dotted path of a function returning ``(ok, why)``."""

import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hallucination_bench import DATA, REPO_ROOT
from hallucination_bench.command import Command, many, text

Outcome = tuple[bool, str]
PLUGIN_ENV = "HBENCH_CHECKS"  # name=package.module:function[,name2=...]: checks from outside


@dataclass(frozen=True)
class Check:
    """``target`` is ``package.module:function``, imported only when the check runs. The
    function takes its options as keyword arguments (strings) and returns ``(ok, why)``."""

    name: str
    guards: str
    target: str

    def function(self) -> Callable[..., Outcome]:
        module, _, attr = self.target.partition(":")
        return getattr(importlib.import_module(module), attr)

    def options(self) -> dict[str, bool]:
        """``{option: required}`` read off the function's keyword arguments."""
        params = inspect.signature(self.function()).parameters.values()
        named = [p for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        return {p.name: p.default is inspect.Parameter.empty for p in named}


REGISTRY: dict[str, Check] = {}


def register_check(name: str, guards: str, target: str) -> Check:
    """Add a check. ``target`` is the dotted path ``package.module:function``."""
    if ":" not in target:
        raise ValueError(f"check {name!r}: target must be package.module:function, got {target!r}")
    REGISTRY[name] = Check(name, guards, target)
    return REGISTRY[name]


def load_plugins(env: dict[str, str] | None = None) -> None:
    """Register the checks named in ``$HBENCH_CHECKS`` (``name=package.module:function``, comma
    separated). The one-line description is the function's docstring."""
    for pair in (os.environ if env is None else env).get(PLUGIN_ENV, "").split(","):
        if not pair.strip():
            continue
        name, sep, target = pair.strip().partition("=")
        if not sep:
            raise ValueError(f"{PLUGIN_ENV}: expected name=package.module:function, got {pair!r}")
        fn = register_check(name, "", target).function()
        doc = (inspect.getdoc(fn) or "external check").split("\n\n")[0]
        register_check(name, " ".join(doc.split()), target)


def all_checks() -> dict[str, Check]:
    """Built-in checks plus the ``$HBENCH_CHECKS`` plug-ins, in registration order."""
    load_plugins()
    return dict(REGISTRY)


def run(name: str, **options: str) -> Outcome:
    """Run one check by name. An unknown name, an option the check does not take or a missing
    required option is a ``ValueError``; an exception inside the check is a FAIL with its text."""
    checks = all_checks()
    if name not in checks:
        raise ValueError(f"unknown check {name!r}; known: {', '.join(checks)}")
    check = checks[name]
    takes = check.options()
    if extra := sorted(set(options) - set(takes)):
        raise ValueError(f"{name} does not take {extra}; it takes {sorted(takes) or 'no options'}")
    if missing := sorted(k for k, required in takes.items() if required and k not in options):
        raise ValueError(f"{name} needs {', '.join(k + '=' for k in missing)}")
    try:
        ok, why = check.function()(**options)
    except SystemExit as e:  # the guards fail closed with SystemExit(reason)
        return False, str(e.code)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return bool(ok), " ".join(str(why).split())


def offline() -> list[str]:
    """The checks ``hbench check all`` runs: those with no required option."""
    return [n for n, c in all_checks().items() if not any(c.options().values())]


def needing_options() -> dict[str, list[str]]:
    """The checks ``check all`` cannot run, each with the options it requires."""
    out = {}
    for name, check in all_checks().items():
        if required := [k for k, needed in check.options().items() if needed]:
            out[name] = required
    return out


def pytest_outcome(*paths: str) -> Outcome:
    """Run test files of this checkout as a check (the tests already hold the assertions). A
    test that skipped itself did not check anything, so the line names each skip and its reason."""
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-rs", "-p", "no:cacheprovider", *paths]
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    lines = [x for x in p.stdout.strip().splitlines() if x.strip()] or [p.stderr.strip()[-200:]]
    if p.returncode == 0:
        skips = [x.split("] ", 1)[-1].strip() for x in lines if x.startswith("SKIPPED")]
        note = f"; NOT checked (skipped): {'; '.join(skips)}" if skips else ""
        return True, f"{lines[-1].strip()} ({', '.join(paths)}){note}"
    failed = next((x for x in lines if x.startswith(("FAILED", "ERROR"))), lines[-1])
    return False, failed.strip()


# ------------------------------------------------------------------------------ built-in checks


def _manifest(path: Path) -> dict[str, str]:
    """``{file name: sha256}`` of a ``MANIFEST.sha256`` (``<digest>  <name>`` lines)."""
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {name: digest for digest, name in (x.split("  ", 1) for x in lines)}


def bank_manifests() -> Outcome:
    manifests = sorted(DATA.glob("*/MANIFEST.sha256"))
    n = 0
    for manifest in manifests:
        for name, digest in _manifest(manifest).items():
            file = manifest.parent / name
            if not file.is_file():
                return False, f"{manifest.parent.name}/{name} is listed but missing"
            if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                return False, f"{manifest.parent.name}/{name} differs from its MANIFEST.sha256"
            n += 1
    sets = ", ".join(m.parent.name for m in manifests)
    return bool(n), f"{n} frozen files match their MANIFEST.sha256 ({sets})"


def index_manifest() -> Outcome:
    from hallucination_bench.judges import claimmap_v2 as cm
    from hallucination_bench.judges import continuation_verifier as cv

    path = DATA / "index" / "v8_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    prov = manifest["provenance"]
    bank = _manifest(DATA / "wsbench_chat_x3" / "MANIFEST.sha256")["items.json"]
    if prov["bank"]["sha256"] != bank:
        return False, "the index manifest was scored on another bank than data/wsbench_chat_x3"
    if prov["versions"]["claim_map"] not in cm.VERSIONS:
        return False, f"claim-map prompt {prov['versions']['claim_map']} is not in the registry"
    if manifest["excuse_verifier"] not in cv.VERSIONS:
        return False, f"excuse verifier {manifest['excuse_verifier']} is not in the registry"
    return True, (
        f"{manifest['index_version']}: scored on the frozen x3 bank, with "
        f"{prov['versions']['claim_map']} and {manifest['excuse_verifier']} both registered"
    )


CALIBRATION_SETS = ("dev.jsonl", "heldout.jsonl", "dev_v2.jsonl", "heldout_v2_synthetic.jsonl")


def calibration_sets() -> Outcome:
    counts = []
    for name in CALIBRATION_SETS:
        path = DATA / "calibration" / name
        if not path.is_file():
            return False, f"data/calibration/{name} is missing"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows:
            return False, f"data/calibration/{name} is empty"
        counts.append(f"{name} {len(rows)}")
    return True, "calibration cases present and parse: " + ", ".join(counts)


def judge_prompts() -> Outcome:
    return pytest_outcome("tests/test_registry_pins.py", "tests/test_v5c_frozen.py")


def readouts_contract(readouts: str) -> Outcome:
    from hallucination_bench.readouts import load_readouts

    path = Path(readouts)
    if not path.is_file():
        return False, f"no such file: {path}"
    cells, rep = load_readouts(path)
    bad = {k: v for k, v in rep.skipped.items() if v}
    what = f"{len(cells)} cells of {len({c.id for c in cells})} items, kind {rep.kind}"
    if bad:
        return False, f"{what}; rows not read: {json.dumps(bad)}"
    return bool(cells), what if cells else "no readout rows"


def dry_run(family: str = "") -> Outcome:
    from hallucination_bench import cli, tryout

    families = [family] if family else list(tryout.FAMILIES)
    with tempfile.TemporaryDirectory(prefix="hbench-check-") as tmp:
        for fam in families:
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                code = cli.main(["try", f"family={fam}", "--dry_run", f"out={tmp}/{fam}"])
            if code:
                return False, f"{fam}: exit {code}: {out.getvalue().strip().splitlines()[-1][:160]}"
    return True, f"{', '.join(families)}: prompts built from examples/readouts, no call"


_HERE = "hallucination_bench.checks"
register_check(
    "bank_manifests",
    "Every frozen data set is byte-identical to its MANIFEST.sha256.",
    f"{_HERE}:bank_manifests",
)
register_check(
    "index_manifest",
    "The frozen index manifest names the shipped x3 bank and registered judge versions.",
    f"{_HERE}:index_manifest",
)
register_check(
    "calibration_sets",
    "The calibration case files are present and parse.",
    f"{_HERE}:calibration_sets",
)
register_check(
    "judge_prompts",
    "The frozen judge prompts and schemas still hash to their pinned values.",
    f"{_HERE}:judge_prompts",
)
register_check(
    "readouts_contract",
    "A readouts file parses under the contract (readouts=FILE).",
    f"{_HERE}:readouts_contract",
)
if importlib.util.find_spec("hallucination_bench.tryout"):  # the check wraps `hbench try`
    register_check(
        "dry_run",
        "The pipeline builds its judge prompts end to end on the example readouts, with no call.",
        f"{_HERE}:dry_run",
    )


# ------------------------------------------------------------------------------------- the CLI


def cmd_check(args: Command) -> int:
    pairs = args.option or []
    if bad := [p for p in pairs if "=" not in p]:
        raise SystemExit(f"option= expects KEY=VALUE, got {bad}")
    options = dict(p.split("=", 1) for p in pairs)
    options |= {k: v for k in ("family", "readouts") if (v := getattr(args, k))}
    try:
        known = all_checks()
        if not args.name:
            for check in known.values():
                takes = "".join(
                    f" {k}=" if required else f" [{k}=]" for k, required in check.options().items()
                )
                print(f"{check.name}{takes}\n    {check.guards}")
            print("\nrun one: hbench check <name> [key=value ...]   all offline: hbench check all")
            return 0
        names = offline() if args.name == "all" else [args.name]
        failed = 0
        for name in names:
            ok, why = run(name, **({} if args.name == "all" else options))
            failed += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {name}: {why}")
        if args.name == "all":
            for name, needs in needing_options().items():
                keys = " ".join(f"{k}=..." for k in needs)
                print(f"SKIP  {name}: skipped (needs {keys}); run: hbench check {name} {keys}")
    except (ValueError, ImportError, AttributeError) as e:
        raise SystemExit(f"check: {e}") from None
    return 1 if failed else 0


class RunChecks(Command):
    about = (
        "gates and sanity checks: no name lists them, a name runs one (PASS / FAIL + why), "
        "`all` runs every offline one"
    )
    positional = "name"

    def __init__(self) -> None:
        super().__init__()
        self.key("name", "", text, "a check name, or `all`")
        self.key("family", "", text, "for checks that take a family")
        self.key("readouts", "", text, "for checks that take a readouts file")
        self.key("option", None, many, "any other option as KEY=VALUE (plug-ins; repeatable)")

    def execute(self) -> int:
        return cmd_check(self)


COMMANDS = {"check": RunChecks}
