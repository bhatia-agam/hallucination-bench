"""The pieces every ``hbench`` subcommand shares: a ``pydra.Config`` base class whose keys are
declared in ``__init__`` (default, kind, one-line meaning) and coerced in ``finalize()``, the
step that turns a command line into pydra overrides (``key=value``, ``--flag`` for a boolean,
and the old ``--some-flag value`` spelling), and the ``--help`` text."""

import json
import re
import textwrap
from collections.abc import Callable, Iterable
from typing import Any

import pydra

EXIT_USAGE = 2
HELP_FLAGS = ("--help", "-h")
SPELLINGS = (
    "Give a key as key=value and a boolean key as --key. The old spelling still works: "
    "--some-key value, --some-key=value, --some-flag."
)
_KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", re.DOTALL)


class UsageError(ValueError):
    """A command line that cannot be parsed (exit code ``EXIT_USAGE``)."""


# ----------------------------------------------------------------------------------- key kinds
# A kind turns the raw value of a key into its final type, or raises ValueError.


def text(v: object) -> str:
    return str(v)


def integer(v: object) -> int:
    if isinstance(v, bool):
        raise ValueError(f"invalid int value: {v!r}")
    try:
        return int(v)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        raise ValueError(f"invalid int value: {v!r}") from None


def number(v: object) -> float:
    if isinstance(v, bool):
        raise ValueError(f"invalid float value: {v!r}")
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"invalid float value: {v!r}") from None


def boolean(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ("true", "t", "1", "false", "f", "0"):
        return v.lower() in ("true", "t", "1")
    raise ValueError(f"expected True or False, got {v!r}")


def many(v: object) -> list[str] | None:
    """A repeatable key: ``None`` when never given, else the list of its values. In the
    ``key=value`` spelling one value may hold a comma list; ``--key v`` adds ``v`` whole."""
    if v is None:
        return None
    if isinstance(v, str):
        return [x for x in v.split(",") if x]
    if isinstance(v, Iterable):
        return [str(x) for x in v]
    raise ValueError(f"expected a list, got {v!r}")


def optional(kind: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """``kind``, with ``None`` (the key was not given) kept."""

    def inner(v: object) -> object:
        return None if v is None else kind(v)

    return inner


def choice(*options: str | Callable[[], Iterable[str]]) -> Callable[[Any], str]:
    """One of ``options`` (a single callable is asked for them when a value is checked)."""

    def inner(v: object) -> str:
        allowed = list(options[0]()) if len(options) == 1 and callable(options[0]) else options
        if v not in allowed:
            names = ", ".join(repr(x) for x in allowed)
            raise ValueError(f"invalid choice: {v!r} (choose from {names})")
        return str(v)

    return inner


# ----------------------------------------------------------------------------------- commands


class Command(pydra.Config):
    """One subcommand. A subclass declares its keys in ``__init__`` with ``self.key(...)`` and
    runs in ``execute()``; ``finalize()`` gives every key its type."""

    prog = ""  # "hbench run", set where the command is registered
    about = ""  # one line for the command list and the top of --help
    positional = ""  # the key a bare argument fills (`hbench check <name>`)

    def __init__(self) -> None:
        super().__init__()
        self._kinds: dict[str, Callable[[Any], Any]] = {}
        self._meanings: dict[str, str] = {}

    def key(self, name: str, default: object, kind: Callable[[Any], Any], meaning: str) -> None:
        """Declare key ``name``: its default, its kind and a one-line meaning for ``--help``."""
        self._kinds[name] = kind
        self._meanings[name] = meaning
        setattr(self, name, default)

    def finalize(self) -> None:
        for name, kind in self._kinds.items():
            try:
                setattr(self, name, kind(getattr(self, name)))
            except ValueError as e:
                raise UsageError(f"argument {name}: {e}") from None

    def execute(self) -> int:
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        return {k: getattr(self, k) for k in self._kinds}

    @classmethod
    def of(cls, **values: object) -> "Command":
        """This command with ``values`` set, finalized: how one command runs another."""
        command = cls()
        if unknown := sorted(set(values) - set(command._kinds)):
            raise UsageError(f"unknown key {unknown[0]!r}")
        pydra.apply_overrides(command, [f"{k}={v!r}" for k, v in values.items()])
        return command


def _resolve(command: Command, name: str) -> str:
    """The key an old-spelling flag names: itself, or the one key it abbreviates."""
    if name in command._kinds:
        return name
    found = [k for k in command._kinds if k.startswith(name)] if name else []
    if len(found) == 1:
        return found[0]
    if found:
        raise UsageError(f"ambiguous option: --{name} could match {', '.join(found)}")
    raise UsageError(f"unknown key {name!r} ({command.prog} --help lists the keys)")


def to_pydra(command: Command, argv: list[str]) -> list[str]:
    """``argv`` as pydra overrides for ``command``. Accepts ``key=value``, ``--key`` for a
    boolean, the old ``--some-key value`` / ``--some-key=value`` / ``--some-flag`` spelling
    (dashes become underscores; a unique prefix of a key is enough, as before) and one bare
    argument for the command's positional key. Every value reaches pydra quoted, so a key gets
    the text that was typed and its kind decides the type; ``key=None`` unsets a key."""
    kinds = command._kinds
    single: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    show = False
    bare: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        i += 1
        if arg == "--show":
            show = True
            continue
        old = arg.startswith("-") and len(arg) > 1
        if old:
            name, eq, value = arg.lstrip("-").partition("=")
            key = _resolve(command, name.replace("-", "_"))
            if not eq and kinds[key] is boolean:
                single[key] = "True"
                continue
            if not eq:
                if i >= len(argv):
                    raise UsageError(f"argument {arg}: expected one argument")
                value = argv[i]
                i += 1
        elif m := _KEY_VALUE.fullmatch(arg):
            key, value = m[1], m[2]
            if key not in kinds:
                raise UsageError(f"unknown key {key!r} ({command.prog} --help lists the keys)")
        else:
            bare.append(arg)
            continue
        if kinds[key] is many:
            lists.setdefault(key, []).extend([value] if old else many(value) or [])
        else:
            single[key] = "None" if value == "None" and not old else repr(value)
    if bare and command.positional and command.positional not in single:
        single[command.positional] = repr(bare.pop(0))
    if bare:
        raise UsageError(f"unrecognized arguments: {' '.join(bare)}")
    out = [f"{k}={v}" for k, v in single.items()] + [f"{k}={v!r}" for k, v in lists.items()]
    return [*out, "--show"] if show else out


def parse(command: Command, argv: list[str]) -> bool:
    """Apply ``argv`` to ``command`` and finalize it; True when ``--show`` was given. Raises
    ``UsageError`` for anything wrong with the command line."""
    try:
        return pydra.apply_overrides(command, to_pydra(command, argv))
    except UsageError:
        raise
    except (ValueError, TypeError, AttributeError, IndexError, SyntaxError) as e:
        raise UsageError(str(e)) from None


def show(command: Command) -> str:
    """The resolved config, as ``--show`` prints it."""
    return json.dumps(command.to_dict(), indent=1)


def usage(command: Command) -> str:
    bare = f" [{command.positional}]" if command.positional else ""
    return f"usage: {command.prog}{bare} [key=value ...] [--key ...] [--show | --help]"


def help_text(command: Command, width: int = 100) -> str:
    """``--help``: every key with its default and its meaning."""
    lines = [usage(command)]
    if command.about:
        lines += ["", *textwrap.wrap(command.about, width)]
    if command._kinds:
        lines += ["", "keys (key=default, then what it means):"]
    for name in command._kinds:
        default = getattr(command, name)
        shown = "<required>" if default is pydra.REQUIRED else repr(default)
        lines.append(f"  {name}={shown}")
        lines += textwrap.wrap(
            command._meanings[name], width, initial_indent="      ", subsequent_indent="      "
        )
    return "\n".join([*lines, "", *textwrap.wrap(SPELLINGS, width)])
