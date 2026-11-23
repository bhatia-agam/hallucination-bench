"""Extra reader (arm) names. The index and the safety exports keep only the readers on their
allow-lists; ``HBENCH_EXTRA_READERS=a,b`` (or :func:`add`) puts more names on every one of them."""

import os
from collections.abc import Iterable

ENV = "HBENCH_EXTRA_READERS"  # comma-separated reader names, read at call time
_added: list[str] = []


def add(*names: str) -> None:
    """Allow these reader names for the rest of the process (the code-side twin of the variable)."""
    _added.extend(n for n in names if n and n not in _added)


def clear() -> None:
    """Forget the names given to :func:`add` (the environment variable is not touched)."""
    _added.clear()


def extra() -> tuple[str, ...]:
    """The extra reader names: the variable's, then :func:`add`'s, in order, without repeats."""
    env = [n.strip() for n in os.environ.get(ENV, "").split(",") if n.strip()]
    return tuple(dict.fromkeys([*env, *_added]))


def with_extra(base: Iterable[str]) -> tuple[str, ...]:
    """``base`` followed by the extra names it does not already hold."""
    return tuple(dict.fromkeys([*base, *extra()]))
