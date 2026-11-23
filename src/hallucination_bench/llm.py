"""One structured-JSON judge client with two routes, shared by every stage of a process.

Ported from WorkspaceBench ``src/wsbench/llm.py`` @ 92d763e (MIT), with these changes:

* **Keyless Anthropic auth.** ``claude-*`` models go to the Anthropic SDK. When
  ``ANTHROPIC_API_KEY`` is unset, :func:`api_key` returns ``None`` (the source raised) and the
  client is built as ``AsyncAnthropic()`` with no ``api_key``, so the SDK picks up the ``ant``
  CLI OAuth profile. With ``ANTHROPIC_AUTH_TOKEN`` set (and no key) the client sends that OAuth
  bearer token instead, as WorkspaceBench does. Everything else goes to OpenRouter
  (``OPENROUTER_API_KEY``).
* **No temperature on the Anthropic route** (newer Claude models reject it), as in the source.
* **Thinking.** Claude Opus 5.5 rejects ``thinking: {type: "disabled"}`` (400), so the request
  omits ``thinking`` and sets ``output_config.effort`` (default ``"low"``, the closest available
  setting to "thinking off"). Structured output uses ``output_config.format`` (json_schema),
  exactly as the source.
* **One client per route per process**, one asyncio semaphore (``concurrency``) and the
  process-wide RPM pacer: several arms and stages judged concurrently never multiply the rate.
* **Per-call usage** (tokens, USD) is returned with each result so callers can store it in the
  cache and tally spend per stage.

* **Opt-in Message Batches backend.** ``LLM(backend="batches")`` (``--judge-backend batches``,
  ``$HBENCH_JUDGE_BACKEND``) queues the structured Anthropic calls of concurrent workers and
  sends them through :mod:`hallucination_bench.judge_batches`: the same requests at half the
  token price, outside the live rate limits. Whatever a batch does not answer cleanly is re-run
  live, so results, cache fingerprints and resume are those of the live backend.

Transient failures (408/409/429/5xx, timeouts, connection errors, garbled bodies) are retried
with jittered backoff; any other per-call failure yields ``None`` (an unjudged call, never a
clean one). Problems that would fail every call raise :class:`JudgeConfigError`.
"""

import asyncio
import json
import os
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from hallucination_bench import judge_batches

OPENROUTER = "https://openrouter.ai/api/v1"
_TRANSIENT = (408, 409, 429, 500, 502, 503, 504, 529)
_FATAL = ("AuthenticationError", "PermissionDeniedError", "NotFoundError")
_ATTEMPTS = 12
_DEFAULT_MAX_TOKENS = {"openrouter": 8000, "anthropic": 16000}
DEFAULT_EFFORT = "low"
# plan 0029 §3.2: above this ``max_tokens`` the Anthropic route streams (the SDK refuses long
# non-streaming requests); per-layer calls (16k) never stream, joint calls (64k) always do
STREAM_ABOVE_TOKENS = 21_333
JOINT_MAX_TOKENS = 64_000
JOINT_TIMEOUT = 1800.0

# USD per million tokens: (input, output, cache write 5m, cache read). First-party list prices.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 5.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 6.25, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 2.5, 0.20),
    "claude-haiku-4-5": (1.0, 5.0, 1.25, 0.10),
}

Route = Literal["anthropic", "openrouter"]


class JudgeConfigError(RuntimeError):
    """A configuration problem that would fail every judge call (raised, never degraded)."""


@dataclass
class Usage:
    """Tally for one call or many. ``calls`` counts HTTP 200 responses (including garbled bodies
    that were then retried); ``errors``/``refusals`` count calls left unjudged."""

    usd: float = 0.0
    calls: int = 0
    retries: int = 0
    errors: int = 0
    refusals: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, other: "Usage | dict[str, Any] | None") -> None:
        if other is None:
            return
        d = other if isinstance(other, dict) else asdict(other)
        for k in asdict(self):
            setattr(self, k, getattr(self, k) + (d.get(k) or 0))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def report(self) -> str:
        return (
            f"calls={self.calls} retries={self.retries} errors={self.errors} "
            f"refusals={self.refusals} spend=${self.usd:.4f} in_tok={self.input_tokens} "
            f"out_tok={self.output_tokens} cache_w={self.cache_write_tokens} "
            f"cache_r={self.cache_read_tokens}"
        )


def route(model: str) -> Route:
    return "anthropic" if model.startswith("claude-") else "openrouter"


def api_key(model: str) -> str | None:
    """The key for ``model``'s route. Anthropic: ``ANTHROPIC_API_KEY`` or ``None`` (keyless: the
    SDK falls back to the ``ant`` CLI profile). OpenRouter: required."""
    if route(model) == "anthropic":
        if _ANTHROPIC_CLIENT is not None:
            return None  # the injected client carries its own credentials
        return os.environ.get("ANTHROPIC_API_KEY") or None
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key.startswith("sk-or-"):
        raise JudgeConfigError("OPENROUTER_API_KEY is missing or not an OpenRouter key (sk-or-…)")
    return key


def schema_block(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": list(required),
            "properties": properties,
        },
    }


def anthropic_usd(model: str, u: Usage) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0
    return (
        u.input_tokens * p[0]
        + u.output_tokens * p[1]
        + u.cache_write_tokens * p[2]
        + u.cache_read_tokens * p[3]
    ) / 1e6


# ---------------------------------------------------------------- pacing / backoff seams


def _now() -> float:
    return time.monotonic()


class _Pacer:
    """Process-wide next-slot time behind a lock: shared by every loop, thread and stage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0.0

    def reset(self) -> None:
        with self._lock:
            self._next = 0.0

    def reserve(self, rpm: float) -> float:
        """Reserve the next slot; return how long the caller must wait for it (seconds)."""
        with self._lock:
            now = _now()
            slot = max(now, self._next)
            self._next = slot + 60.0 / rpm
            return slot - now


_PACER = _Pacer()

_sleep = asyncio.sleep  # seam: tests patch this instead of the global asyncio.sleep


async def _pace(rpm: float) -> None:
    delay = _PACER.reserve(rpm)
    if delay > 0:
        await _sleep(delay)


def _backoff(attempt: int, status: int | None) -> float:
    cap = 90.0 if status == 429 else 30.0
    return min(cap, 2.0 * 2**attempt) * (0.5 + random.random())


TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"  # an OAuth bearer token; used only when no API key is set
OAUTH_BETA = "oauth-2025-04-20"


_ANTHROPIC_CLIENT: Callable[[], Any] | None = None


def use_anthropic_client(factory: Callable[[], Any] | None) -> None:
    """Build every Anthropic client with ``factory()`` (an ``AsyncAnthropic``; ``None`` restores
    the default). For credentials the environment cannot carry, e.g. workload identity
    federation with a token provider. ``LLM.aclose`` closes the client it was given."""
    global _ANTHROPIC_CLIENT
    _ANTHROPIC_CLIENT = factory


BACKEND_ENV = "HBENCH_JUDGE_BACKEND"  # live (default) | batches
BATCH_ENV_PREFIX = "HBENCH_JUDGE_BATCH"  # _DIR, _MAX_REQUESTS, ...: judge_batches.Settings.from_env
BATCH_PRICE = 0.5  # Message Batches bill every token at half the list price


def set_backend(name: str) -> None:
    """Choose how structured Anthropic calls travel, for every ``LLM`` this process builds."""
    os.environ[BACKEND_ENV] = judge_backend(name)


def judge_backend(name: str | None = None) -> str:
    """``name`` (default: ``$HBENCH_JUDGE_BACKEND``, else ``live``), checked."""
    name = name or os.environ.get(BACKEND_ENV, "") or "live"
    if name not in judge_batches.BACKENDS:
        raise JudgeConfigError(f"judge backend must be live or batches, got {name!r}")
    return name


def _make_client(route_: str, key: str | None) -> Any:
    """The ONLY place SDK clients are built (tests monkeypatch this to inject fakes)."""
    if route_ == "anthropic":
        if _ANTHROPIC_CLIENT is not None:
            return _ANTHROPIC_CLIENT()
        from anthropic import AsyncAnthropic

        if key is None and (token := os.environ.get(TOKEN_ENV)):
            # a static OAuth bearer token: the SDK adds the OAuth beta flag only for its own
            # credential providers, so it goes in as a default header (as in WorkspaceBench)
            return AsyncAnthropic(
                auth_token=token,
                api_key=None,
                max_retries=0,
                default_headers={"anthropic-beta": OAUTH_BETA},
            )
        if key is None:  # keyless: the SDK resolves the `ant` CLI OAuth profile
            return AsyncAnthropic(max_retries=0)
        return AsyncAnthropic(api_key=key, max_retries=0)
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=key, base_url=OPENROUTER, max_retries=0)


# ---------------------------------------------------------------- one call


def _parse_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    data = json.loads(content)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("not a JSON object", content, 0)
    return data


async def _openrouter_once(
    client: Any,
    system: str,
    user: str,
    *,
    schema: dict[str, Any],
    model: str,
    max_tokens: int,
    timeout: float,
    usage: Usage,
) -> dict[str, Any] | None:
    resp = await client.chat.completions.create(
        timeout=timeout,
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": schema},
        extra_body={"usage": {"include": True}, "provider": {"require_parameters": True}},
    )
    usage.calls += 1
    if not getattr(resp, "choices", None):  # an error delivered in a 200 body
        raise ValueError("response has no choices")
    u = getattr(resp, "usage", None)
    cost = getattr(u, "cost", None) if u is not None else None
    if cost is None and u is not None:
        cost = (getattr(u, "model_extra", None) or {}).get("cost")
    usage.usd += float(cost or 0.0)
    if u is not None:
        usage.input_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
        usage.output_tokens += int(getattr(u, "completion_tokens", 0) or 0)
    return _parse_object(resp.choices[0].message.content or "")


def anthropic_params(
    system: str,
    user: str,
    *,
    schema: dict[str, Any],
    model: str,
    effort: str | None,
    max_tokens: int,
) -> dict[str, Any]:
    """The Messages request of one structured call: what the live call sends (plus its
    ``timeout``) and, unchanged, the ``params`` of the same call in a batch."""
    output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema["schema"]}}
    if effort is not None:
        output_config["effort"] = effort
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
        "output_config": output_config,
    }


async def _anthropic_once(
    client: Any,
    system: str,
    user: str,
    *,
    schema: dict[str, Any],
    model: str,
    effort: str | None,
    max_tokens: int,
    timeout: float,
    usage: Usage,
    stream: bool = False,
) -> dict[str, Any] | None:
    kwargs: dict[str, Any] = {
        "timeout": timeout,
        **anthropic_params(
            system, user, schema=schema, model=model, effort=effort, max_tokens=max_tokens
        ),
    }
    if stream:  # plan 0029: long joint calls; the final message has the non-streaming shape
        async with client.messages.stream(**kwargs) as s:
            resp = await s.get_final_message()
    else:
        resp = await client.messages.create(**kwargs)
    return _anthropic_result(resp, model=model, usage=usage)


def _anthropic_result(
    resp: Any, *, model: str, usage: Usage, price: float = 1.0
) -> dict[str, Any] | None:
    """Tally one Messages response (live, or from a batch at ``price`` x the list price) and
    parse its JSON object."""
    usage.calls += 1
    u = getattr(resp, "usage", None)
    if u is not None:
        call = Usage(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
            cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        )
        call.usd = anthropic_usd(model, call) * price
        usage.add(call)
    stop = getattr(resp, "stop_reason", None)
    if stop == "refusal":
        usage.refusals += 1
        print(f"  llm refusal: {model}", flush=True)
        return None
    if stop in ("max_tokens", "model_context_window_exceeded"):
        usage.errors += 1
        print(f"  llm error: {model} stop_reason={stop}", flush=True)
        return None
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    if text is None:
        raise ValueError("response has no text block")
    return _parse_object(text)


_RUN_LIVE: Any = object()  # a batch's answer for "ask this one through the live path"


class _BatchQueue:
    """Gathers the single calls of concurrent workers into Message Batches. A call waits in
    :meth:`ask`; the waiting calls are sent once none arrived for ``linger_seconds`` (or
    ``max_requests`` wait). ``ask`` returns the batch's result object for the request, or
    ``_RUN_LIVE`` for whatever the batch did not settle (errored / expired / canceled / missing
    entries, and every call of a round smaller than ``min_requests``)."""

    def __init__(self, client: Callable[[], Any], settings: judge_batches.Settings) -> None:
        self._client = client
        self.settings = settings
        self._waiting: list[tuple[dict[str, Any], asyncio.Future[Any]]] = []
        self._last = 0.0
        self._tasks: set[asyncio.Task[None]] = set()
        self._draining = False

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def ask(self, params: dict[str, Any]) -> Any:
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._waiting.append((params, fut))
        self._last = _now()
        if not self._draining:
            self._draining = True
            self._spawn(self._drain())
        return await fut

    async def _drain(self) -> None:
        cfg = self.settings
        try:
            while self._waiting:
                while (
                    len(self._waiting) < cfg.max_requests
                    and (quiet := _now() - self._last) < cfg.linger_seconds
                ):
                    await asyncio.sleep(max(0.005, cfg.linger_seconds - quiet))
                take = self._waiting[: cfg.max_requests]
                del self._waiting[: cfg.max_requests]
                self._spawn(self._send(take))
        finally:
            self._draining = False

    async def _send(self, take: list[tuple[dict[str, Any], asyncio.Future[Any]]]) -> None:
        futs = dict(
            zip(judge_batches.unique_ids([p for p, _f in take]), [f for _p, f in take], strict=True)
        )

        def entry(custom_id: str, result: Any) -> None:
            fut = futs[custom_id]
            if not fut.done():
                ok = getattr(result, "type", None) == "succeeded"
                fut.set_result(result.message if ok else _RUN_LIVE)

        try:
            if len(take) >= self.settings.min_requests:
                print(f"  batches: {len(take)} request(s)", flush=True)
                requests = list(zip(futs, [p for p, _f in take], strict=True))
                await judge_batches.run(
                    self._client(), requests, entry, settings=self.settings, sleep=_sleep
                )
        except Exception as e:
            name = type(e).__name__
            if name in _FATAL or getattr(e, "status_code", None) in (401, 402, 403):
                err = JudgeConfigError(f"{name}: {str(e)[:300]}")
                for fut in futs.values():
                    if not fut.done():
                        fut.set_exception(err)
                return
            # the Batches API itself is failing: nothing is lost, the live path takes over
            print(f"  batches: {name}: {str(e)[:200]}; unanswered calls run live", flush=True)
        finally:
            for fut in futs.values():
                if not fut.done():
                    fut.set_result(_RUN_LIVE)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


@dataclass
class CallResult:
    result: dict[str, Any] | None
    usage: Usage = field(default_factory=Usage)


def joint_llm(*, rpm: float, concurrency: int, effort: str | None) -> "LLM":
    """Plan 0029 §3.2: the joint 11-layer cells' own client (64k output tokens, 30 min,
    streaming); per-layer calls keep the default 16k / 300 s client."""
    return LLM(
        rpm=rpm,
        concurrency=concurrency,
        effort=effort,
        timeout=JOINT_TIMEOUT,
        max_tokens=JOINT_MAX_TOKENS,
        stream=True,
    )


class LLM:
    """The process's judge client: lazily built SDK clients (one per route), one semaphore over
    every call, the shared RPM pacer, and a preflight per model."""

    def __init__(
        self,
        *,
        rpm: float = 1500.0,
        concurrency: int = 64,
        effort: str | None = DEFAULT_EFFORT,
        timeout: float = 300.0,
        max_tokens: int | None = None,
        stream: bool | None = None,
        backend: str | None = None,
    ) -> None:
        if concurrency < 1 or rpm <= 0:
            raise JudgeConfigError(
                f"concurrency must be >= 1 and rpm > 0 (got {concurrency}, {rpm})"
            )
        self.rpm = rpm
        self.effort = effort
        self.timeout = timeout
        self.max_tokens = max_tokens
        # None: stream exactly when max_tokens > STREAM_ABOVE_TOKENS (Anthropic route only)
        self.stream = stream
        self._sem = asyncio.Semaphore(concurrency)
        self._clients: dict[str, Any] = {}
        self._preflighted: set[str] = set()
        self._pf_lock = asyncio.Lock()
        # backend None: $HBENCH_JUDGE_BACKEND, else live. The queue is shared by with_effort copies
        self.backend = judge_backend(backend)
        self._batches: _BatchQueue | None = None
        if self.backend == "batches":
            self._batches = _BatchQueue(
                lambda: self._client("claude-"),  # the shared Anthropic client, built once
                judge_batches.Settings.from_env(BATCH_ENV_PREFIX),
            )

    def _client(self, model: str) -> Any:
        r = route(model)
        if r not in self._clients:
            self._clients[r] = _make_client(r, api_key(model))
        return self._clients[r]

    async def call(
        self, *, model: str, system: str, user: str, schema: dict[str, Any]
    ) -> CallResult:
        """One structured call -> parsed object (or ``None``) plus its usage. Raises
        :class:`JudgeConfigError` on a fatal configuration error. Under the batches backend an
        Anthropic call first waits for its batch (a request cannot stream there; ``timeout``,
        ``rpm`` and ``concurrency`` bound only the calls that still run live)."""
        return await self._call(model=model, system=system, user=user, schema=schema)

    async def _call(
        self, *, model: str, system: str, user: str, schema: dict[str, Any], batch: bool = True
    ) -> CallResult:
        r = route(model)
        client = self._client(model)
        max_tokens = self.max_tokens or _DEFAULT_MAX_TOKENS[r]
        # detokenised lens readouts can contain lone surrogates, which both APIs reject every time
        system = system.encode("utf-8", "replace").decode("utf-8")
        user = user.encode("utf-8", "replace").decode("utf-8")
        usage = Usage()
        if batch and self._batches is not None and r == "anthropic":
            msg = await self._batches.ask(
                anthropic_params(
                    system,
                    user,
                    schema=schema,
                    model=model,
                    effort=self.effort,
                    max_tokens=max_tokens,
                )
            )
            if msg is not _RUN_LIVE:
                try:
                    res = _anthropic_result(msg, model=model, usage=usage, price=BATCH_PRICE)
                    return CallResult(res, usage)
                except ValueError:  # an unparsable body (JSONDecodeError included)
                    usage.retries += 1  # ... is retried by the live path below
        async with self._sem:
            for attempt in range(_ATTEMPTS):
                await _pace(self.rpm)
                try:
                    if r == "anthropic":
                        res = await _anthropic_once(
                            client,
                            system,
                            user,
                            schema=schema,
                            model=model,
                            effort=self.effort,
                            max_tokens=max_tokens,
                            timeout=self.timeout,
                            usage=usage,
                            stream=self.streams(max_tokens),
                        )
                    else:
                        res = await _openrouter_once(
                            client,
                            system,
                            user,
                            schema=schema,
                            model=model,
                            max_tokens=max_tokens,
                            timeout=self.timeout,
                            usage=usage,
                        )
                    return CallResult(res, usage)
                except Exception as e:
                    name = type(e).__name__
                    status = getattr(e, "status_code", None)
                    if name in _FATAL or (status in (400, 401, 402, 403) and _is_config_400(e)):
                        raise JudgeConfigError(f"{name}: {str(e)[:300]}") from e
                    if _no_credentials(e):
                        raise JudgeConfigError(
                            f"no Anthropic credentials for {model}: export ANTHROPIC_API_KEY, "
                            "or ANTHROPIC_AUTH_TOKEN (an OAuth bearer token), or log in with "
                            "`ant auth login` (`hbench try --dry_run` needs none of them)"
                        ) from e
                    transient = (
                        status in _TRANSIENT
                        or "Timeout" in name
                        or "Connection" in name
                        or "Deadline" in name
                        or isinstance(e, json.JSONDecodeError | ValueError)
                    )
                    if transient and attempt < _ATTEMPTS - 1:
                        usage.retries += 1
                        await _sleep(_backoff(attempt, status))
                        continue
                    usage.errors += 1
                    print(f"  llm error: {name}: {str(e)[:200]}", flush=True)
                    return CallResult(None, usage)
        return CallResult(None, usage)

    def with_effort(self, effort: str | None) -> "LLM":
        """This client at another ``effort`` (plan 0029 Amendment 2: the continuation verifier
        runs at ``medium``). The copy shares the semaphore, the SDK clients and the preflight
        set, so ``--concurrency`` still bounds every call of the process together."""
        import copy

        out = copy.copy(self)
        out.effort = effort
        return out

    def streams(self, max_tokens: int) -> bool:
        return self.stream if self.stream is not None else max_tokens > STREAM_ABOVE_TOKENS

    async def preflight(self, model: str) -> None:
        """One tiny structured call per model, so a bad key/auth/model/structured-output setup
        fails loudly before a long run starts."""
        async with self._pf_lock:
            if model in self._preflighted:
                return
            schema = schema_block("t", {"a": {"type": "integer"}}, ["a"])
            # a transient failure (e.g. the org output-token rate limit is saturated by other
            # runs) is retried with a backoff for up to ~30 min; a 401 still raises at once
            for attempt in range(30):
                out = await self._call(  # one call, answered now: never batched
                    model=model,
                    system="Reply JSON.",
                    user='Return {"a":1}',
                    schema=schema,
                    batch=False,
                )
                if out.result == {"a": 1} or out.result is not None:
                    break
                print(
                    f"  preflight {model}: no answer (attempt {attempt + 1}/30), retry in 60 s",
                    flush=True,
                )
                await asyncio.sleep(60)
            if out.result != {"a": 1}:
                raise JudgeConfigError(f"preflight failed for {model}: {out.result}")
            self._preflighted.add(model)

    async def aclose(self) -> None:
        if self._batches is not None:
            await self._batches.aclose()
        for c in self._clients.values():
            close = getattr(c, "close", None)
            if close is not None:
                await close()
        self._clients.clear()


def _no_credentials(e: Exception) -> bool:
    """The SDK found no key, token or profile: it raises a ``TypeError`` before any request, so
    every call would fail the same way. Fatal, like a 401."""
    return isinstance(e, TypeError) and "authentication method" in str(e)


def _is_config_400(e: Exception) -> bool:
    """401/402/403 are always fatal (bad key, no credits, no access). A 400 is fatal only when it
    rejects a request parameter (it would fail every call), not a single prompt's content."""
    status = getattr(e, "status_code", None)
    if status != 400:
        return True
    msg = str(e).lower()
    return any(s in msg for s in ("not supported", "unsupported", "extra inputs", "schema"))
