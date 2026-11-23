"""Offline tests for the Message Batches judge backend: a fake Anthropic client, no network."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import FakeLLM, stage_of

from hallucination_bench import cli, judge_batches
from hallucination_bench import llm as L
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.llm import LLM, JudgeConfigError

SCHEMA = L.schema_block("t", {"a": {"type": "integer"}}, ["a"])
CLAUDE = "claude-opus-5-5"


class AuthenticationError(Exception):
    status_code = 401


class NotFoundError(Exception):
    status_code = 404


def _message(text: str, stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def _user(params: dict) -> str:
    return params["messages"][0]["content"]


def _echo(params: dict, bump: int = 0) -> Any:
    return _message(json.dumps({"a": int(_user(params)) + bump}))


class Server:
    """What outlives one client: the submitted batches (as the API keeps them between runs)."""

    def __init__(self) -> None:
        self.batches: dict[str, list[dict]] = {}


class FakeClient:
    """``AsyncAnthropic``-shaped: ``messages.create`` (live) and ``messages.batches``.
    ``reply(params)`` writes a live message; ``answer(params)`` a batch result (default: a
    success carrying ``reply``'s message; ``None`` = no row for the request)."""

    def __init__(self, server: Server | None = None, *, reply=None, answer=None):
        self.server = server or Server()
        self.reply = reply or (lambda p: _echo(p, 1000))  # +1000: answered live
        self.answer = answer or (
            lambda p: SimpleNamespace(type="succeeded", message=(reply or _echo)(p))
        )
        self.live: list[dict[str, Any]] = []
        self.created: list[list[dict]] = []
        self.fail_retrieve: BaseException | None = None
        self.closed = False
        batches = SimpleNamespace(
            create=self._create, retrieve=self._retrieve, results=self._results
        )
        self.messages = SimpleNamespace(create=self._live, batches=batches)

    async def _live(self, **kw: Any) -> Any:
        self.live.append(kw)
        return self.reply(kw)

    async def _create(self, *, requests: list[dict]) -> Any:
        self.created.append(requests)
        batch_id = f"msgbatch_{len(self.server.batches)}"
        self.server.batches[batch_id] = requests
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    async def _retrieve(self, batch_id: str) -> Any:
        if self.fail_retrieve is not None:
            raise self.fail_retrieve
        if batch_id not in self.server.batches:
            raise NotFoundError(batch_id)
        return SimpleNamespace(id=batch_id, processing_status="ended")

    async def _results(self, batch_id: str) -> Any:
        async def stream() -> Any:
            for req in reversed(self.server.batches[batch_id]):  # any order
                result = self.answer(req["params"])
                if result is not None:
                    yield SimpleNamespace(custom_id=req["custom_id"], result=result)

        return stream()

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    async def no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(L, "_sleep", no_sleep)
    monkeypatch.setattr(L, "api_key", lambda m: None)
    monkeypatch.setenv("HBENCH_JUDGE_BATCH_MIN_REQUESTS", "1")
    monkeypatch.setenv("HBENCH_JUDGE_BATCH_LINGER_SECONDS", "0.02")
    for name in ("HBENCH_JUDGE_BACKEND", "HBENCH_JUDGE_BATCH_DIR"):
        monkeypatch.delenv(name, raising=False)


def _client(monkeypatch, client: FakeClient) -> FakeClient:
    monkeypatch.setattr(L, "_make_client", lambda r, k: client)
    return client


def _ask(n: int, **kw: Any) -> list[L.CallResult]:
    """``n`` concurrent calls (users "0".."n-1") on one client, as the pipeline's workers."""

    async def go() -> list[L.CallResult]:
        llm = LLM(rpm=6e6, **{"backend": "batches", **kw})
        try:
            return await asyncio.gather(
                *(
                    llm.call(model=CLAUDE, system="judge", user=str(i), schema=SCHEMA)
                    for i in range(n)
                )
            )
        finally:
            await llm.aclose()

    return asyncio.run(go())


def _values(outs: list[L.CallResult]) -> list[Any]:
    return [o.result and o.result["a"] for o in outs]


# ---------------------------------------------------------------- request parity


def test_batch_requests_are_the_live_requests(monkeypatch):
    live = _client(monkeypatch, FakeClient())
    _ask(3, backend="live", effort="medium", max_tokens=777, timeout=12.0)
    batched = _client(monkeypatch, FakeClient())
    _ask(3, effort="medium", max_tokens=777, timeout=12.0)
    assert len(live.live) == 3 and batched.live == [] and len(batched.created) == 1
    want = sorted(({k: v for k, v in kw.items() if k != "timeout"} for kw in live.live), key=_user)
    got = sorted((r["params"] for r in batched.created[0]), key=_user)
    assert got == want  # same model, max_tokens, cached system block, schema, effort
    assert got[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert got[0]["output_config"]["effort"] == "medium" and got[0]["max_tokens"] == 777
    ids = [r["custom_id"] for r in batched.created[0]]
    assert len(set(ids)) == 3 and all(len(i) <= 64 and i.replace("-", "").isalnum() for i in ids)


def test_long_calls_are_batched_without_streaming(monkeypatch):
    client = _client(monkeypatch, FakeClient())
    outs = _ask(2, max_tokens=64_000, stream=True)  # the joint client's settings
    assert _values(outs) == [0, 1] and client.live == []
    assert all("stream" not in r["params"] for r in client.created[0])


def test_backend_selection(monkeypatch):
    client = _client(monkeypatch, FakeClient())
    assert _values(_ask(2, backend=None)) == [1000, 1001] and client.created == []  # default
    monkeypatch.setenv("HBENCH_JUDGE_BACKEND", "batches")
    assert _values(_ask(2, backend=None)) == [0, 1]
    with pytest.raises(JudgeConfigError, match="live or batches"):
        LLM(backend="soon")
    args = cli.build_parser().parse_args(["run", "--arms", "x", "--judge-backend", "batches"])
    assert args.judge_backend == "batches"
    assert cli.build_parser().parse_args(["run", "--arms", "x"]).judge_backend is None


# ---------------------------------------------------------------- results


def test_results_usage_and_half_price(monkeypatch):
    _client(monkeypatch, FakeClient())
    live = _ask(1, backend="live")[0].usage
    batched = _ask(1)[0].usage
    assert (live.calls, batched.calls) == (1, 1) and live.input_tokens == batched.input_tokens
    assert live.usd == pytest.approx(4.0) and batched.usd == pytest.approx(2.0)


def test_calls_arriving_together_share_a_batch_and_later_ones_get_the_next(monkeypatch):
    client = _client(monkeypatch, FakeClient())

    async def go() -> list[int]:
        llm = LLM(backend="batches")
        ask = lambda i: llm.call(model=CLAUDE, system="judge", user=str(i), schema=SCHEMA)  # noqa: E731
        first = await asyncio.gather(*(ask(i) for i in range(5)))
        second = await asyncio.gather(*(ask(i) for i in range(5, 8)))  # the next stage
        await llm.aclose()
        return _values([*first, *second])

    assert asyncio.run(go()) == list(range(8))
    assert [len(b) for b in client.created] == [5, 3] and client.live == []


def test_a_round_below_min_requests_runs_live_and_so_does_preflight(monkeypatch):
    monkeypatch.setenv("HBENCH_JUDGE_BATCH_MIN_REQUESTS", "4")
    client = _client(monkeypatch, FakeClient())
    assert _values(_ask(3)) == [1000, 1001, 1002] and client.created == []
    assert _values(_ask(4)) == [0, 1, 2, 3] and len(client.created) == 1
    monkeypatch.setenv("HBENCH_JUDGE_BATCH_MIN_REQUESTS", "1")
    client = _client(monkeypatch, FakeClient(reply=lambda p: _message('{"a": 1}')))
    asyncio.run(LLM(backend="batches").preflight(CLAUDE))
    assert len(client.live) == 1 and client.created == []


def test_requests_are_chunked(monkeypatch):
    monkeypatch.setenv("HBENCH_JUDGE_BATCH_MAX_REQUESTS", "3")
    client = _client(monkeypatch, FakeClient())
    assert _values(_ask(7)) == list(range(7))
    assert sorted(len(b) for b in client.created) == [1, 3, 3]


@pytest.mark.parametrize("kind", ["errored", "expired", "canceled", "missing", "garbled"])
def test_unsettled_entries_are_rerun_live(monkeypatch, kind):
    def answer(params: dict) -> Any:
        if _user(params) not in ("1", "3"):
            return SimpleNamespace(type="succeeded", message=_echo(params))
        if kind == "missing":
            return None
        if kind == "garbled":
            return SimpleNamespace(type="succeeded", message=_message("not json"))
        return SimpleNamespace(type=kind, error=SimpleNamespace(type="api_error"))

    client = _client(monkeypatch, FakeClient(answer=answer))
    outs = _ask(5)
    assert sorted(_user(kw) for kw in client.live) == ["1", "3"]  # only those two, live
    assert _values(outs) == [0, 1001, 2, 1003, 4]
    assert all(o.usage.errors == 0 for o in outs)


def test_refusal_and_max_tokens_are_the_live_outcomes(monkeypatch):
    def answer(params: dict) -> Any:
        stop = {"0": "refusal", "1": "max_tokens"}.get(_user(params), "end_turn")
        return SimpleNamespace(type="succeeded", message=_message('{"a": 5}', stop))

    client = _client(monkeypatch, FakeClient(answer=answer))
    refused, cut, ok = _ask(3)
    assert refused.result is None and refused.usage.refusals == 1  # the caller's fallback rule
    assert cut.result is None and cut.usage.errors == 1
    assert ok.result == {"a": 5} and client.live == []


def test_a_revoked_token_raises_and_the_next_run_reattaches(monkeypatch, tmp_path):
    monkeypatch.setenv("HBENCH_JUDGE_BATCH_DIR", str(tmp_path / "journal"))
    server = Server()
    first = _client(monkeypatch, FakeClient(server))
    first.fail_retrieve = AuthenticationError("OAuth access token has been revoked.")
    with pytest.raises(JudgeConfigError, match="revoked"):
        _ask(4)
    journal = judge_batches.Journal(tmp_path / "journal")
    assert list(journal.undelivered()) == ["msgbatch_0"] and first.live == []
    second = _client(monkeypatch, FakeClient(server))
    assert _values(_ask(4)) == [0, 1, 2, 3]  # collected: nothing is sent or paid twice
    assert second.created == [] and second.live == [] and journal.undelivered() == {}
    third = _client(monkeypatch, FakeClient(server))
    _ask(4)  # a delivered batch is never re-read: asking again is a new request, as live
    assert len(third.created) == 1


def test_a_failing_batches_api_falls_back_to_live(monkeypatch):
    class BadRequestError(Exception):
        status_code = 400

    client = _client(monkeypatch, FakeClient())

    async def create(**_kw: Any) -> Any:
        raise BadRequestError("batches are not enabled")

    client.messages.batches.create = create
    assert _values(_ask(3)) == [1000, 1001, 1002]


def test_with_effort_copies_share_the_queue(monkeypatch):
    client = _client(monkeypatch, FakeClient())

    async def go() -> None:
        llm = LLM(backend="batches", effort="low")
        medium = llm.with_effort("medium")
        await asyncio.gather(
            llm.call(model=CLAUDE, system="judge", user="1", schema=SCHEMA),
            medium.call(model=CLAUDE, system="judge", user="2", schema=SCHEMA),
        )
        await llm.aclose()

    asyncio.run(go())
    assert len(client.created) == 1
    assert sorted(r["params"]["output_config"]["effort"] for r in client.created[0]) == [
        "low",
        "medium",
    ]


def test_injected_anthropic_client(monkeypatch):
    monkeypatch.undo()  # the real api_key / _make_client
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = FakeClient()
    L.use_anthropic_client(lambda: client)
    try:
        assert _values(_ask(2, backend="live")) == [1000, 1001] and client.closed
    finally:
        L.use_anthropic_client(None)


# ---------------------------------------------------------------- the pipeline's cache


def _pipeline_reply(params: dict) -> Any:
    """Answer a real pipeline prompt as conftest's FakeLLM would."""
    system = params["system"][0]["text"]
    if system == "Reply JSON.":  # the preflight
        return _message('{"a": 1}')
    return _message(json.dumps(FakeLLM().responders[stage_of(system)](_user(params))))


def _judge(readouts: Path, out: Path, backend: str, limit: int | None = None) -> None:
    async def go() -> None:
        llm = LLM(rpm=6e6, concurrency=8, backend=backend)
        try:
            await wrun.run_arms_async(
                [("toy", readouts)],
                out,
                wrun.Models(),
                layers=[44],
                limit=limit,
                rpm=6e6,
                concurrency=8,
                allow_missing=True,
                llm=llm,
            )
        finally:
            await llm.aclose()

    asyncio.run(go())


def _cache_rows(out: Path) -> set[tuple[str, str, str]]:
    path = wrun.arm_dir(out, wrun.Models(), "toy") / "cache.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    return {(r["key"], r["fingerprint"], json.dumps(r["result"], sort_keys=True)) for r in rows}


def _cells(out: Path) -> list[dict]:
    path = wrun.arm_dir(out, wrun.Models(), "toy") / "cells.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_pipeline_cache_is_the_same_and_a_run_can_switch_backends(
    monkeypatch, tmp_path, prose_readouts
):
    live, batched, mixed = (tmp_path / n for n in ("live", "batched", "mixed"))
    client = _client(monkeypatch, FakeClient(reply=_pipeline_reply))
    _judge(prose_readouts, live, "live")
    assert client.created == [] and client.live
    client = _client(monkeypatch, FakeClient(reply=_pipeline_reply))
    _judge(prose_readouts, batched, "batches")
    assert len(client.live) == 1  # the preflight; every judge call went through a batch
    assert len(client.created) >= 2  # one batch per stage round
    assert _cache_rows(batched) == _cache_rows(live) and len(_cache_rows(live)) > 15
    assert _cells(batched) == _cells(live)
    # started live (5 cells), finished with batches: only uncached calls are batched
    _judge(prose_readouts, mixed, "live", limit=5)
    done = len(_cache_rows(mixed))
    client = _client(monkeypatch, FakeClient(reply=_pipeline_reply))
    _judge(prose_readouts, mixed, "batches")
    assert len(client.live) == 1  # the preflight
    assert sum(len(b) for b in client.created) == len(_cache_rows(live)) - done
    assert _cache_rows(mixed) == _cache_rows(live) and _cells(mixed) == _cells(live)
    # ... and a batches-filled cache answers a live run without a call
    client = _client(monkeypatch, FakeClient(reply=_pipeline_reply))
    _judge(prose_readouts, batched, "live")
    assert len(client.live) == 1 and client.created == []  # the preflight only
