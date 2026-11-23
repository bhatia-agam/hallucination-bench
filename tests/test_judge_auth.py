"""Credentials of the Anthropic judge route: what happens with none at all."""


def test_missing_credentials_fail_at_once(monkeypatch):
    """No key and no `ant` profile: the SDK raises a TypeError before any request. That is a
    config error (exit 3) on the first call, not a failure to retry for half an hour."""
    import asyncio

    import pytest

    from hallucination_bench import llm as L

    class _Messages:
        async def create(self, **_kw):
            raise TypeError(
                '"Could not resolve authentication method. Expected one of api_key, '
                'auth_token, or credentials to be set."'
            )

    class _Client:
        messages = _Messages()

    monkeypatch.setattr(L, "_make_client", lambda r, k: _Client())
    monkeypatch.setattr(L, "api_key", lambda m: None)
    schema = L.schema_block("t", {"a": {"type": "integer"}}, ["a"])
    with pytest.raises(L.JudgeConfigError, match="ANTHROPIC_API_KEY"):
        asyncio.run(L.LLM().call(model="claude-opus-5-5", system="s", user="u", schema=schema))
