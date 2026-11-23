"""Credentials of the Anthropic judge route: none at all, and the OAuth bearer-token route."""


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


def test_bearer_token_route(monkeypatch):
    """ANTHROPIC_AUTH_TOKEN with no API key: a bearer-token client with the OAuth beta header,
    and no API key anywhere. An API key, when set, still wins."""
    from hallucination_bench import llm as L

    monkeypatch.setattr(L, "_ANTHROPIC_CLIENT", None, raising=False)  # no injected client
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv(L.TOKEN_ENV, "tok-123")
    assert L.api_key("claude-opus-5-5") is None
    client = L._make_client("anthropic", None)
    assert client.auth_token == "tok-123" and client.api_key is None
    assert client.default_headers["anthropic-beta"] == L.OAUTH_BETA
    assert "X-Api-Key" not in client.auth_headers
    keyed = L._make_client("anthropic", "sk-ant-test")
    assert keyed.api_key == "sk-ant-test"
