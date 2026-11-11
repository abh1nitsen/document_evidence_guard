"""Quota failover uses synthetic credentials and mocked transport only."""

import io
import json
from urllib.error import HTTPError

import pytest

from evidence_guard import groq_verifier as g
from evidence_guard.extractive import VERIFY_SCHEMA
from evidence_guard.models import ModelError

OK = {"decision": "accept", "reason": "Explicit witness", "recovery_query": ""}


def verifier(tmp_path, **kwargs):
    keys = []
    for i in range(2):
        path = tmp_path / f"key{i}.txt"
        path.write_text("gsk_" + str(i) * 40)
        keys.append(path)
    return g.GroqVerifier(keys, tmp_path / "runtime", **kwargs)


def transport(monkeypatch, outcomes):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            value = next(outcomes)
            if isinstance(value, Exception):
                raise value
            response = io.StringIO(json.dumps(value))
            response.headers = {"x-ratelimit-remaining-requests": "99"}
            return response

    monkeypatch.setattr(g, "build_opener", lambda *a: Opener())
    return calls


def response(content=OK):
    return {
        "id": "synthetic",
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(content)}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
    }


def test_rotation_persistent_cooldown_cache_and_no_secrets(tmp_path, monkeypatch):
    m = verifier(tmp_path)
    calls = transport(
        monkeypatch,
        iter(
            [
                HTTPError(m.endpoint, 429, "private error", {"retry-after": "300"}, None),
                response(),
                response(),
            ]
        ),
    )
    assert m.run("audit", {"case": 1}, 10, schema=VERIFY_SCHEMA) == OK
    assert [e["credential"] for e in m.last_events] == [
        "organization_1",
        "organization_2",
    ]
    assert len(calls) == 2
    assert m.run("audit", {"case": 1}, 10, schema=VERIFY_SCHEMA) == OK
    assert m.last_events[0]["status"] == "cache_hit"
    m = verifier(tmp_path)
    assert m.run("audit", {"case": 2}, 10, schema=VERIFY_SCHEMA) == OK
    assert m.last_events[0]["status"] == "skipped"
    assert len(calls) == 3
    for path in m.directory.rglob("*.json"):
        assert "gsk_" not in path.read_text()
    sent = json.loads(calls[-1].data)
    assert sent["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize("code", [429, 401, 503])
def test_all_credentials_fail_closed(tmp_path, monkeypatch, code):
    m = verifier(tmp_path)
    calls = transport(
        monkeypatch,
        iter(
            [
                HTTPError(m.endpoint, code, "secret provider body", {}, None)
                for _ in range(2)
            ]
        ),
    )
    with pytest.raises(ModelError, match="hosted_") as error:
        m.run("audit", {}, 10, schema=VERIFY_SCHEMA)
    assert "secret" not in str(error.value)
    assert len(calls) == 2
    assert not list(m.cache.glob("*.json"))


def test_invalid_schema_is_not_cached(tmp_path, monkeypatch):
    m = verifier(tmp_path)
    transport(monkeypatch, iter([response({"decision": "accept"})]))
    with pytest.raises(ModelError, match="hosted_invalid_schema"):
        m.run("audit", {}, 10, schema=VERIFY_SCHEMA)
    assert m.last_events[-1]["status"] == "error"
    assert not list(m.cache.glob("*.json"))


def test_daily_cap_does_not_rotate_to_bypass_budget(tmp_path, monkeypatch):
    m = verifier(tmp_path, daily_calls=1)
    calls = transport(monkeypatch, iter([response()]))
    m.run("audit", {"case": 1}, 10, schema=VERIFY_SCHEMA)
    with pytest.raises(ModelError, match="hosted_daily_budget_exhausted"):
        m.run("audit", {"case": 2}, 10, schema=VERIFY_SCHEMA)
    assert len(calls) == 1


@pytest.mark.parametrize("state", [[], {"calls": "oops"}, {"credentials": []}])
def test_invalid_quota_state_fails_closed(tmp_path, monkeypatch, state):
    m = verifier(tmp_path)
    calls = transport(monkeypatch, iter([]))
    (m.directory / "state.json").write_text(json.dumps(state))
    with pytest.raises(ModelError, match="hosted_state_unreadable"):
        m.run("audit", {}, 10, schema=VERIFY_SCHEMA)
    assert calls == []


@pytest.mark.parametrize(
    "value,expected",
    [("1m14s", 74), ("2h3m", 7380), ("500ms", 0.5), ("3", 3), ("bad", 0)],
)
def test_reset_duration(value, expected):
    assert g.seconds(value) == expected
