"""Explicitly enabled hosted audit, independent credential quotas and private cache."""

import hashlib
import json
import re
from contextlib import contextmanager
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .models import GROQ_ENDPOINT, GROQ_MODEL, ModelError, NoRedirect, parse_object
from .storage import outside_repository
from .working import valid


def seconds(value):
    """Groq reset durations, Retry-After seconds, or HTTP-date."""
    if not value:
        return 0
    try:
        return max(0, float(value))
    except ValueError:
        pass
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)", value)
    if parts and "".join(n + unit for n, unit in parts) == value:
        factors = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}
        return sum(float(n) * factors[unit] for n, unit in parts)
    try:
        return max(0, parsedate_to_datetime(value).timestamp() - time())
    except (ValueError, TypeError, OverflowError):
        return 0


def atomic_json(path, value):
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    temporary.replace(path)


@contextmanager
def state_lock(path):
    with path.open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if __import__("os").name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ModelError("hosted_verifier_busy") from None
        try:
            yield
        finally:
            handle.seek(0)
            if __import__("os").name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class GroqVerifier:
    hosted = True

    def __init__(self, key_files, data_dir, model=GROQ_MODEL, daily_calls=200):
        self.endpoint, self.model = GROQ_ENDPOINT, model
        self.directory = outside_repository(Path(data_dir)) / "groq-verifier"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache = self.directory / "cache"
        self.cache.mkdir(exist_ok=True)
        self.credentials = []
        seen = set()
        for position, path in enumerate(key_files, 1):
            secret = Path(path).read_text(encoding="utf-8-sig").strip()
            if not re.fullmatch(r"gsk_[A-Za-z0-9_-]{20,}", secret):
                raise ValueError("A configured Groq key file has an invalid format.")
            identity = hashlib.sha256(secret.encode()).hexdigest()
            if identity in seen:
                raise ValueError(
                    "Duplicate Groq key files do not add an independent quota."
                )
            seen.add(identity)
            self.credentials.append((identity, f"organization_{position}", secret))
        if not self.credentials or daily_calls < 1:
            raise ValueError("Configure Groq key files and a positive daily call budget.")
        self.daily_calls = daily_calls
        self.last_response, self.last_usage, self.last_events = None, None, []

    def run(self, instruction, payload, timeout, schema=None, max_output_tokens=None):
        self.last_response, self.last_usage, self.last_events = None, None, []
        if schema is None:
            raise ModelError("hosted_schema_required")
        key = hashlib.sha256(
            json.dumps(
                {
                    "model": self.model,
                    "instruction": instruction,
                    "payload": payload,
                    "schema": schema,
                    "protocol": "groq-audit-v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        with state_lock(self.directory / "state.lock"):
            cached = self.cache / (key + ".json")
            if cached.exists():
                try:
                    entry = json.loads(cached.read_text(encoding="utf-8"))
                    if entry["expires"] > time() and valid(entry["response"], schema):
                        self.last_response = entry["raw_response"]
                        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
                        self.last_events = [
                            {"status": "cache_hit", "request_sha256": key}
                        ]
                        return entry["response"]
                except (ValueError, KeyError, TypeError):
                    pass
            state_path = self.directory / "state.json"
            try:
                state = (
                    json.loads(state_path.read_text(encoding="utf-8"))
                    if state_path.exists()
                    else {}
                )
            except (ValueError, OSError):
                raise ModelError("hosted_state_unreadable") from None
            if (
                not isinstance(state, dict)
                or type(state.get("calls", 0)) is not int
                or state.get("calls", 0) < 0
                or not isinstance(state.get("credentials", {}), dict)
                or any(
                    not isinstance(slot, dict)
                    for slot in state.get("credentials", {}).values()
                )
            ):
                raise ModelError("hosted_state_unreadable")
            day = datetime.now(UTC).date().isoformat()
            if state.get("day") != day:
                state["day"], state["calls"] = day, 0
            state.setdefault("credentials", {})
            deadline = monotonic() + timeout
            last_error = "hosted_quota_exhausted"
            for identity, label, secret in self.credentials:
                if state["calls"] >= self.daily_calls:
                    raise ModelError("hosted_daily_budget_exhausted")
                slot = state["credentials"].setdefault(identity + ":" + self.model, {})
                if slot.get("disabled") or slot.get("retry_at", 0) > time():
                    self.last_events.append(
                        {
                            "credential": label,
                            "status": "skipped",
                            "retry_at": slot.get("retry_at"),
                        }
                    )
                    continue
                remaining = deadline - monotonic()
                if remaining < 1:
                    raise ModelError("hosted_timeout")
                body = {
                    "model": self.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": instruction},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                    "max_completion_tokens": 1024,
                    "reasoning_effort": "low",
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "document_audit",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                }
                request = Request(
                    self.endpoint,
                    data=json.dumps(body).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + secret,
                        "User-Agent": "drishti-evidence-guard/0.11.0",
                    },
                    method="POST",
                )
                event = {"credential": label, "status": "started", "network_call": True}
                self.last_events.append(event)
                state["calls"] += 1
                atomic_json(state_path, state)
                try:
                    with build_opener(ProxyHandler({}), NoRedirect()).open(
                        request, timeout=remaining
                    ) as response:
                        headers = response.headers
                        value = json.load(response)
                    choice = value["choices"][0]
                    self.last_response = choice["message"]["content"]
                    self.last_usage = value.get("usage")
                    event.update(status="completed", request_id=value.get("id"))
                    event["quota"] = self._quota(headers)
                    slot.update(event["quota"])
                    if slot.get("remaining_requests") == "0":
                        slot["retry_at"] = time() + seconds(slot.get("reset_requests"))
                    elif slot.get("remaining_tokens") == "0":
                        slot["retry_at"] = time() + seconds(slot.get("reset_tokens"))
                    atomic_json(state_path, state)
                    if choice.get("finish_reason") == "length":
                        raise ModelError("hosted_output_truncated")
                    result = parse_object(self.last_response)
                    if not valid(result, schema):
                        raise ModelError("hosted_invalid_schema")
                    atomic_json(
                        cached,
                        {
                            "expires": time() + 86400,
                            "response": result,
                            "raw_response": self.last_response,
                        },
                    )
                    return result
                except HTTPError as exc:
                    event.update(
                        status="error", code=exc.code, quota=self._quota(exc.headers)
                    )
                    slot.update(event["quota"])
                    if exc.code == 429:
                        delay = seconds(exc.headers.get("retry-after"))
                        if slot.get("remaining_requests") == "0":
                            delay = max(delay, seconds(slot.get("reset_requests")))
                        if slot.get("remaining_tokens") == "0":
                            delay = max(delay, seconds(slot.get("reset_tokens")))
                        slot["retry_at"] = time() + max(delay, 1)
                        last_error = "hosted_quota_exhausted"
                    elif exc.code in {401, 403}:
                        slot["disabled"] = True
                        last_error = "hosted_credentials_unavailable"
                    elif exc.code >= 500:
                        slot["retry_at"] = time() + 5
                        last_error = "hosted_provider_unavailable"
                    else:
                        raise ModelError(f"hosted_http_{exc.code}") from None
                    atomic_json(state_path, state)
                except TimeoutError:
                    event.update(status="error", code="timeout")
                    raise ModelError("hosted_timeout") from None
                except ModelError as exc:
                    event.update(status="error", code=str(exc))
                    raise
                except (URLError, OSError, ValueError, KeyError, TypeError, IndexError):
                    event.update(status="error", code="transport_or_invalid_response")
                    raise ModelError("hosted_transport_or_invalid_response") from None
            raise ModelError(last_error)

    @staticmethod
    def _quota(headers):
        values = {}
        for short, name in [
            ("remaining_requests", "x-ratelimit-remaining-requests"),
            ("remaining_tokens", "x-ratelimit-remaining-tokens"),
            ("reset_requests", "x-ratelimit-reset-requests"),
            ("reset_tokens", "x-ratelimit-reset-tokens"),
        ]:
            raw = headers.get(name)
            if raw is not None and re.fullmatch(r"[0-9a-zA-Z. -]{1,40}", raw):
                values[short] = raw
        return values
