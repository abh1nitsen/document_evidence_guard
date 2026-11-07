"""JSON chat adapters. The document workflow accepts a loopback server only."""

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"


class ModelError(Exception):
    """Sanitized model failure. Provider bodies can contain prompt data."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class ChatResult:
    content: str
    usage: dict | None
    request_id: str | None


def chat(
    endpoint: str,
    model: str,
    messages: list[dict],
    timeout: float,
    key: str | None = None,
    schema: dict | None = None,
) -> ChatResult:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    body["max_completion_tokens" if key else "max_tokens"] = 768 if key else 640
    if schema is not None:
        body["response_format"]["schema"] = schema
    if key:
        body["reasoning_effort"] = "low"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "drishti-evidence-guard/0.1.2",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(
        endpoint, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(
            request, timeout=timeout
        ) as response:
            value = json.load(response)
        choice = value["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ModelError("model_output_truncated")
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("missing text")
        return ChatResult(content, value.get("usage"), value.get("id"))
    except HTTPError as exc:
        raise ModelError(f"http_{exc.code}") from None
    except TimeoutError:
        raise ModelError("model_timeout") from None
    except (URLError, OSError, ValueError, KeyError, TypeError, IndexError):
        raise ModelError("transport_or_invalid_response") from None


def parse_object(content: str) -> dict:
    try:
        value = json.loads(content)
    except ValueError:
        raise ModelError("invalid_model_json") from None
    if not isinstance(value, dict):
        raise ModelError("invalid_model_json")
    return value


class LocalModel:
    def __init__(self, url: str, model: str):
        parts = urlsplit(url)
        if (
            parts.scheme != "http"
            or parts.hostname not in {"127.0.0.1", "::1"}
            or parts.username is not None
            or parts.query
            or parts.fragment
            or parts.path.rstrip("/") != "/v1"
        ):
            raise ValueError("Local model URL must be http://127.0.0.1:PORT/v1.")
        self.endpoint = url.rstrip("/") + "/chat/completions"
        self.model = model

    def run(
        self,
        instruction: str,
        payload: dict,
        timeout: float,
        schema: dict | None = None,
    ) -> dict:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        return parse_object(
            chat(self.endpoint, self.model, messages, timeout, schema=schema).content
        )
