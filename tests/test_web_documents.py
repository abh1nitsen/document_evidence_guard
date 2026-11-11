"""Real local HTTP tests for upload, selection and immutable source links."""

import json
import socket
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode

import pytest
from test_documents import pdf_bytes

from evidence_guard import web
from evidence_guard.documents import DocumentStore
from evidence_guard.telemetry import Trace
from evidence_guard.workflow import local_answer


@pytest.fixture
def library_server(tmp_path, monkeypatch, request):
    store = DocumentStore(tmp_path)
    ready = threading.Event()
    servers = []
    requests = []
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def server_factory(address, handler):
        server = ThreadingHTTPServer(address, handler)
        servers.append(server)
        ready.set()
        return server

    def ask(
        question,
        topic=None,
        context=None,
        document_id=None,
        on_event=None,
        role_assignment=None,
        local_only=False,
    ):
        requests.append((document_id, topic, context))
        index, _ = store.get_index(document_id)
        trace = Trace(tmp_path, "synthetic", on_event)
        result = local_answer(question, index, None, trace, 1)
        return {
            **result,
            "document": store.metadata(document_id),
            "events": trace.events,
            "role_assignment": role_assignment,
            "local_only": local_only,
        }

    monkeypatch.setattr(web, "ThreadingHTTPServer", server_factory)
    thread = threading.Thread(
        target=web.serve,
        args=(port, ask, tmp_path, False),
        kwargs={
            "documents": store,
            "model_roles": {
                "default": "primary",
                "primary": {"answer": "qwen", "verifier": "gemma"},
                "reversed": {"answer": "gemma", "verifier": "qwen"},
            },
            "local_verification": {"answer": "qwen", "verifier": "gemma"}
            if getattr(request, "param", True)
            else None,
        },
        daemon=True,
    )
    thread.start()
    assert ready.wait(5)

    def request(method, path, body=None, mime="application/json", origin=True):
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        headers = {"Content-Type": mime}
        if origin:
            headers["Origin"] = f"http://127.0.0.1:{port}"
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        status, raw = response.status, response.read()
        connection.close()
        return status, raw

    yield store, request, requests
    servers[0].shutdown()
    thread.join(5)


def upload(request, text, name):
    status, raw = request(
        "POST",
        "/api/documents?" + urlencode({"name": name}),
        pdf_bytes(text),
        "application/pdf",
    )
    assert status == 200
    return json.loads(raw)["document"]


def test_upload_search_and_source_are_bound_to_requested_document(library_server):
    _, request, requests = library_server
    a = upload(
        request, "1.1 Reserve\nReserve is a holding period of three days.", "A.pdf"
    )
    b = upload(request, "1.1 Reserve\nReserve is a holding period of nine days.", "B.pdf")
    for doc, expected, other in (
        (a, "three days", "nine days"),
        (b, "nine days", "three days"),
    ):
        status, raw = request(
            "POST",
            "/api/search",
            json.dumps(
                {
                    "question": "What is reserve?",
                    "document_id": doc["id"],
                    "topic": None,
                    "context": None,
                }
            ),
        )
        assert status == 200
        result = next(
            json.loads(line)["result"]
            for line in raw.splitlines()
            if json.loads(line)["type"] == "result"
        )
        assert result["document"]["id"] == doc["id"]
        text = " ".join(c["text"] for c in result["claims"])
        assert expected in text
        assert other not in text
        status, source = request("GET", "/api/document?document_id=" + doc["id"])
        assert status == 200
        assert source.startswith(b"%PDF-")
    assert requests == [(a["id"], None, None), (b["id"], None, None)]


def test_search_requires_explicit_document_and_upload_requires_origin(library_server):
    _, request, _ = library_server
    upload(request, "An unrelated manual.", "Manual.pdf")
    status, _ = request("POST", "/api/search", json.dumps({"question": "Explain this"}))
    assert status == 400
    status, _ = request(
        "POST",
        "/api/documents?name=Other.pdf",
        pdf_bytes("Other content."),
        "application/pdf",
        origin=False,
    )
    assert status == 403
    status, _ = request("GET", "/api/document?document_id=../not-a-document")
    assert status == 404


def test_scan_rejection_keeps_existing_library(library_server):
    store, request, _ = library_server
    good = upload(request, "The valid document remains available.", "Good.pdf")
    status, raw = request(
        "POST", "/api/documents?name=Scan.pdf", pdf_bytes(), "application/pdf"
    )
    assert status == 422
    assert "OCR" in json.loads(raw)["error"]
    assert [d["id"] for d in store.list()] == [good["id"]]


def test_role_assignment_is_exposed_validated_and_forwarded(library_server):
    _, request, _ = library_server
    doc = upload(request, "1.1 Reserve\nReserve is a holding period.", "Roles.pdf")
    status, raw = request("GET", "/api/config")
    assert status == 200
    assert json.loads(raw)["model_roles"]["primary"]["verifier"] == "gemma"
    for assignment in ("primary", "reversed"):
        status, raw = request(
            "POST",
            "/api/search",
            json.dumps(
                {
                    "question": "What is reserve?",
                    "document_id": doc["id"],
                    "role_assignment": assignment,
                }
            ),
        )
        assert status == 200
        result = next(
            json.loads(line)["result"]
            for line in raw.splitlines()
            if json.loads(line)["type"] == "result"
        )
        assert result["role_assignment"] == assignment
    for assignment in ("unknown", [], {"url": "http://outside.example"}):
        status, _ = request(
            "POST",
            "/api/search",
            json.dumps(
                {
                    "question": "What is reserve?",
                    "document_id": doc["id"],
                    "role_assignment": assignment,
                }
            ),
        )
        assert status == 400


def test_local_only_is_a_strict_boolean_and_forwarded_before_search(library_server):
    _, request, requests = library_server
    doc = upload(request, "1.1 Reserve\nReserve is a holding period.", "Confidential.pdf")
    status, raw = request("GET", "/api/config")
    config = json.loads(raw)
    assert status == 200
    assert config["local_verification"]["verifier"] == "gemma"
    assert "less reliable" in config["local_verification_warning"]
    for flag in (True, False):
        status, raw = request(
            "POST",
            "/api/search",
            json.dumps(
                {
                    "question": "What is reserve?",
                    "document_id": doc["id"],
                    "local_only": flag,
                }
            ),
        )
        assert status == 200
        result = next(
            json.loads(line)["result"]
            for line in raw.splitlines()
            if json.loads(line)["type"] == "result"
        )
        assert result["local_only"] is flag
    assert len(requests) == 2

    for invalid in ("false", "true", 0, 1, None, [], {}):
        status, _ = request(
            "POST",
            "/api/search",
            json.dumps(
                {
                    "question": "Who is the organizer?",
                    "document_id": doc["id"],
                    "local_only": invalid,
                }
            ),
        )
        assert status == 400
    assert len(requests) == 2


@pytest.mark.parametrize("library_server", [False], indirect=True)
def test_unavailable_local_override_is_rejected_before_any_search(library_server):
    _, request, requests = library_server
    doc = upload(request, "1.1 Reserve\nReserve is a holding period.", "Manual.pdf")
    status, _ = request(
        "POST",
        "/api/search",
        json.dumps(
            {
                "question": "What is reserve?",
                "document_id": doc["id"],
                "local_only": True,
            }
        ),
    )
    assert status == 400
    assert requests == []
