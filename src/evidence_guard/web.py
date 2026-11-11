"""Loopback web interface for the confidential search and synthetic demo."""

import json
import os
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from .demo import CASES, run_demo
from .documents import MAX_UPLOAD_BYTES, DocumentStore
from .evaluation import source_fingerprint
from .working import CONTRACT as CONTRACT_VERSION

ASSETS = {
    "/": ("index.html", "text/html"),
    "/app.css": ("app.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
LOCAL_VERIFICATION_WARNING = (
    "Local verification may be less reliable than the larger Groq verifier. "
    "It can miss errors or reject valid evidence. Review the source excerpts."
)


def serve(
    port: int,
    ask: Callable[..., dict],
    data_dir: Path,
    enable_live: bool,
    document: Path | None = None,
    contract: str = CONTRACT_VERSION,
    documents: DocumentStore | None = None,
    loop_profile: str = "fast",
    model_roles: dict | None = None,
    hosted_verification: bool = False,
    local_verification: dict | None = None,
) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    authority = f"127.0.0.1:{port}"
    origin = f"http://{authority}"
    live_enabled = enable_live and bool(os.environ.get("GROQ_API_KEY", "").strip())
    busy = Lock()
    assets = {
        route: (files(__package__).joinpath("static", name).read_bytes(), mime)
        for route, (name, mime) in ASSETS.items()
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # The workflow already records metadata. Keep HTTP inputs out of logs.
            pass

        def send(self, status: int, value: bytes, mime: str) -> None:
            self.send_response(status)
            self.send_header(
                "Content-Type",
                mime if mime == "application/pdf" else f"{mime}; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(value)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; object-src 'none'; base-uri 'none'; "
                "frame-ancestors 'none'; form-action 'self'",
            )
            self.end_headers()
            self.wfile.write(value)

        def reply(self, status: int, value) -> None:
            self.send(status, json.dumps(value).encode("utf-8"), "application/json")

        def stream_search(
            self,
            question: str,
            topic: str | None,
            context=None,
            document_id=None,
            role_assignment=None,
            local_only=False,
        ) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            connected = True

            def emit(value):
                nonlocal connected
                if connected:
                    try:
                        self.wfile.write((json.dumps(value) + "\n").encode("utf-8"))
                        self.wfile.flush()
                    except OSError:
                        connected = False

            try:
                arguments = {
                    "topic": topic,
                    "context": context,
                    "on_event": lambda event: emit({"type": "event", "event": event}),
                }
                if documents is not None:
                    arguments["document_id"] = document_id
                if model_roles is not None:
                    arguments["role_assignment"] = role_assignment
                if local_only:
                    arguments["local_only"] = True
                result = ask(question, **arguments)
            except Exception:
                emit({"type": "error", "error": "Run failed. Check the local model."})
            else:
                emit({"type": "result", "result": result})
            self.close_connection = True

        def do_GET(self):
            request = urlsplit(self.path)
            if self.headers.get("Host") != authority:
                self.reply(403, {"error": "Open the exact loopback URL."})
            elif request.path in assets:
                value, mime = assets[request.path]
                self.send(200, value, mime)
            elif request.path == "/api/documents" and documents is not None:
                self.reply(
                    200,
                    {"documents": documents.list(), "default_id": documents.default_id},
                )
            elif request.path == "/api/document" and documents is not None:
                try:
                    document_id = parse_qs(request.query).get("document_id", [None])[0]
                    raw = documents.read_pdf(document_id)
                except (ValueError, OSError):
                    self.reply(
                        404, {"error": "Choose a document from the local library."}
                    )
                    return
                self.send(200, raw, "application/pdf")
            elif request.path == "/api/document" and document is not None:
                self.send(200, document.read_bytes(), "application/pdf")
            elif request.path == "/api/config":
                summary = None
                evaluation_info = None
                reports = sorted(
                    (data_dir / "evaluations").glob("*.json"),
                    key=lambda path: path.stat().st_mtime,
                )
                if reports:
                    try:
                        report = json.loads(reports[-1].read_text(encoding="utf-8"))
                        summary = report["summary"]
                        manifest = report.get("manifest", {})
                        evaluation_info = {
                            "version": manifest.get("package", "unrecorded"),
                            "current_source": manifest.get("source_tree_sha256")
                            == source_fingerprint(),
                            "created_utc": report.get("created_utc"),
                        }
                    except (OSError, ValueError, KeyError):
                        pass
                self.reply(
                    200,
                    {
                        "live_enabled": live_enabled,
                        "version": version("drishti-evidence-guard"),
                        "contract": contract,
                        "evaluation": summary if documents is None else None,
                        "evaluation_info": evaluation_info,
                        "documents": documents.list() if documents else None,
                        "default_document_id": documents.default_id
                        if documents
                        else None,
                        "max_upload_bytes": MAX_UPLOAD_BYTES,
                        "loop_profile": loop_profile,
                        "model_roles": model_roles,
                        "hosted_verification": hosted_verification,
                        "local_verification": local_verification,
                        "local_verification_warning": LOCAL_VERIFICATION_WARNING,
                    },
                )
            else:
                self.reply(404, {"error": "Page unavailable."})

        def do_POST(self):
            request = urlsplit(self.path)
            if (
                self.headers.get("Host") != authority
                or self.headers.get("Origin") != origin
            ):
                self.reply(403, {"error": "Request must come from the local interface."})
                return
            if request.path == "/api/documents" and documents is not None:
                if self.headers.get_content_type() != "application/pdf":
                    self.reply(415, {"error": "Upload a PDF file."})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 1 <= length <= MAX_UPLOAD_BYTES:
                        raise ValueError("PDF uploads must be 25 MB or smaller.")
                except ValueError:
                    self.reply(413, {"error": "PDF uploads must be 25 MB or smaller."})
                    return
                if not busy.acquire(blocking=False):
                    self.reply(409, {"error": "A run or upload is already in progress."})
                    return
                try:
                    self.connection.settimeout(15)
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError("The upload was incomplete. Try again.")
                    name = parse_qs(request.query).get("name", ["document.pdf"])[0]
                    result = documents.upload(raw, name)
                except ValueError as exc:
                    self.reply(422, {"error": str(exc)})
                except Exception:
                    self.reply(500, {"error": "The PDF could not be indexed locally."})
                else:
                    self.reply(200, result)
                finally:
                    self.connection.settimeout(None)
                    busy.release()
                return
            if self.headers.get_content_type() != "application/json":
                self.reply(415, {"error": "Use a JSON request from the local interface."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 8192:
                    raise ValueError
                self.connection.settimeout(5)
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError
            except (ValueError, OSError):
                self.reply(400, {"error": "Invalid request."})
                return
            finally:
                self.connection.settimeout(None)

            if not busy.acquire(blocking=False):
                self.reply(409, {"error": "A run is already in progress."})
                return
            try:
                if self.path == "/api/search":
                    question = payload.get("question")
                    if (
                        not isinstance(question, str)
                        or not 1 <= len(question.strip()) <= 1500
                    ):
                        self.reply(
                            400, {"error": "Enter a question up to 1500 characters."}
                        )
                        return
                    topic = payload.get("topic")
                    if topic is not None and (
                        not isinstance(topic, str) or not 1 <= len(topic) <= 120
                    ):
                        self.reply(400, {"error": "Invalid conversation subject."})
                        return
                    context = payload.get("context")
                    if context is not None and (
                        not isinstance(context, str) or not 1 <= len(context) <= 3000
                    ):
                        self.reply(
                            400, {"error": "Context must have 1 to 3000 characters."}
                        )
                        return
                    document_id = payload.get("document_id")
                    if documents is not None:
                        try:
                            documents.metadata(document_id)
                        except ValueError as exc:
                            self.reply(400, {"error": str(exc)})
                            return
                    assignment = payload.get("role_assignment")
                    if assignment is not None and (
                        not isinstance(assignment, str)
                        or model_roles is None
                        or assignment not in {"primary", "reversed"}
                        or assignment not in model_roles
                    ):
                        self.reply(400, {"error": "Invalid model role assignment."})
                        return
                    local_only = payload.get("local_only", False)
                    if type(local_only) is not bool:
                        self.reply(
                            400,
                            {"error": "Local-only verification must be true or false."},
                        )
                        return
                    if local_only and local_verification is None:
                        self.reply(
                            400,
                            {
                                "error": "Local-only verification is unavailable "
                                "in this configuration."
                            },
                        )
                        return
                    self.stream_search(
                        question, topic, context, document_id, assignment, local_only
                    )
                    return
                elif self.path == "/api/demo":
                    live = payload.get("live") is True
                    if live and not live_enabled:
                        self.reply(403, {"error": "Live synthetic demo is disabled."})
                        return
                    cases = {case.name: case.question for case in CASES}
                    result = run_demo(data_dir, "cascade", live)
                    for row in result:
                        row["question"] = cases[row["case"]]
                else:
                    self.reply(404, {"error": "Unknown endpoint."})
                    return
            except Exception:
                self.reply(
                    500, {"error": "Run failed. Check the PDF and local model server."}
                )
                return
            finally:
                busy.release()
            self.reply(200, result)

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Evidence Guard is ready at {origin}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
