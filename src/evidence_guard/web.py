"""Loopback web interface for the confidential search and synthetic demo."""

import json
import os
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Lock

from .demo import CASES, run_demo

ASSETS = {
    "/": ("index.html", "text/html"),
    "/app.css": ("app.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}


def serve(port: int, ask: Callable[..., dict], data_dir: Path, enable_live: bool) -> None:
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
            self.send_header("Content-Type", f"{mime}; charset=utf-8")
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

        def stream_search(self, question: str) -> None:
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
                result = ask(
                    question,
                    on_event=lambda event: emit({"type": "event", "event": event}),
                )
            except Exception:
                emit({"type": "error", "error": "Run failed. Check the local model."})
            else:
                emit({"type": "result", "result": result})
            self.close_connection = True

        def do_GET(self):
            if self.headers.get("Host") != authority:
                self.reply(403, {"error": "Open the exact loopback URL."})
            elif self.path in assets:
                value, mime = assets[self.path]
                self.send(200, value, mime)
            elif self.path == "/api/config":
                self.reply(200, {"live_enabled": live_enabled})
            else:
                self.reply(404, {"error": "Page unavailable."})

        def do_POST(self):
            if (
                self.headers.get("Host") != authority
                or self.headers.get("Origin") != origin
                or self.headers.get_content_type() != "application/json"
            ):
                self.reply(403, {"error": "Request must come from the local interface."})
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
                    self.stream_search(question)
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
