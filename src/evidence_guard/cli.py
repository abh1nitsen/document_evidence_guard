"""Portable command line entry points."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

from .demo import run_demo
from .evidence import load_pdf
from .models import LocalModel
from .telemetry import Trace, summarize
from .web import serve
from .workflow import local_answer


def outside_repository(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if any(
        (parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
        for parent in (resolved, *resolved.parents)
    ):
        raise ValueError("PDF inputs and runtime data must be outside any Git checkout.")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(prog="evidence-guard")
    parser.add_argument("--data-dir", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Confidential local-only PDF question")
    ui = sub.add_parser("ui", help="Start the local browser interface")
    for command in (ask, ui):
        command.add_argument("--pdf", type=Path, required=True)
        command.add_argument("--local-url", default="http://127.0.0.1:8080/v1")
        command.add_argument("--local-model", default="qwen2.5-1.5b")
        command.add_argument("--seconds", type=float, default=90)
    ui.add_argument("--port", type=int, default=8787)
    ui.add_argument("--enable-live-demo", action="store_true")
    ask.add_argument("question", nargs="?", help="Omit to enter outside shell history")
    demo = sub.add_parser("demo", help="Synthetic-only routing and Groq smoke test")
    demo.add_argument("--mode", choices=("cascade", "baseline"), default="cascade")
    demo.add_argument(
        "--live", action="store_true", help="Enable billable synthetic calls"
    )
    report = sub.add_parser("report", help="Metadata and volume projection")
    report.add_argument("--monthly-questions", type=int, default=100_000)
    report.add_argument("--local-cost-per-question", type=float, required=True)
    report.add_argument("--expected-frontier-rate", type=float)
    args = parser.parse_args()

    try:
        data_dir = outside_repository(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        if args.command in {"ask", "ui"}:
            pdf = outside_repository(args.pdf)
            model = LocalModel(args.local_url, args.local_model)
            if not math.isfinite(args.seconds) or args.seconds <= 0:
                raise ValueError("Local seconds must be finite and positive.")

            def ask_pdf(question: str, on_event=None) -> dict:
                question = question.strip()
                if not 1 <= len(question) <= 1500:
                    raise ValueError("Question must have 1 to 1500 characters.")
                trace = Trace(data_dir, "confidential_local", on_event=on_event)
                trace.log("start", parser="pypdf_layout", model=model.model)
                start = perf_counter()
                trace.activity(
                    "context",
                    "running",
                    "Load the report index",
                    "Resolve the active document and its source pages.",
                )
                index, hit = load_pdf(pdf, data_dir)
                trace.log(
                    "index_load",
                    start,
                    cache_hit=hit,
                    pages=len(index.pages),
                    document_sha256=index.fingerprint,
                )
                trace.activity(
                    "context",
                    "completed",
                    "Source pages ready",
                    "The local index is available for retrieval.",
                    pages_total=len(index.pages),
                )
                answer = local_answer(question, index, model, trace, args.seconds)
                trace.log("finish", route=answer["route"])
                return dict(answer, trace_id=trace.id, events=trace.events)

            if args.command == "ui":
                serve(args.port, ask_pdf, data_dir, args.enable_live_demo)
                return
            result = ask_pdf(args.question or input("Question: "))
        elif args.command == "demo":
            result = run_demo(data_dir, args.mode, args.live)
        else:
            rate = args.expected_frontier_rate
            local = args.local_cost_per_question
            if (
                args.monthly_questions <= 0
                or not math.isfinite(local)
                or local < 0
                or (rate is not None and not 0 <= rate <= 1)
            ):
                raise ValueError("Use positive volume, nonnegative cost, and rate 0–1.")
            result = summarize(data_dir, args.monthly_questions, local, rate)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2))
