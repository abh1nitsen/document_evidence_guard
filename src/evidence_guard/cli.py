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
    ask.add_argument("--pdf", type=Path, required=True)
    ask.add_argument("--local-url", default="http://127.0.0.1:8080/v1")
    ask.add_argument("--local-model", default="qwen2.5-1.5b")
    ask.add_argument("--seconds", type=float, default=90)
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
        if args.command == "ask":
            pdf = outside_repository(args.pdf)
            model = LocalModel(args.local_url, args.local_model)
            question = (args.question or input("Question: ")).strip()
            if not 1 <= len(question) <= 1500:
                raise ValueError("Question must have 1 to 1500 characters.")
            if not math.isfinite(args.seconds) or args.seconds <= 0:
                raise ValueError("Local seconds must be finite and positive.")
            trace = Trace(data_dir, "confidential_local")
            trace.log("start", parser="pypdf_layout", model=args.local_model)
            start = perf_counter()
            index, hit = load_pdf(pdf, data_dir)
            trace.log(
                "index_load",
                start,
                cache_hit=hit,
                pages=len(index.pages),
                document_sha256=index.fingerprint,
            )
            result = local_answer(question, index, model, trace, args.seconds)
            trace.log("finish", route=result["route"])
            result.update(trace_id=trace.id, events=trace.events)
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
