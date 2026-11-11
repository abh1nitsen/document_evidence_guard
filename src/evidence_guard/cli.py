"""Portable command line entry points."""

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .demo import run_demo
from .documents import DocumentStore
from .evaluation import evaluate, source_fingerprint
from .evidence import load_pdf
from .groq_verifier import GroqVerifier
from .models import LocalModel
from .storage import outside_repository
from .telemetry import Trace, summarize
from .warmup import prepare
from .web import LOCAL_VERIFICATION_WARNING, serve
from .workflow import local_answer
from .working import save_record


def main() -> None:
    parser = argparse.ArgumentParser(prog="evidence-guard")
    parser.add_argument("--data-dir", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Confidential local-only PDF question")
    ui = sub.add_parser("ui", help="Start the local browser interface")
    evaluation = sub.add_parser(
        "evaluate", help="Run a labeled external evaluation suite"
    )
    evaluation.add_argument("--suite", type=Path, required=True)
    for command in (ask, ui, evaluation):
        command.add_argument("--pdf", type=Path, required=command is not ui)
        command.add_argument("--local-url", default="http://127.0.0.1:8080/v1")
        command.add_argument("--local-model", default="qwen2.5-1.5b")
        command.add_argument(
            "--verifier-url",
            default="http://127.0.0.1:18090/v1",
            help="Loopback verifier server (working workflow)",
        )
        command.add_argument("--verifier-model", default="gemma-2-2b-it")
        command.add_argument(
            "--verifier-provider", choices=("local", "groq"), default="local"
        )
        command.add_argument("--groq-key-file", type=Path, action="append", default=[])
        command.add_argument("--groq-model", default="openai/gpt-oss-120b")
        command.add_argument("--groq-daily-calls", type=int, default=200)
        command.add_argument(
            "--swap-model-roles",
            action="store_true",
            help="Use the verifier model to answer and the answer model to verify",
        )
        command.add_argument("--seconds", type=float, default=120)
        command.add_argument(
            "--loop-profile", choices=("classic", "compact", "fast"), default="fast"
        )
        command.add_argument(
            "--context-file",
            type=Path,
            help="External UTF-8 user context, up to 3000 characters",
        )
        command.add_argument("--planner-url", help="Loopback semantic query planner")
        command.add_argument("--planner-model", default="gemma-2-2b-it")
        command.add_argument(
            "--workflow",
            choices=("extractive", "working", "legacy"),
            default="extractive",
        )
        command.add_argument(
            "--semantic-model-dir",
            type=Path,
            help="Existing offline FastEmbed cache; install [semantic]",
        )
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
        if args.command in {"ask", "ui", "evaluate"}:
            pdf = outside_repository(args.pdf) if args.pdf else None
            documents = DocumentStore(data_dir) if args.command == "ui" else None
            if documents is not None and pdf is not None:
                documents.register_path(pdf)
            model = LocalModel(args.local_url, args.local_model)
            local_verifier = LocalModel(args.verifier_url, args.verifier_model)
            verifier = local_verifier
            hosted = args.verifier_provider == "groq"
            if hosted:
                if args.workflow != "extractive" or args.swap_model_roles:
                    raise ValueError(
                        "Groq requires extractive workflow and primary roles."
                    )
                verifier = GroqVerifier(
                    args.groq_key_file, data_dir, args.groq_model, args.groq_daily_calls
                )
            if args.swap_model_roles and args.workflow == "legacy":
                raise ValueError(
                    "Role swapping requires a working or extractive workflow."
                )
            default_assignment = "reversed" if args.swap_model_roles else "primary"
            role_options = (
                {
                    "default": default_assignment,
                    "primary": {"answer": model.model, "verifier": verifier.model},
                    "reversed": {"answer": verifier.model, "verifier": model.model},
                }
                if args.workflow != "legacy"
                else None
            )
            planner = (
                LocalModel(args.planner_url, args.planner_model)
                if args.planner_url and args.workflow == "legacy"
                else None
            )
            if hosted:
                role_options.pop("reversed", None)
            semantic = None
            if args.semantic_model_dir:
                from .semantic import SemanticSearch

                semantic = SemanticSearch(
                    outside_repository(args.semantic_model_dir), data_dir
                )
            initial_context = None
            if args.context_file:
                initial_context = outside_repository(args.context_file).read_text(
                    encoding="utf-8"
                )
                if not 1 <= len(initial_context) <= 3000:
                    raise ValueError("Context must contain 1 to 3000 characters.")
            if not math.isfinite(args.seconds) or args.seconds <= 0:
                raise ValueError("Local seconds must be finite and positive.")

            def ask_pdf(
                question: str,
                on_event=None,
                topic: str | None = None,
                context: str | None = None,
                document_id: str | None = None,
                role_assignment: str | None = None,
                local_only: bool = False,
            ) -> dict:
                if type(local_only) is not bool:
                    raise ValueError("Local-only verification must be true or false.")
                if local_only and args.workflow != "extractive":
                    raise ValueError(
                        "Local-only override requires the extractive workflow."
                    )
                assignment = role_assignment or default_assignment
                if assignment not in {"primary", "reversed"}:
                    raise ValueError("Unknown model role assignment.")
                if hosted and assignment != "primary":
                    raise ValueError(
                        "Hosted verification cannot be assigned to source selection."
                    )
                answer_model, audit_model = (
                    (verifier, model) if assignment == "reversed" else (model, verifier)
                )
                if local_only and hosted:
                    answer_model, audit_model = model, local_verifier
                provider = "groq" if getattr(audit_model, "hosted", False) else "local"
                selected_pdf = documents.path(document_id) if documents else pdf
                document_info = documents.metadata(document_id) if documents else None
                if documents is None or document_id == documents.default_id:
                    context = context or initial_context
                question = question.strip()
                if not 1 <= len(question) <= 1500:
                    raise ValueError("Question must have 1 to 1500 characters.")
                trace = Trace(
                    data_dir,
                    "evaluation_v3"
                    if args.command == "evaluate"
                    else "confidential_local_v2",
                    on_event=on_event,
                )
                source_hash = source_fingerprint()
                trace.log(
                    "start",
                    parser="pypdf_layout",
                    model=answer_model.model,
                    verifier_model=audit_model.model if role_options else None,
                    role_assignment=assignment if role_options else None,
                    local_only=local_only,
                    verifier_provider=provider,
                    model_identity="server_alias_only",
                    budget_seconds=args.seconds,
                    planner_model=planner.model if planner else None,
                    workflow=args.workflow,
                    loop_profile=args.loop_profile,
                    package_version=version("drishti-evidence-guard"),
                )
                start = perf_counter()
                trace.activity(
                    "context",
                    "running",
                    "Load the report index",
                    "Resolve the active document and its source pages.",
                )
                index, hit = (
                    documents.get_index(document_id)
                    if documents
                    else load_pdf(selected_pdf, data_dir)
                )
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
                if semantic is not None:
                    started = perf_counter()
                    semantic.prepare(index)
                    trace.log("semantic_prepare", started, offline=True)
                answer = local_answer(
                    question,
                    index,
                    answer_model,
                    trace,
                    args.seconds,
                    topic,
                    planner,
                    args.workflow,
                    semantic,
                    context,
                    args.loop_profile,
                    verifier=audit_model if role_options else None,
                    **(
                        {"document_path": selected_pdf}
                        if args.workflow == "extractive"
                        else {}
                    ),
                )
                if role_options:
                    answer["role_assignment"] = assignment
                    answer.setdefault(
                        "model_roles",
                        {
                            "answer": {
                                "model": answer_model.model,
                                "endpoint": answer_model.endpoint,
                            },
                            "verifier": {
                                "model": audit_model.model,
                                "endpoint": audit_model.endpoint,
                            },
                            "shared_model": answer_model.endpoint == audit_model.endpoint
                            and answer_model.model == audit_model.model,
                        },
                    )
                answer["local_only"] = local_only
                answer["verification_provider"] = provider
                if provider == "local":
                    answer["verification_notice"] = LOCAL_VERIFICATION_WARNING
                if "review_packet" in answer:
                    answer["review_packet"].update(
                        local_only=local_only,
                        verification_provider=provider,
                        verification_notice=answer.get("verification_notice"),
                    )
                if trace.query_plan:
                    answer["query_plan"] = trace.query_plan
                answer["document"] = document_info or {
                    "id": index.fingerprint,
                    "name": selected_pdf.name,
                    "pages": len(index.pages),
                    "searchable_pages": len(index.pages),
                }
                answer["build"] = {
                    "version": version("drishti-evidence-guard"),
                    "contract": answer.get("contract", "coverage-v3"),
                }
                trace.log("finish", route=answer["route"])
                result = dict(answer, trace_id=trace.id, events=trace.events)
                record = save_record(
                    data_dir / "query-records",
                    trace.id,
                    {
                        "original_question": question,
                        "conversation_subject": topic,
                        "explicit_context": context,
                        "document_sha256": index.fingerprint,
                        "workflow": args.workflow,
                        "loop_profile": args.loop_profile,
                        "role_assignment": assignment if role_options else None,
                        "local_only": local_only,
                        "verification_provider": provider,
                        "model_roles": answer.get("model_roles"),
                        "document": answer["document"],
                        "source_sha256": source_hash,
                        "source_changed_during_run": source_hash != source_fingerprint(),
                        "result": result,
                    },
                )
                result["record"] = record
                return result

            if args.command == "ui":
                print(
                    "Preparing local model instructions with synthetic inputs…",
                    flush=True,
                )
                prepare(
                    planner,
                    model,
                    data_dir,
                    workflow=args.workflow,
                    verifier=verifier if role_options and not hosted else None,
                )
                serve(
                    args.port,
                    ask_pdf,
                    data_dir,
                    args.enable_live_demo,
                    document=pdf,
                    documents=documents,
                    loop_profile=args.loop_profile,
                    model_roles=role_options,
                    hosted_verification=hosted,
                    local_verification={
                        "answer": model.model,
                        "verifier": local_verifier.model,
                    }
                    if args.workflow == "extractive"
                    else None,
                    contract="source-selection-v1"
                    if args.workflow == "extractive"
                    else "document-work-v1"
                    if args.workflow == "working"
                    else "coverage-v3",
                )
                return
            if args.command == "evaluate":
                suite = outside_repository(args.suite)
                result = evaluate(suite, data_dir, ask_pdf, pdf)
            else:
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
    if args.command == "evaluate" and (
        result["summary"]["passed"] != result["summary"]["total"]
        or any(
            result["manifest"].get(key, False)
            for key in (
                "source_changed_during_run",
                "suite_changed_during_run",
                "document_changed_during_run",
            )
        )
    ):
        parser.exit(1)
