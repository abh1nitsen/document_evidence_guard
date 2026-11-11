"""Prepare local instruction prefixes with synthetic inputs before serving UI."""

from time import perf_counter

from .adaptive import ASSESS, coverage_schema
from .models import ModelError
from .telemetry import Trace
from .understanding import INSTRUCTION, SCHEMA


def prepare(planner, selector, data_dir, workflow="legacy", verifier=None):
    trace = Trace(data_dir, "local_warmup")
    calls = [
        (
            selector,
            ASSESS,
            {"needs": ["result"], "excerpts": []},
            coverage_schema(["result"], 1),
        )
    ]
    if workflow == "working":
        from .working import INSTRUCTION as WORK
        from .working import schema

        calls = [
            (
                selector,
                WORK,
                {"original_question": "What is this?", "evidence": [], "tool_result": {}},
                schema(semantic=False, discovered={}, read={}),
            )
        ]
        if verifier is not None:
            from .working import AUDIT, AUDIT_SCHEMA

            calls.append(
                (
                    verifier,
                    AUDIT,
                    {
                        "original_question": "Describe the synthetic example.",
                        "requirements": ["Describe the example"],
                        "claims": [],
                        "sources": [],
                    },
                    AUDIT_SCHEMA,
                )
            )
    if workflow == "extractive":
        from .extractive import PREP_SCHEMA, PREPARE, VERIFY, VERIFY_SCHEMA

        calls = [
            (
                selector,
                PREPARE,
                {"original_question": "What is this?", "outline": [], "context": None},
                PREP_SCHEMA,
            )
        ]
        if verifier is not None:
            calls.append(
                (
                    verifier,
                    VERIFY,
                    {
                        "original_question": "Who is named?",
                        "selected_passages": [],
                        "requirements": ["Identify a name"],
                    },
                    VERIFY_SCHEMA,
                )
            )
    if planner:
        calls.insert(
            0,
            (
                planner,
                INSTRUCTION,
                {
                    "question": "Describe the synthetic example.",
                    "document_outline": [],
                    "conversation_subject": None,
                },
                SCHEMA,
            ),
        )
    for model, instruction, payload, schema in calls:
        start = perf_counter()
        status = "prefix_prepared"
        try:
            model.run(instruction, payload, 20, schema=schema, max_output_tokens=8)
        except ModelError as exc:
            if str(exc) != "model_output_truncated":
                status = str(exc)
        trace.log(
            "local_model",
            start,
            stage="warmup",
            model=model.model,
            status=status,
            usage=getattr(model, "last_usage", None),
            synthetic_only=True,
            output_token_limit=8,
        )
