"""Prepare local instruction prefixes with synthetic inputs before serving UI."""

from time import perf_counter

from .adaptive import ASSESS, coverage_schema
from .models import ModelError
from .telemetry import Trace
from .understanding import INSTRUCTION, SCHEMA


def prepare(planner, selector, data_dir):
    trace = Trace(data_dir, "local_warmup")
    calls = [
        (
            selector,
            ASSESS,
            {"needs": ["result"], "excerpts": []},
            coverage_schema(["result"], 1),
        )
    ]
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
