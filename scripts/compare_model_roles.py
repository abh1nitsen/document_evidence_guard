"""Compare reversible local roles using external cases; preserve private evidence."""

import argparse
import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from evidence_guard.evaluation import source_fingerprint
from evidence_guard.evidence import Index, Page, load_pdf
from evidence_guard.models import LocalModel
from evidence_guard.storage import outside_repository
from evidence_guard.telemetry import Trace
from evidence_guard.workflow import local_answer
from evidence_guard.working import save_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answer-url", default="http://127.0.0.1:18089/v1")
    parser.add_argument("--answer-model", default="qwen2.5-1.5b")
    parser.add_argument("--verifier-url", default="http://127.0.0.1:18090/v1")
    parser.add_argument("--verifier-model", default="gemma-2-2b-it")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.repeats < 1:
        parser.error("Use positive finite seconds and positive repeats.")
    cases_file = outside_repository(args.cases)
    cases = json.loads(cases_file.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        parser.error("Cases must be a nonempty array.")
    seen = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or case["id"] in seen
            or not isinstance(case.get("question"), str)
            or not 1 <= len(case["question"]) <= 1500
            or bool(case.get("pdf")) == bool(case.get("pages"))
            or (
                case.get("context") is not None
                and (
                    not isinstance(case["context"], str)
                    or not 1 <= len(case["context"]) <= 3000
                )
            )
            or (
                case.get("pages")
                and (
                    not isinstance(case["pages"], list)
                    or any(not isinstance(p, str) for p in case["pages"])
                )
            )
        ):
            parser.error("Each unique case needs question and either pdf or pages.")
        seen.add(case["id"])
    output = outside_repository(args.output) / uuid4().hex
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(
        source / "src",
        output / "source" / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(__file__, output / "source" / Path(__file__).name)
    shutil.copy2(source / "pyproject.toml", output / "source" / "pyproject.toml")
    shutil.copy2(cases_file, output / "cases.json")
    source_hash = source_fingerprint()
    models = (
        LocalModel(args.answer_url, args.answer_model),
        LocalModel(args.verifier_url, args.verifier_model),
    )
    rows = []
    for repeat in range(args.repeats):
        for position, case in enumerate(cases):
            if case.get("pdf"):
                pdf = outside_repository(Path(case["pdf"]))
                index, _ = load_pdf(pdf, output)
            else:
                raw = json.dumps(case["pages"], ensure_ascii=False).encode()
                index = Index(
                    [Page(i, p) for i, p in enumerate(case["pages"], 1)],
                    hashlib.sha256(raw).hexdigest(),
                )
            order = ["primary", "reversed"]
            if (position + repeat) % 2:
                order.reverse()
            for assignment in order:
                answer, verifier = models if assignment == "primary" else models[::-1]
                trace = Trace(output, "role_comparison")
                start = perf_counter()
                result = local_answer(
                    case["question"],
                    index,
                    answer,
                    trace,
                    args.seconds,
                    workflow="working",
                    loop_profile="fast",
                    context=case.get("context"),
                    verifier=verifier,
                )
                row = {
                    "case_id": case["id"],
                    "repeat": repeat + 1,
                    "role_assignment": assignment,
                    "elapsed_ms": round((perf_counter() - start) * 1000, 2),
                    "first_draft_ms": next(
                        (
                            e["elapsed_ms"]
                            for e in trace.events
                            if e["kind"] == "local_model"
                            and e.get("role") == "writer"
                            and e.get("status") == "completed"
                        ),
                        None,
                    ),
                    "document_sha256": index.fingerprint,
                    "source_sha256": source_hash,
                    "source_changed": source_fingerprint() != source_hash,
                    "trace_id": trace.id,
                    "result": result,
                    "events": trace.events,
                    "human_adjudication": "pending",
                }
                save_record(output / "runs", trace.id, row)
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "case_id": case["id"],
                            "assignment": assignment,
                            "elapsed_ms": row["elapsed_ms"],
                            "reason": result.get("reason", result["route"]),
                        }
                    ),
                    flush=True,
                )
    save_record(
        output,
        "comparison",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "source_sha256": source_hash,
            "seconds": args.seconds,
            "repeats": args.repeats,
            "rows": rows,
            "interpretation": "Diagnostic comparisons; audit approval is not accuracy.",
        },
    )
    hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    save_record(output, "manifest", {"sha256": hashes})
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
