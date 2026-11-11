"""Fixed-draft local judge comparison. Test labels never enter model requests."""

import argparse
import hashlib
import json
import math
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from judge_protocol import INSTRUCTION, assess, schema

from evidence_guard.evaluation import source_fingerprint
from evidence_guard.models import LocalModel, ModelError
from evidence_guard.storage import outside_repository
from evidence_guard.working import AUDIT, AUDIT_SCHEMA, save_record, valid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--tokens", type=int, default=700)
    parser.add_argument("--qwen-url", default="http://127.0.0.1:18089/v1")
    parser.add_argument("--gemma-url", default="http://127.0.0.1:18090/v1")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.tokens < 1:
        parser.error("Use positive finite seconds and tokens.")
    cases_path = outside_repository(args.cases)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        parser.error("Expected a nonempty external case array.")
    seen = set()
    for case in cases:
        if case["id"] in seen or type(case["expected_approve"]) is not bool:
            parser.error("Unique IDs and boolean expected_approve labels required.")
        seen.add(case["id"])
    output = outside_repository(args.output) / uuid4().hex
    output.mkdir(parents=True)
    scripts = Path(__file__).parent
    for name in ("evaluate_judges.py", "judge_protocol.py"):
        shutil.copy2(scripts / name, output / name)
    shutil.copytree(
        scripts.parent / "src",
        output / "source",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(cases_path, output / "cases.json")
    source_hash = source_fingerprint()
    models = {
        "qwen": LocalModel(args.qwen_url, "qwen2.5-1.5b"),
        "gemma": LocalModel(args.gemma_url, "gemma-2-2b-it"),
    }
    conditions = [
        ("qwen", "current"),
        ("gemma", "current"),
        ("qwen", "proposed"),
        ("gemma", "proposed"),
    ]
    preregistration = {
        "created_utc": datetime.now(UTC).isoformat(),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "source_sha256": source_hash,
        "seconds": args.seconds,
        "tokens": args.tokens,
        "current_instruction": AUDIT,
        "current_schema": AUDIT_SCHEMA,
        "proposed_instruction": INSTRUCTION,
        "conditions": conditions,
        "case_order": [case["id"] for case in cases],
        "labels": "Assistant-labeled pilot, not independent human adjudication.",
        "scope": "Prompt + schema + gate comparison; not prompt-only attribution.",
        "live_workflow_modified": False,
    }
    save_record(output, "preregistration", preregistration)
    print(str(output), flush=True)
    rows = []
    for position, case in enumerate(cases):
        shift = position % len(conditions)
        for model_key, protocol in conditions[shift:] + conditions[:shift]:
            model = models[model_key]
            # Fixed allowlist: expected outcomes and diagnostic labels remain private.
            payload = deepcopy(case["payload"])
            spec = AUDIT_SCHEMA if protocol == "current" else schema(payload)
            instruction = AUDIT if protocol == "current" else INSTRUCTION
            record = {
                "case_id": case["id"],
                "model": model.model,
                "protocol": protocol,
                "endpoint": model.endpoint,
                "request": payload,
                "request_sha256": digest(payload),
                "schema": spec,
                "instruction": instruction,
                "expected_approve": case["expected_approve"],
                "seconds": args.seconds,
                "tokens": args.tokens,
                "source_sha256": source_hash,
            }
            start = perf_counter()
            try:
                response = model.run(
                    instruction,
                    payload,
                    args.seconds,
                    schema=spec,
                    max_output_tokens=args.tokens,
                )
                record["response"] = response
                if not valid(response, spec):
                    raise ModelError("invalid_schema")
                if protocol == "current":
                    approved = all(response["checks"].values()) and not response["issues"]
                    outcome = {
                        "model_pass": approved,
                        "approved": approved,
                        "gate_issues": [],
                    }
                else:
                    outcome = assess(response, payload)
                record.update(outcome)
                record["status"] = "completed"
            except ModelError as exc:
                record.update(
                    status="error",
                    error=str(exc),
                    model_pass=None,
                    approved=False,
                    gate_issues=[str(exc)],
                )
            finally:
                record["elapsed_ms"] = round((perf_counter() - start) * 1000, 2)
                record["usage"] = model.last_usage
                record["raw_response"] = model.last_response
                record["source_changed"] = source_fingerprint() != source_hash
            save_record(output / "calls", uuid4().hex, record)
            rows.append(record)
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in (
                            "case_id",
                            "model",
                            "protocol",
                            "status",
                            "approved",
                            "elapsed_ms",
                        )
                    }
                ),
                flush=True,
            )
    save_record(output, "results", {"preregistration": preregistration, "rows": rows})
    save_record(
        output,
        "manifest",
        {
            "sha256": {
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.rglob("*")
                if p.is_file()
            }
        },
    )
    print("Finished: " + str(output), flush=True)


if __name__ == "__main__":
    main()
