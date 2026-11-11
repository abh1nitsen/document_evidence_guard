"""Matched development diagnostics. All content-bearing inputs/outputs stay outside Git.

This deliberately does not tune on the frozen Evaluation split or award semantic
accuracy from model agreement, citation identity, or retrieval labels.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from evidence_guard.cli import outside_repository
from evidence_guard.evaluation import source_fingerprint
from evidence_guard.evidence import load_pdf
from evidence_guard.models import LocalModel
from evidence_guard.telemetry import Trace
from evidence_guard.workflow import local_answer
from evidence_guard.working import save_record


class RecordingModel(LocalModel):
    def __init__(self, *args):
        super().__init__(*args)
        self.records = []

    def run(self, instruction, payload, timeout, **kwargs):
        record = {"instruction": instruction, "payload": payload, "options": kwargs}
        self.records.append(record)
        start = perf_counter()
        try:
            value = super().run(instruction, payload, timeout, **kwargs)
            record["response"] = value
            return value
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["duration_ms"] = round((perf_counter() - start) * 1000, 2)
            record["usage"] = getattr(self, "last_usage", None)
            record["raw_response"] = getattr(self, "last_response", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--local-url", default="http://127.0.0.1:18089/v1")
    parser.add_argument("--local-model", default="qwen2.5-1.5b")
    parser.add_argument("--semantic-model-dir", type=Path)
    parser.add_argument(
        "--regression-file",
        type=Path,
        help="External JSON array of Regression cases; never stored in Git",
    )
    args = parser.parse_args()
    pdf, bank_path, output = map(outside_repository, (args.pdf, args.bank, args.output))
    raw = bank_path.read_bytes()
    bank = json.loads(raw)
    cases = [c for c in bank if c["id"] in args.ids]
    if len(cases) != len(set(args.ids)) or any(
        c["split"] not in {"Development", "Regression"} for c in cases
    ):
        parser.error("Select existing Development/Regression IDs only.")
    extra_raw = None
    if args.regression_file:
        extra_raw = outside_repository(args.regression_file).read_bytes()
        extra_cases = json.loads(extra_raw)
        if not isinstance(extra_cases, list) or any(
            not isinstance(c, dict)
            or c.get("split") != "Regression"
            or not isinstance(c.get("id"), str)
            or not isinstance(c.get("question"), str)
            or not 1 <= len(c["question"].strip()) <= 1500
            for c in extra_cases
        ):
            parser.error("Regression file must contain bounded Regression cases.")
        cases.extend(extra_cases)
    if len({c["id"] for c in cases}) != len(cases):
        parser.error("Case IDs must be unique across the bank and regressions.")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = output / run_id
    directory.mkdir(parents=True)
    (directory / "question-bank.json").write_bytes(raw)
    if extra_raw is not None:
        (directory / "regression-cases.json").write_bytes(extra_raw)
    source_hash = source_fingerprint()
    model = RecordingModel(args.local_url, args.local_model)
    index, _ = load_pdf(pdf, output)
    semantic = None
    semantic_start = perf_counter()
    if args.semantic_model_dir:
        from evidence_guard.semantic import SemanticSearch

        semantic = SemanticSearch(outside_repository(args.semantic_model_dir), output)
        semantic.prepare(index)
    preparation_ms = round((perf_counter() - semantic_start) * 1000, 2)
    repository = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repository)
    (directory / "tracked-changes.patch").write_bytes(patch)
    snapshot = directory / "source-snapshot"
    for path in sorted((repository / "src").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            target = snapshot / path.relative_to(repository)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    for name in ("pyproject.toml", "uv.lock", "scripts/compare_workflows.py"):
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository / name).read_bytes())
    manifest = {
        "run_id": run_id,
        "git_head": head,
        "source_sha256": source_hash,
        "bank_sha256": hashlib.sha256(raw).hexdigest(),
        "regression_sha256": hashlib.sha256(extra_raw).hexdigest() if extra_raw else None,
        "document_sha256": index.fingerprint,
        "model_alias": args.local_model,
        "seconds_per_path": args.seconds,
        "semantic_enabled": semantic is not None,
        "semantic_preparation_ms": preparation_ms,
        "hosted_calls": 0,
        "question_ids": [c["id"] for c in cases],
        "baseline": "original question + same subject context, legacy selection",
        "assisted": "original question + same subject context, document working loop",
        "timing": "Independent sequential paths, alternating order; models warm. "
        "Document and semantic preparation reported separately.",
        "scope": "Development/regression diagnostics, human adjudication pending; "
        "not confirmatory evaluation or a savings estimate.",
    }
    save_record(directory, "manifest-start", manifest)
    rows = []
    for position, case in enumerate(cases):
        paths = ("legacy", "working") if position % 2 == 0 else ("working", "legacy")
        for workflow in paths:
            trace = Trace(output, "comparison_" + workflow)
            start, first = perf_counter(), len(model.records)
            result = local_answer(
                case["question"],
                index,
                model,
                trace,
                args.seconds,
                context=case.get("context") or None,
                workflow=workflow,
                semantic=semantic if workflow == "working" else None,
            )
            elapsed = round((perf_counter() - start) * 1000, 2)
            trace.log("finish", route=result["route"])
            result["events"] = trace.events
            row = {
                "id": case["id"],
                "workflow": workflow,
                "question": case["question"],
                "context": case.get("context", ""),
                "elapsed_ms": elapsed,
                "result": result,
                "model_calls": model.records[first:],
                "human_review": {
                    "status": "pending",
                    "correctness": None,
                    "completeness": None,
                    "citation_support": None,
                    "unsupported_assertions": None,
                },
            }
            evidence = result.get("evidence", []) + result.get("claims", [])
            retrieved = {p["passage_id"] for p in evidence if "passage_id" in p}
            groups = case.get("evidence_groups", [])
            row["retrieval_diagnostic"] = {
                "required_groups": len(groups),
                "groups_seen": sum(
                    bool(retrieved.intersection(g["passage_ids"])) for g in groups
                ),
                "scope": "Source availability only; not answer correctness.",
            }
            save_record(directory, case["id"] + "-" + workflow, row)
            rows.append(row)
            with (output / "failure_history.jsonl").open("a", encoding="utf-8") as handle:
                # Pending review remains in history, including cases later passing.
                handle.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "id": case["id"],
                            "question": case["question"],
                            "context": case.get("context", ""),
                            "workflow": workflow,
                            "route": result["route"],
                            "reason": result.get("reason"),
                            "review_status": "pending",
                            "artifact": str(
                                directory / (case["id"] + "-" + workflow + ".json")
                            ),
                        }
                    )
                    + "\n"
                )
            print(case["id"], workflow, result["route"], elapsed, "ms", flush=True)
    manifest["source_changed_during_run"] = source_hash != source_fingerprint()
    manifest["bank_changed_during_run"] = raw != bank_path.read_bytes()
    summary = {"manifest": manifest, "paths": {}}
    for workflow in ("legacy", "working"):
        selected = [r for r in rows if r["workflow"] == workflow]
        summary["paths"][workflow] = {
            "cases": len(selected),
            "median_ms": statistics.median(r["elapsed_ms"] for r in selected),
            "model_calls": sum(len(r["model_calls"]) for r in selected),
            "routes": {
                route: sum(r["result"]["route"] == route for r in selected)
                for route in {r["result"]["route"] for r in selected}
            },
            "semantic_accuracy": "pending independent review",
        }
    save_record(directory, "summary", summary)
    hashes = {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }
    save_record(directory, "artifact-hashes", hashes)
    print("Saved", directory, flush=True)


if __name__ == "__main__":
    main()
