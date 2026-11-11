"""Compare a frozen build and current profiles using external document/question data.

Each case runs once per condition in rotating order. This is a diagnostic pilot,
not independent semantic adjudication. No case contents belong in this repository.
"""

import argparse
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from uuid import uuid4


def single(args):
    sys.path.insert(0, str(args.source / "src"))
    from evidence_guard.evaluation import source_fingerprint
    from evidence_guard.evidence import Index, Page, load_pdf
    from evidence_guard.models import LocalModel
    from evidence_guard.telemetry import Trace
    from evidence_guard.workflow import local_answer
    from evidence_guard.working import save_record

    args.output.mkdir(parents=True, exist_ok=True)
    case = next(
        c
        for c in json.loads(args.cases.read_text(encoding="utf-8"))
        if c["id"] == args.case_id
    )
    if case.get("pdf"):
        index, _ = load_pdf(Path(case["pdf"]), args.output)
    else:
        raw = json.dumps(case["pages"], ensure_ascii=False)
        index = Index(
            [Page(i, t) for i, t in enumerate(case["pages"], 1)],
            hashlib.sha256(raw.encode()).hexdigest(),
        )

    class RecordedModel(LocalModel):
        def __init__(self):
            super().__init__(args.local_url, args.local_model)
            self.records = []

        def run(self, instruction, payload, timeout, **kwargs):
            record = {
                "instruction": instruction,
                "request": deepcopy(payload),
                "options": kwargs,
                "timeout": timeout,
            }
            self.records.append(record)
            start = perf_counter()
            try:
                result = super().run(instruction, payload, timeout, **kwargs)
                record["response"] = deepcopy(result)
                return result
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                raise
            finally:
                record["duration_ms"] = (perf_counter() - start) * 1000
                record["usage"] = getattr(self, "last_usage", None)
                record["raw_response"] = getattr(self, "last_response", None)

    model = RecordedModel()
    trace = Trace(args.output, "latency_" + args.profile)
    source_hash = source_fingerprint()
    options = {"workflow": "working", "context": case.get("context") or None}
    if "loop_profile" in inspect.signature(local_answer).parameters:
        options["loop_profile"] = args.profile
    start = perf_counter()
    result = local_answer(case["question"], index, model, trace, args.seconds, **options)
    elapsed = round((perf_counter() - start) * 1000, 2)
    trace.log("finish", route=result["route"])
    first_draft = next(
        (
            e["elapsed_ms"]
            for e in trace.events
            if e["kind"] == "local_model"
            and e.get("role") == "writer"
            and e.get("status") == "completed"
        ),
        None,
    )
    row = {
        "case_id": case["id"],
        "profile": args.profile,
        "question": case["question"],
        "context": case.get("context", ""),
        "source_sha256": source_hash,
        "source_changed": source_hash != source_fingerprint(),
        "document_sha256": index.fingerprint,
        "elapsed_ms": elapsed,
        "first_draft_ms": first_draft,
        "model_calls": model.records,
        "model_alias": model.model,
        "result": result,
        "events": trace.events,
        "human_review": {
            "status": "pending",
            "correctness": None,
            "completeness": None,
            "citation_support": None,
        },
    }
    save_record(args.output, "result", row)
    print(case["id"], args.profile, round(elapsed / 1000, 2), result["route"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--local-url", default="http://127.0.0.1:18089/v1")
    parser.add_argument("--local-model", default="qwen2.5-1.5b")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--profile", default="fast")
    args = parser.parse_args()
    if args.single:
        single(args)
        return
    if not args.baseline_source or not (args.baseline_source / "src").is_dir():
        parser.error("Provide a frozen baseline source directory.")
    from evidence_guard.storage import outside_repository

    args.cases = outside_repository(args.cases)
    args.output = outside_repository(args.output)
    cases_raw = args.cases.read_bytes()
    cases = json.loads(cases_raw)
    if (
        not cases
        or any(
            not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", c["id"])
            or c.get("split") not in {"Development", "Regression"}
            for c in cases
        )
        or len({c["id"] for c in cases}) != len(cases)
    ):
        parser.error("Use unique safe Development/Regression IDs only.")
    for case in cases:
        if case.get("pdf"):
            outside_repository(Path(case["pdf"]))
    directory = args.output / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    directory.mkdir(parents=True)
    (directory / "cases.json").write_bytes(cases_raw)
    repository = Path(__file__).resolve().parents[1]
    snapshot = directory / "optimized-source"
    shutil.copytree(
        repository / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    for name in ("pyproject.toml", "uv.lock", "scripts/benchmark_latency.py"):
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository / name).read_bytes())
    conditions = [
        ("baseline", args.baseline_source),
        ("compact", snapshot),
        ("fast", snapshot),
    ]
    manifest = {
        "conditions": [c[0] for c in conditions],
        "seconds": args.seconds,
        "case_sha256": hashlib.sha256(cases_raw).hexdigest(),
        "timing": "Sequential real paths, rotating order, warm shared model. "
        "Index preparation excluded. Repetitions are not independent samples.",
        "quality": "Human review pending; no semantic superiority claim.",
        "hosted_inference_calls": 0,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    rows = []
    for number, case in enumerate(cases):
        order = conditions[number % 3 :] + conditions[: number % 3]
        for profile, source in order:
            target = directory / (case["id"] + "-" + profile)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single",
                "--source",
                str(source),
                "--cases",
                str(directory / "cases.json"),
                "--case-id",
                case["id"],
                "--profile",
                profile,
                "--output",
                str(target),
                "--seconds",
                str(args.seconds),
                "--local-url",
                args.local_url,
                "--local-model",
                args.local_model,
            ]
            subprocess.run(command, check=True, timeout=args.seconds + 60)
            row = json.loads((target / "result.json").read_text(encoding="utf-8"))
            rows.append(row)
            with (args.output / "observations.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(
                        {
                            "run": directory.name,
                            "case_id": case["id"],
                            "question": case["question"],
                            "profile": profile,
                            "route": row["result"]["route"],
                            "artifact": str(target / "result.json"),
                        }
                    )
                    + "\n"
                )
    summary = {"manifest": manifest, "conditions": {}}
    for profile, _ in conditions:
        selected = [r for r in rows if r["profile"] == profile]
        summary["conditions"][profile] = {
            "cases": len(selected),
            "median_ms": median(r["elapsed_ms"] for r in selected),
            "model_calls": sum(len(r["model_calls"]) for r in selected),
            "drafts": sum(bool(r["result"].get("draft_answer")) for r in selected),
            "source_changed": any(r["source_changed"] for r in selected),
        }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    hashes = {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }
    (directory / "hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print("Saved", directory, flush=True)


if __name__ == "__main__":
    main()
