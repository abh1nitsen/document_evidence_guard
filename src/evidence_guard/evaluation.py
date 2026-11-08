"""External evaluation suites and reproducible contract measurements."""

import hashlib
import json
import math
import platform
import statistics
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from .evidence import load_pdf
from .workflow import CONTRACT_VERSION


def source_fingerprint() -> str:
    source = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate(suite: Path, data_dir: Path, ask, document: Path) -> dict:
    raw = suite.read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError("Evaluation suite exceeds the supported size.")
    source_hash = source_fingerprint()
    index, _ = load_pdf(document, data_dir)
    page_text = {p.number: " ".join(p.text.split()) for p in index.pages}
    cases = json.loads(raw)
    if not isinstance(cases, list) or not 1 <= len(cases) <= 1000:
        raise ValueError("Evaluation suite must be a nonempty JSON array.")
    seen = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or case["id"] in seen
            or not isinstance(case.get("question"), str)
            or not isinstance(case.get("routes"), list)
            or not case["routes"]
        ):
            raise ValueError("Each case needs a unique id, question and allowed routes.")
        if any(
            not isinstance(case.get(key, []), list)
            or any(not isinstance(value, str) for value in case.get(key, []))
            for key in ("routes", "contains", "forbidden")
        ):
            raise ValueError("Route and content labels must be arrays of strings.")
        if (
            not isinstance(case.get("pages", []), list)
            or any(type(page) is not int or page < 1 for page in case.get("pages", []))
            or not isinstance(case.get("category", "unspecified"), str)
        ):
            raise ValueError("Page labels must be positive integers.")
        if case.get("topic") is not None and (
            not isinstance(case["topic"], str) or not 1 <= len(case["topic"]) <= 120
        ):
            raise ValueError("Conversation subjects must be short strings.")
        if "calculation" in case and (
            type(case["calculation"]) not in (int, float)
            or not math.isfinite(case["calculation"])
        ):
            raise ValueError("Calculation labels must be finite numbers.")
        seen.add(case["id"])
    rows = []
    for case in cases:
        result = ask(case["question"], topic=case.get("topic"))
        comparison = None
        plan = result.get("query_plan")
        if plan and case.get("pages"):
            original = index.search(plan["original_question"], limit=6)
            fused = index.search_queries(
                plan["original_question"],
                plan["queries"],
                limit=6,
            )
            expected = set(case["pages"])
            comparison = {
                "original_reference_rank": next(
                    (rank for rank, p in enumerate(original, 1) if p.page in expected),
                    None,
                ),
                "fused_reference_rank": next(
                    (rank for rank, p in enumerate(fused, 1) if p.page in expected),
                    None,
                ),
            }
        claims = result.get("claims", [])
        text = " ".join(claim.get("text", "") for claim in claims)
        issues = []
        if result["route"] not in case["routes"]:
            issues.append("route")
        if any(
            value.casefold() not in text.casefold() for value in case.get("contains", [])
        ):
            issues.append("required_content")
        if any(
            value.casefold() in text.casefold() for value in case.get("forbidden", [])
        ):
            issues.append("forbidden_content")
        pages = {claim.get("page") for claim in claims}
        if case.get("pages") and not pages.intersection(case["pages"]):
            issues.append("evidence_page")
        if "calculation" in case:
            got = result.get("calculation", {}).get("value")
            if not isinstance(got, (int, float)) or not math.isclose(
                got, case["calculation"], rel_tol=1e-7, abs_tol=1e-7
            ):
                issues.append("calculation")
        if result["route"] == "evidence_only" and any(
            claim["text"] != claim["quote"] for claim in claims
        ):
            issues.append("extractive_contract")
        if any(
            claim.get("page") not in page_text
            or not isinstance(claim.get("quote"), str)
            or not claim["quote"].strip()
            or " ".join(claim["quote"].split()) not in page_text[claim["page"]]
            for claim in claims
        ):
            issues.append("source_provenance")
        finish = next(
            (e for e in reversed(result["events"]) if e["kind"] == "finish"), {}
        )
        rows.append(
            {
                "id": case["id"],
                "category": case.get("category", "unspecified"),
                "passed": not issues,
                "issues": issues,
                "elapsed_ms": finish.get("elapsed_ms", 0),
                "result": result,
                "label": case,
                "retrieval_comparison": comparison,
            }
        )
    latency = sorted(row["elapsed_ms"] for row in rows)
    by_category = {}
    for row in rows:
        group = by_category.setdefault(row["category"], {"passed": 0, "total": 0})
        group["total"] += 1
        group["passed"] += row["passed"]
    with document.open("rb") as handle:
        document_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "manifest": {
            "package": version("drishti-evidence-guard"),
            "contract": CONTRACT_VERSION,
            "source_tree_sha256": source_hash,
            "source_changed_during_run": source_hash != source_fingerprint(),
            "suite_sha256": hashlib.sha256(raw).hexdigest(),
            "document_sha256": index.fingerprint,
            "document_changed_during_run": document_hash != index.fingerprint,
            "suite_changed_during_run": hashlib.sha256(suite.read_bytes()).hexdigest()
            != hashlib.sha256(raw).hexdigest(),
            "python": sys.version.split()[0],
            "platform": platform.system(),
            "pypdf": version("pypdf"),
            "model_alias": next(
                (
                    event.get("model")
                    for event in rows[0]["result"]["events"]
                    if event["kind"] == "start"
                ),
                None,
            ),
            "planner_alias": next(
                (
                    e.get("planner_model")
                    for e in rows[0]["result"]["events"]
                    if e["kind"] == "start"
                ),
                None,
            ),
        },
        "summary": {
            "passed": sum(row["passed"] for row in rows),
            "total": len(rows),
            "by_category": by_category,
            "routes": {
                route: sum(row["result"]["route"] == route for row in rows)
                for route in sorted({row["result"]["route"] for row in rows})
            },
            "median_ms": round(statistics.median(latency), 2),
            "p95_ms": latency[math.ceil(len(latency) * 0.95) - 1],
            "local_model_calls": sum(
                event["kind"] == "local_model"
                for row in rows
                for event in row["result"]["events"]
            ),
            "model_failure_cases": sum(
                any(event["kind"] == "local_issue" for event in row["result"]["events"])
                for row in rows
            ),
            "provenance_failures": sum(
                "source_provenance" in row["issues"] for row in rows
            ),
            "source_contract_failures": sum(
                "extractive_contract" in row["issues"] for row in rows
            ),
            "planner_accepted_cases": sum(
                bool(row["result"].get("query_plan")) for row in rows
            ),
            "planner_fallback_cases": sum(
                any(
                    e["kind"] == "query_understanding" and e.get("status") == "fallback"
                    for e in row["result"]["events"]
                )
                for row in rows
            ),
            "retrieval_comparisons": [
                {"id": row["id"], **row["retrieval_comparison"]}
                for row in rows
                if row["retrieval_comparison"]
            ],
            "scope": "Automated route, content and source-contract checks. "
            "Pass rate is not an independently measured semantic accuracy.",
        },
        "cases": rows,
    }
    directory = data_dir / "evaluations"
    directory.mkdir(exist_ok=True)
    output = directory / f"{uuid4().hex}.json"
    with NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    Path(handle.name).replace(output)
    return {
        "report_file": str(output),
        "summary": report["summary"],
        "manifest": report["manifest"],
    }
