"""Public synthetic fixture. Only these static prompts can reach Groq."""

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .evidence import Index, Page
from .models import GROQ_ENDPOINT, GROQ_MODEL, ModelError, chat, parse_object
from .telemetry import Trace, cost


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    pages: tuple[int, ...]
    correct: str
    local_reason: str


PAGES = [
    Page(1, "The fictional North Annex badge desk closes at 17:30 on weekdays."),
    Page(2, "Visitors may enter an exhibit after closing only during an open session."),
    Page(
        3,
        "A demonstration is an open session only when a host is present. "
        "The Blue Room is hosting a demonstration at 19:00; a host is present.",
    ),
    Page(
        4,
        "The South Annex register lists entries, but no independent head count "
        "was taken on opening day.",
    ),
]

CASES = (
    Case(
        "lookup",
        "When does the North Annex badge desk close on weekdays?",
        (1,),
        "17:30",
        "local_verified",
    ),
    Case(
        "cross_reference",
        "Can a visitor enter the Blue Room exhibit at 19:00, "
        "after closing, during the scheduled demonstration?",
        (2, 3),
        "yes",
        "cross_section_interpretation_unresolved",
    ),
    Case(
        "missing_measurement",
        "What was the independent South Annex opening-day head count?",
        (4,),
        "unknown",
        "missing_source_fact",
    ),
)


def frontier(case: Case, trace: Trace, key: str) -> str:
    # The only caller supplies built-in fixtures. No user question or PDF path.
    start = perf_counter()
    trace.log("frontier_start", provider="groq", model=GROQ_MODEL, case=case.name)
    try:
        result = chat(
            GROQ_ENDPOINT,
            GROQ_MODEL,
            [
                {
                    "role": "system",
                    "content": "Use only the fictional excerpts. "
                    "Return JSON with one string field answer. For a time use HH:MM. "
                    "For a yes/no question return yes or no. If the fact was not "
                    "established, return unknown.",
                },
                {
                    "role": "user",
                    "content": case.question
                    + "\n"
                    + "\n".join(p.text for p in PAGES if p.number in case.pages),
                },
            ],
            20,
            key,
        )
    except ModelError as exc:
        trace.log("frontier_end", start, status="error", code=str(exc))
        return "provider_error"
    usage = result.usage if isinstance(result.usage, dict) else {}
    details = usage.get("prompt_tokens_details") or {}
    trace.log(
        "frontier_end",
        start,
        status="response",
        provider_request_id=result.request_id,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        cached_prompt_tokens=details.get("cached_tokens", 0)
        if isinstance(details, dict)
        else None,
        list_cost_usd=cost(usage),
    )
    start = perf_counter()
    try:
        candidate = parse_object(result.content).get("answer")
        valid = isinstance(candidate, str) and candidate.strip().lower() == case.correct
    except ModelError:
        valid = False
    trace.log("frontier_verify", start, passed=valid, verifier="synthetic_fixture_key")
    if not valid:
        return "frontier_unverified"
    return "insufficient_evidence" if case.correct == "unknown" else "answer_frontier"


def run_demo(data_dir: Path, mode: str, live: bool) -> list[dict]:
    key = os.environ.get("GROQ_API_KEY", "").strip() if live else ""
    if live and not key:
        raise ValueError("Set GROQ_API_KEY in the environment before using --live.")
    results = []
    index = Index(PAGES, "synthetic-v1")
    for case in CASES:
        trace = Trace(data_dir, mode if live else f"{mode}_offline")
        trace.log(
            "start", case=case.name, corpus="synthetic-v1", local="scripted_fixture"
        )
        start = perf_counter()
        fetched = index.fetch(case.question, 1)
        trace.log("fetch", start, attempt=1, pages=[p.number for p in fetched])
        if case.name == "cross_reference":
            start = perf_counter()
            # The local controller opens the referenced rule on its targeted retry.
            trace.log("fetch", start, attempt=2, pages=list(case.pages))
        trace.log("first_verified", pages=list(case.pages))
        should_call = mode == "baseline" or case.name == "cross_reference"
        reason = "always_frontier_baseline" if mode == "baseline" else case.local_reason
        if should_call and live:
            trace.log("route", decision="call", reason=reason, pages=list(case.pages))
            route = frontier(case, trace, key)
        elif should_call:
            trace.log(
                "route",
                decision="skip",
                reason="offline_demo",
                would_call_reason=reason,
                pages=list(case.pages),
            )
            route = "would_call"
        else:
            trace.log("route", decision="skip", reason=reason, pages=list(case.pages))
            route = "answer_local" if case.name == "lookup" else "insufficient_evidence"
        trace.log("finish", route=route)
        results.append(
            {
                "case": case.name,
                "route": route,
                "trace_id": trace.id,
                "events": trace.events,
            }
        )
    return results
