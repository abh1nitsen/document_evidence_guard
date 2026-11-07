"""Metadata-only timing, API usage, and scenario projections."""

import json
import math
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4

# USD per million tokens. Snapshot of the public Groq model card, 2026-11-06.
LIST_RATES = (0.15, 0.075, 0.60)


def cost(usage: dict | None) -> float | None:
    if not isinstance(usage, dict):
        return None
    inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if any(type(n) is not int or n < 0 for n in (inputs, outputs)):
        return None
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    if type(cached) is not int or not 0 <= cached <= inputs:
        return None
    return round(
        (
            (inputs - cached) * LIST_RATES[0]
            + cached * LIST_RATES[1]
            + outputs * LIST_RATES[2]
        )
        / 1_000_000,
        12,
    )


class Trace:
    def __init__(
        self, data_dir: Path, mode: str, on_event: Callable[[dict], None] | None = None
    ):
        self.path = data_dir / "telemetry.sqlite"
        self.id = uuid4().hex
        self.mode = mode
        self.started = perf_counter()
        self.events: list[dict] = []
        self.on_event = on_event
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "seq INTEGER PRIMARY KEY, run_id TEXT, mode TEXT, "
                "kind TEXT, event TEXT)"
            )

    def log(self, kind: str, start: float | None = None, **metadata) -> None:
        now = perf_counter()
        event = {
            "kind": kind,
            "at_utc": datetime.now(UTC).isoformat(),
            "elapsed_ms": round((now - self.started) * 1000, 2),
            **metadata,
        }
        if start is not None:
            event["duration_ms"] = round((now - start) * 1000, 2)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT INTO events (run_id, mode, kind, event) VALUES (?, ?, ?, ?)",
                (self.id, self.mode, kind, json.dumps(event, allow_nan=False)),
            )
        self.events.append(event)
        if self.on_event:
            self.on_event(dict(event, trace_id=self.id))

    def activity(self, role: str, status: str, action: str, reason: str, **metadata):
        self.log(
            "activity",
            role=role,
            status=status,
            action=action,
            reason=reason,
            **metadata,
        )


def summarize(
    data_dir: Path, volume: int, local_cost: float, expected_rate: float | None
) -> dict:
    path = data_dir / "telemetry.sqlite"
    if not path.exists():
        raise ValueError("No telemetry yet. Run ask or demo first.")
    with closing(sqlite3.connect(path)) as db:
        rows = db.execute(
            "SELECT run_id, mode, kind, event FROM events ORDER BY seq"
        ).fetchall()
    totals = {}
    costs: dict[str, list[float]] = {"cascade": [], "baseline": []}
    latencies: dict[str, list[float]] = {}
    first_evidence: dict[str, list[float]] = {}
    for _, mode, kind, raw in rows:
        event = json.loads(raw)
        stats = totals.setdefault(
            mode,
            {
                "questions": 0,
                "verified_answers": 0,
                "calls_started": 0,
                "calls_without_usage": 0,
                "api_list_cost_usd": 0.0,
                "routes": {},
                "gate_reasons": {},
            },
        )
        if kind == "route":
            reason = event["reason"]
            stats["gate_reasons"][reason] = stats["gate_reasons"].get(reason, 0) + 1
        if kind == "finish":
            route = event["route"]
            stats["questions"] += 1
            stats["verified_answers"] += route in {"answer_local", "answer_frontier"}
            stats["routes"][route] = stats["routes"].get(route, 0) + 1
            latencies.setdefault(mode, []).append(event["elapsed_ms"])
        if kind == "first_verified":
            first_evidence.setdefault(mode, []).append(event["elapsed_ms"])
        if kind == "frontier_start":
            stats["calls_started"] += 1
            stats["calls_without_usage"] += 1
        if kind == "frontier_end" and event.get("list_cost_usd") is not None:
            usd = event["list_cost_usd"]
            stats["api_list_cost_usd"] += usd
            stats["calls_without_usage"] -= 1
            if mode in costs:
                costs[mode].append(usd)
    for mode, stats in totals.items():
        stats["api_list_cost_usd"] = round(stats["api_list_cost_usd"], 12)
        for name, samples in (
            ("end_to_end", latencies.get(mode, [])),
            ("first_verified_evidence", first_evidence.get(mode, [])),
        ):
            if samples:
                ordered = sorted(samples)
                stats[f"{name}_ms"] = {
                    "p50": ordered[math.ceil(len(ordered) * 0.50) - 1],
                    "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
                    "samples": len(ordered),
                }
    projection: dict
    if not all(costs.values()):
        projection = {
            "status": "need_metered_synthetic_cascade_and_baseline",
        }
    elif any(totals[mode]["calls_without_usage"] for mode in costs):
        projection = {"status": "incomplete_usage_in_comparison"}
    else:
        cascade = totals["cascade"]
        rate = (
            expected_rate
            if expected_rate is not None
            else cascade["calls_started"] / cascade["questions"]
        )
        baseline_usd = volume * mean(costs["baseline"])
        api_usd = volume * rate * mean(costs["cascade"])
        projection = {
            "status": "synthetic_scenario",
            "monthly_questions": volume,
            "assumed_frontier_rate": rate,
            "rate_source": "user_scenario" if expected_rate is not None else "demo_mix",
            "frontier_calls_at_scale": volume * rate,
            "avoided_frontier_calls_at_scale": volume * (1 - rate),
            "all_frontier_list_cost_usd": round(baseline_usd, 4),
            "cascade_api_list_cost_usd": round(api_usd, 4),
            "api_savings_usd": round(baseline_usd - api_usd, 4),
            "assumed_local_cost_usd": round(local_cost * volume, 4),
            "net_savings_usd": round(baseline_usd - api_usd - local_cost * volume, 4),
            "metered_calls": {key: len(value) for key, value in costs.items()},
        }
    return {
        "by_mode": totals,
        "projection": projection,
        "pricing": {
            "model": "openai/gpt-oss-120b",
            "as_of": "2026-11-06",
            "usd_per_million": dict(
                zip(("input", "cached_input", "output"), LIST_RATES, strict=True)
            ),
            "source": "https://console.groq.com/docs/model/openai/gpt-oss-120b",
        },
    }
