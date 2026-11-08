"""Synthetic request interpretation, coverage and routing regressions."""

import json
import sqlite3
from time import perf_counter

import pytest

from evidence_guard.adaptive import run, validate_supports
from evidence_guard.evidence import Index, Page
from evidence_guard.models import ModelError
from evidence_guard.telemetry import Trace
from evidence_guard.understanding import understand


def plan(**changes):
    result = {
        "intent": "procedure",
        "subject": "Alpha assessment",
        "question": "Explain the Alpha assessment method.",
        "needs": ["steps", "calculation"],
        "queries": ["Alpha assessment methodology", "Alpha assessment formula"],
        "choices": [],
    }
    result.update(changes)
    return result


class Planner:
    model = "synthetic-planner"

    def __init__(self, value):
        self.value = value

    def run(self, *args, **kwargs):
        return self.value.copy()


def interpreted(tmp_path, value=None):
    trace = Trace(tmp_path, "synthetic")
    index = Index(
        [
            Page(
                1,
                "Chapter 1 Methods\n\n1.1 Alpha assessment\n"
                "Alpha assessment records eligible events.",
            )
        ],
        "test",
    )
    result = understand(
        "What is the Alpha methodology?", index, Planner(value or plan()), trace, 2
    )
    return result, trace, index


def test_plan_preserves_original_and_keeps_multiple_evidence_needs(tmp_path):
    result, trace, _ = interpreted(tmp_path)
    assert result["intent"] == "procedure"
    assert result["needs"] == ["steps", "calculation"]
    assert result["original_question"] == "What is the Alpha methodology?"
    with sqlite3.connect(trace.path) as db:
        persisted = " ".join(row[0] for row in db.execute("select event from events"))
    assert "What is the Alpha methodology?" not in persisted
    assert result["normalized_query"] not in persisted


def test_material_ambiguity_returns_choices_without_calling_assessor(tmp_path):
    value = plan(
        choices=[
            "Explain the project development process.",
            "Explain how the assessment score is calculated.",
        ]
    )
    result, trace, index = interpreted(tmp_path, value)
    answer = run(
        "What method is used here?", index, None, trace, perf_counter() + 2, result
    )
    assert answer["route"] == "ask_clarification"
    assert len(answer["suggestions"]) == 2
    assert answer["routing"]["reason"] == "ambiguous_request"


@pytest.mark.parametrize(
    "quote,slot",
    [
        ("Invented calibrated measurements.", 1),
        ("Alpha assessment records eligible events.", 0),
        ("Alpha assessment records eligible events.", True),
    ],
)
def test_coverage_rejects_fabricated_text_and_invalid_ids(quote, slot):
    index = Index([Page(1, "Alpha assessment records eligible events.")], "test")
    with pytest.raises(ModelError):
        validate_supports(
            {"supports": [{"need": "steps", "ids": [slot], "quote": quote}]},
            ["steps"],
            index.passages,
        )


def test_keyword_relevance_cannot_end_loop_with_uncovered_requirement(tmp_path):
    index = Index(
        [
            Page(i, f"Alpha assessment reference {i}. Use eligible events.")
            for i in range(1, 10)
        ],
        "test",
    )
    request, trace, _ = interpreted(tmp_path)

    class Assessor:
        model = "synthetic-assessor"
        calls = 0

        def run(self, instruction, payload, timeout, schema=None):
            self.calls += 1
            need = "steps" if self.calls == 1 else "calculation"
            assert payload["needs"] == (
                ["steps", "calculation"] if self.calls == 1 else ["calculation"]
            )
            p = payload["excerpts"][0]["sentences"][0]
            return {"supports": [{"need": need, "ids": [p["id"]]}]}

    model = Assessor()
    result = run(
        "Explain Alpha assessment.", index, model, trace, perf_counter() + 3, request
    )
    assert model.calls == 2
    assert result["coverage"]["status"] == "model_matched"
    assert result["coverage"]["semantic_verification"] == "not_independently_verified"
    assert result["routing"]["hosted_candidate"] is True
    assert len(result["claims"]) == 2
    assert result["route"] == "evidence_only"
    fetches = [e for e in trace.events if e["kind"] == "fetch"]
    assert not set(fetches[0]["passage_ids"]) & set(fetches[1]["passage_ids"])
    assert result["routing"]["outbound_allowed"] is False


def test_assessor_outage_is_not_reported_as_absent_source_information(tmp_path):
    request, trace, index = interpreted(tmp_path)

    class Broken:
        model = "broken"

        def run(self, *args, **kwargs):
            raise ModelError("model_timeout")

    result = run(
        "Explain Alpha assessment.", index, Broken(), trace, perf_counter() + 2, request
    )
    assert result["route"] == "local_error"
    assert result["routing"]["reason"] == "local_execution_failure"
    assert result["routing"]["hosted_candidate"] is True
    assert result["routing"]["action"] == "blocked_by_document_policy"
    assert not any(e["kind"] == "frontier_start" for e in trace.events)
    assert result["evidence"]


def test_missing_evidence_does_not_trigger_hosted_invention(tmp_path):
    request, trace, _ = interpreted(tmp_path)
    result = run(
        "unfindable word", Index([], "test"), None, trace, perf_counter() + 2, request
    )
    assert result["route"] == "insufficient_evidence"
    assert result["routing"]["hosted_candidate"] is False


def test_model_cannot_promote_an_unquoted_coverage_claim(tmp_path):
    request, trace, index = interpreted(tmp_path)

    class Fabricator:
        model = "synthetic-fabricator"

        def run(self, *args, **kwargs):
            return {
                "supports": [
                    {
                        "need": "steps",
                        "id": 1,
                        "quote": "Validated engineering calibration.",
                    }
                ]
            }

    result = run(
        "Explain Alpha assessment.",
        index,
        Fabricator(),
        trace,
        perf_counter() + 2,
        request,
    )
    assert not result["claims"]
    assert result["coverage"]["error"] == "invalid_coverage"


def test_invalid_plan_does_not_silently_invent_a_subject(tmp_path):
    bad = plan(subject="")
    result, trace, index = interpreted(tmp_path, bad)
    assert result is None
    answer = run("Explain this.", index, None, trace, perf_counter() + 2, result)
    assert answer["reason"] == "query_understanding_failed"
    assert answer["route"] == "local_error"
    assert json.dumps(answer)


@pytest.mark.parametrize("slot", [0, -1, 100, True, "1"])
def test_only_printed_sentence_ids_can_be_selected(slot):
    index = Index([Page(1, "Only the printed source sentence exists.")], "test")
    with pytest.raises(ModelError, match="invalid_coverage"):
        validate_supports(
            {"supports": [{"need": "result", "ids": [slot]}]}, ["result"], index.passages
        )


def test_sentence_selection_copies_exact_text_and_retains_comparison_sides():
    index = Index([Page(1, "Branch A scored 80. Branch B scored 90.")], "test")
    supports = validate_supports(
        {"supports": [{"need": "comparison", "ids": [1, 2]}]},
        ["comparison"],
        index.passages,
    )
    claims = supports["comparison"]
    assert [c["text"] for c in claims] == ["Branch A scored 80.", "Branch B scored 90."]
    assert all(c["text"] == c["quote"] for c in claims)


def test_generic_morphology_links_scoring_with_score():
    index = Index(
        [Page(1, "1.1 Condition score\nThe score uses bounded deductions.")], "test"
    )
    assert index.search("scoring")[0].page == 1


def test_one_formula_cannot_complete_three_distinct_requirements():
    index = Index([Page(1, "S = 100 multiplied by remaining factors.")], "test")
    supports = validate_supports(
        {
            "supports": [
                {"need": need, "ids": [1]}
                for need in ["steps", "calculation", "aggregation"]
            ]
        },
        ["steps", "calculation", "aggregation"],
        index.passages,
    )
    assert set(supports) == {"calculation"}


def test_explicit_why_request_cannot_silently_become_methodology(tmp_path):
    trace = Trace(tmp_path, "test")
    question = "Why is Branch A's rating high?"
    result = understand(question, Index([], "test"), Planner(plan()), trace, 2)
    assert result["intent"] == "explanation"
    assert result["normalized_query"] == question
    assert result["needs"] == ["result", "mechanism"]
    assert any(
        "rewrite_dropped_causal_request" in e.get("reasons", []) for e in trace.events
    )
