"""Experiment gates reject malformed evidence without treating them as semantic proof."""

from copy import deepcopy

import pytest

from scripts.judge_protocol import assess


@pytest.fixture
def sample():
    payload = {
        "claims": [{"text": "The recipient is Mira Patel.", "citations": ["s1"]}],
        "sources": [{"id": "s1", "text": "The recipient is Mira Patel."}],
    }
    response = {
        "task_match": "pass",
        "task_issue": "",
        "missing_requirements": [],
        "claim_checks": [
            {
                "claim_index": "1",
                "claim": "The recipient is Mira Patel.",
                "verdict": "supported",
                "evidence": [
                    {"source_id": "s1", "quote": "The recipient is Mira Patel."}
                ],
                "issue": "",
            }
        ],
        "evidence_scope": "adequate",
        "recovery_query": "",
    }
    return payload, response


def test_valid_support_is_eligible_for_a_passing_audit(sample):
    payload, response = sample
    assert assess(response, payload) == {
        "model_pass": True,
        "approved": True,
        "gate_issues": [],
    }


def test_positive_explanations_alone_are_not_failures(sample):
    payload, response = sample
    response["task_issue"] = "The answer directly addresses the requested part."
    response["claim_checks"][0]["issue"] = "The evidence supports this claim."
    assert assess(response, payload)["approved"] is True


@pytest.mark.parametrize("kind", ["invented_quote", "no_evidence", "unchecked_claim"])
def test_structural_evidence_gates_override_model_pass(sample, kind):
    payload, response = deepcopy(sample)
    if kind == "invented_quote":
        response["claim_checks"][0]["evidence"][0]["quote"] = "Patel is the author."
    elif kind == "no_evidence":
        response["claim_checks"][0]["evidence"] = []
    else:
        payload["claims"].append({"text": "The author is Arun Shah."})
    result = assess(response, payload)
    assert result["model_pass"] is True
    assert result["approved"] is False
    assert result["gate_issues"]


@pytest.mark.parametrize(
    "kind",
    ["wrong_task", "missing_part", "contradicted", "insufficient", "coverage_unknown"],
)
def test_model_reported_failure_blocks_audit(sample, kind):
    payload, response = deepcopy(sample)
    if kind == "wrong_task":
        response["task_match"] = "fail"
    elif kind == "missing_part":
        response["missing_requirements"] = ["Identify the author"]
    elif kind == "contradicted":
        response["claim_checks"][0]["verdict"] = "contradicted"
    elif kind == "insufficient":
        response["claim_checks"][0]["verdict"] = "insufficient_evidence"
    else:
        response["evidence_scope"] = "unknown"
    assert assess(response, payload)["approved"] is False


def test_invalid_schema_cannot_approve(sample):
    payload, response = sample
    response["extra"] = True
    assert assess(response, payload)["gate_issues"] == ["invalid_schema"]
