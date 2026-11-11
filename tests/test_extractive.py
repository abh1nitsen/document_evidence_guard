"""Source-only output, scope flags, selection boundaries and document identity."""

from time import perf_counter

import pytest

from evidence_guard.evidence import Page
from evidence_guard.extractive import SourceIndex, render, run
from evidence_guard.telemetry import Trace


class Model:
    model = "synthetic-selector"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def run(self, instruction, payload, *args, **kwargs):
        self.calls.append(payload)
        response = next(self.responses)
        return response(payload) if callable(response) else response


def plan(**changes):
    return {
        "query": "methods methodology",
        "intent": "locate",
        "needs_clarification": False,
        "reason": "Locate methodology",
        **changes,
    }


def accept():
    return {
        "decision": "accept",
        "reason": "Section answers the query",
        "recovery_query": "",
    }


@pytest.mark.parametrize(
    "question",
    [
        "does this paper mention any limitations",
        "Which section discusses limitations?",
        "Where are the limitations?",
    ],
)
def test_navigation_and_existence_prioritize_actual_section(question):
    from evidence_guard.extractive import retrieve

    index = SourceIndex(
        [
            Page(1, "Abstract\nOur limitations are mentioned briefly.\n"),
            Page(5, "VII. LIMITATIONS\nThe sample excludes rural locations.\n"),
        ],
        "synthetic",
    )
    assert retrieve(index, question, "overview")[0].page == 5


def test_recovery_retains_original_selected_witness_and_records_hosted_calls(tmp_path):
    index = SourceIndex(
        [Page(i, f"Topic {i}. Evidence {i}.") for i in range(1, 8)], "synthetic"
    )
    model = Model(
        [
            plan(query="evidence"),
            {"ids": ["s1"], "reason": "witness"},
            lambda payload: {"ids": ["s1"], "reason": "retained"},
        ]
    )

    class Hosted(Model):
        hosted = True
        last_events = [{"network_call": True, "status": "completed"}]

    judge = Hosted(
        [
            {
                "decision": "insufficient",
                "reason": "Need context",
                "recovery_query": "Topic 7",
                "passage_indices": [],
            },
            dict(accept(), passage_indices=["1"]),
        ]
    )
    result = run(
        "Explain evidence",
        index,
        model,
        Trace(tmp_path, "synthetic"),
        perf_counter() + 120,
        verifier=judge,
    )
    assert model.calls[1]["candidates"][0] == model.calls[2]["candidates"][0]
    assert result["review_packet"]["hosted_calls"] == 2
    assert result["route"] == "source_excerpt"


@pytest.fixture
def sample(tmp_path):
    index = SourceIndex(
        [
            Page(1, "I. INTRODUCTION\nBackground text.\n"),
            Page(2, "IV. METHODS\nA. Training\nTraining uses only past observations.\n"),
        ],
        "synthetic-doc",
    )
    return index, Trace(tmp_path, "synthetic")


def test_exact_original_source_and_original_query_survive_model_notes(sample):
    index, trace = sample
    pid = next(p.id for p in index.passages if "Training uses" in p.text)
    model = Model(
        [
            plan(),
            lambda payload: {
                "ids": [
                    next(
                        p["id"]
                        for p in payload["candidates"]
                        if "Training uses" in p["text"]
                    )
                ],
                "reason": "An invented rewritten answer.",
            },
        ]
    )
    verifier = Model([accept()])
    result = run(
        "Where is the methodology?",
        index,
        model,
        trace,
        perf_counter() + 120,
        verifier=verifier,
    )
    assert result["route"] == "source_excerpt"
    assert result["claims"][0]["text"] == next(
        p.text for p in index.passages if p.id == pid
    )
    assert "invented" not in result["claims"][0]["text"]
    assert result["claims"][0]["page"] == 2
    assert verifier.calls == []
    assert result["review_packet"]["original_question"] == "Where is the methodology?"
    assert result["checks"][-1]["rule"] == "matching_section_location"
    assert result["review_packet"]["hosted_calls"] == 0


def test_forged_selection_is_a_beta_failure_not_an_answer(sample):
    index, trace = sample
    model = Model([plan(), {"ids": ["invented-id"], "reason": ""}])
    result = run("Where is the methodology?", index, model, trace, perf_counter() + 120)
    assert result["route"] == "beta_flag"
    assert result["claims"] == []
    assert result["reason"] == "invalid_selection_output"


def test_missing_user_referent_is_flagged_after_preparation(sample):
    index, trace = sample
    model = Model([plan()])
    result = run("What does this mean?", index, model, trace, perf_counter() + 120)
    assert result["reason"] == "ambiguous_user_context"
    assert len(model.calls) == 1
    assert not result["claims"]


def test_rejected_passages_are_not_displayed_as_answer(sample):
    index, trace = sample
    model = Model([plan(), {"ids": ["s1"], "reason": ""}])
    verifier = Model(
        [{"decision": "irrelevant", "reason": "Only background", "recovery_query": ""}]
    )
    result = run(
        "Explain methodology.",
        index,
        model,
        trace,
        perf_counter() + 120,
        verifier=verifier,
    )
    assert result["route"] == "beta_flag"
    assert result["reason"] == "selection_check_failed"
    assert result["claims"] == []


def test_source_offsets_cannot_silently_be_rewritten(sample):
    index, _ = sample
    pid = index.passages[0].id
    index.spans[pid] = (0, 1)
    from evidence_guard.models import ModelError

    with pytest.raises(ModelError, match="source_span_mismatch"):
        render(index, [pid])


def test_numeric_sentences_and_reference_authors_are_not_headings():
    index = SourceIndex(
        [
            Page(
                1,
                "IV. M ETHODS\n0.9759. Equation (3) predicts the gain.\n"
                "H. Schwenk, and Y. Bengio, Reference title.\n",
            )
        ],
        "headings",
    )
    assert index.outline() == [{"page": 1, "title": "IV. METHODS"}]


def test_two_document_selections_keep_different_source_text():
    a = SourceIndex([Page(1, "I. POLICY\nThe reserve period is three days.")], "a")
    b = SourceIndex([Page(1, "I. POLICY\nThe reserve period is nine days.")], "b")
    assert a.passages[0].id != b.passages[0].id
    assert "three days" in render(a, [a.passages[0].id])[0]["text"]
    assert "nine days" in render(b, [b.passages[0].id])[0]["text"]


def test_bounded_duration_requires_the_same_subject_and_no_negation():
    from evidence_guard.extractive import objective_fit

    source = SourceIndex(
        [Page(1, "I. POLICY\nThe trial period is 12 calendar days.")], "a"
    )
    assert (
        objective_fit("How long is the trial period?", source.passages)
        == "explicit_duration_statement"
    )
    assert objective_fit("How long is the reserve period?", source.passages) is None
    unrelated = SourceIndex([Page(2, "II. OTHER\nUnrelated material.")], "other")
    assert (
        objective_fit(
            "How long is the trial period?", source.passages + unrelated.passages
        )
        is None
    )
    source = SourceIndex([Page(1, "I. POLICY\nThe trial period is not 12 days.")], "b")
    assert objective_fit("How long is the trial period?", source.passages) is None


def test_background_mention_is_not_an_objective_section_match():
    from evidence_guard.extractive import objective_fit

    source = SourceIndex(
        [Page(1, "III. BACKGROUND\nMethods are discussed elsewhere.")], "a"
    )
    assert objective_fit("Where are the methods?", source.passages) is None


def test_explicit_context_reaches_selection_and_semantic_audit(sample):
    index, trace = sample
    model = Model([plan(), {"ids": ["s1"], "reason": ""}])
    verifier = Model([accept()])
    result = run(
        "Explain this",
        index,
        model,
        trace,
        perf_counter() + 120,
        verifier=verifier,
        context="The training rule",
    )
    assert model.calls[1]["user_context"] == "The training rule"
    assert verifier.calls[0]["user_context"] == "The training rule"
    assert result["prepared_query"]["requirements"] == ["Explain this"]
