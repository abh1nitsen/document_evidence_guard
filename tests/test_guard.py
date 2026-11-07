import json
from io import BytesIO

import pytest

from evidence_guard.cli import outside_repository
from evidence_guard.demo import run_demo
from evidence_guard.evaluation import evaluate
from evidence_guard.evidence import Index, Page, definition_target
from evidence_guard.models import ChatResult, LocalModel, ModelError
from evidence_guard.models import chat as model_chat
from evidence_guard.telemetry import Trace, cost, summarize
from evidence_guard.workflow import anchored_claims, calculate, local_answer


@pytest.mark.parametrize(
    "url",
    ["https://api.groq.com/openai/v1", "http://127.0.0.1@elsewhere.invalid/v1"],
)
def test_confidential_adapter_rejects_nonlocal_destinations(url):
    with pytest.raises(ValueError, match="Local model URL"):
        LocalModel(url, "local")


def test_data_boundary_resolves_checkout_marker(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    (checkout / ".git" / "HEAD").touch()
    with pytest.raises(ValueError, match="outside"):
        outside_repository(checkout / "runs")
    assert outside_repository(tmp_path / "runs") == tmp_path / "runs"


def test_fabricated_number_fails_even_with_real_page():
    source = [Page(4, "The experimental reading was 31.25 points.")]
    draft = {"claims": [{"page": 4, "text": "32.25"}]}
    assert anchored_claims(draft, source) == []
    assert Index(source, "synthetic").fetch("31.25")[0].number == 4


def test_truncated_json_is_reported_as_a_generation_limit(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            return BytesIO(
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": '{"claims":['},
                            }
                        ],
                    }
                ).encode()
            )

    monkeypatch.setattr("evidence_guard.models.build_opener", lambda *args: Opener())
    with pytest.raises(ModelError, match="model_output_truncated"):
        model_chat("http://127.0.0.1:1/v1/chat/completions", "test", [], 1)


def test_conversational_definition_finds_complete_definition_before_broad_mentions():
    glossary = Page(1, "Abbreviations\n ABC    Asset Balance Code")
    broad = Page(2, "ABC appears in several charts. This page can tell you some context.")
    definition = Page(
        4,
        "1.2 Scope\nABC is used throughout this project.\n\n"
        "1.3.1 Asset Balance Code ABC\n"
        "The ABC is a synthetic rating from 0 to 50 that measures asset balance.",
    )
    index = Index([glossary, broad, definition], "synthetic")
    question = "i dont know what abc is, can you tell me"
    assert index.fetch(question)[0].number == 4
    excerpt = index.excerpts([definition], question)[0].text
    assert "rating from 0 to 50" in excerpt
    assert excerpt.startswith("1.3.1")
    assert index.definition("What is the whole-video mean score?") is None
    assert definition_target("What does ABC mean?") == "abc"


def test_explicit_definition_uses_source_without_a_model_call(tmp_path):
    class NoModel:
        model = "must-not-be-called"

        def run(self, *args, **kwargs):
            pytest.fail("An exact source definition should not invoke a model")

    index = Index(
        [
            Page(1, "Glossary\n ABC    Asset Balance Code"),
            Page(
                4,
                "1.3.1 Asset Balance Code ABC\n"
                "ABC is a synthetic rating from 0 to 50. It requires field review.",
            ),
        ],
        "synthetic",
    )
    trace = Trace(tmp_path, "confidential_local")
    result = local_answer(
        "i dont know what abc is, can you tell me",
        index,
        NoModel(),
        trace,
        2,
    )
    assert result["route"] == "answer_local"
    assert result["claims"][0]["page"] == 4
    assert "requires field review" in result["claims"][0]["text"]
    assert any(e.get("reason") == "exact_source_definition" for e in trace.events)


def test_local_loop_can_recover_with_targeted_fetch(tmp_path):
    class Model:
        model = "fake"
        drafted = 0

        def run(self, instruction, payload, timeout, schema=None):
            self.drafted += 1
            if self.drafted == 1:
                return {"selected_ids": [], "status": "missing"}
            return {
                "selected_ids": [payload["passages"][0]["id"]],
                "status": "relevant",
            }

    trace = Trace(tmp_path, "confidential_local")
    result = local_answer(
        "Summarize the waiver.",
        Index(
            [Page(i, f"The waiver applies on day {i}.") for i in range(1, 9)],
            "synthetic",
        ),
        Model(),
        trace,
        2,
    )
    assert result["route"] == "evidence_only"
    assert result["claims"][0]["text"] == result["claims"][0]["quote"]
    assert [e["new_pages"] for e in trace.events if e["kind"] == "fetch"] == [
        [1, 2, 3, 4, 5, 6],
        [7, 8],
    ]


def test_live_fixture_counts_calls_and_prices_a_matched_baseline(tmp_path, monkeypatch):
    seen = []

    def fake_chat(endpoint, model, messages, timeout, key):
        prompt = messages[-1]["content"]
        seen.append(prompt)
        answer = (
            "17:30"
            if "badge desk" in prompt
            else "unknown"
            if "head count" in prompt
            else "yes"
        )
        return ChatResult(
            json.dumps({"answer": answer}),
            {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 200},
            },
            "synthetic-test-id",
        )

    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr("evidence_guard.demo.chat", fake_chat)
    cascade = run_demo(tmp_path, "cascade", live=True)
    run_demo(tmp_path, "baseline", live=True)
    assert [result["route"] for result in cascade] == [
        "answer_local",
        "answer_frontier",
        "insufficient_evidence",
    ]
    assert len(seen) == 4
    events = [event["kind"] for event in cascade[1]["events"]]
    assert events.index("route") < events.index("frontier_start")
    report = summarize(tmp_path, volume=300_000, local_cost=0, expected_rate=None)
    assert report["projection"]["avoided_frontier_calls_at_scale"] == pytest.approx(
        200_000
    )
    assert report["projection"]["api_savings_usd"] == pytest.approx(87.0)
    assert cost(None) is None


@pytest.mark.parametrize("failure", ["model_timeout", "invalid_model_json"])
def test_activity_is_emitted_before_model_work_and_failure_is_distinct(tmp_path, failure):
    observed = []

    class TimeoutModel:
        model = "test-local"

        def run(self, instruction, payload, timeout, schema=None):
            assert observed[-1]["kind"] == "activity"
            assert observed[-1]["status"] == "running"
            raise ModelError(failure)

    trace = Trace(tmp_path, "confidential_local", on_event=observed.append)
    result = local_answer(
        "Explain the sample.",
        Index([Page(1, "A sample definition.")], "test"),
        TimeoutModel(),
        trace,
        2,
    )
    assert result["route"] == "evidence_only"
    assert result["selection_method"] == "lexical_fallback_after_" + failure
    assert any(event.get("status") == "error" for event in observed)
    assert all(event["trace_id"] == trace.id for event in observed)


@pytest.mark.parametrize(
    "question",
    [
        "what is an incident",
        "Define incident.",
        "Explain the incident",
        "I don't know what incident is, can you tell me",
    ],
)
def test_ordinary_definition_paraphrases_preserve_qualifiers(tmp_path, question):
    index = Index(
        [
            Page(
                2,
                "1.4.2 Incident\n"
                "Incident refers to a recorded test event. It requires review.",
            )
        ],
        "synthetic",
    )
    claim = index.definition(question)
    assert claim["page"] == 2
    assert "requires review" in claim["text"]
    assert claim["quote"] in " ".join(index.pages[0].text.split())


def test_self_approval_cannot_release_invented_answer(tmp_path):
    class UnreliableModel:
        model = "fake"

        def run(self, *args, **kwargs):
            return {
                "claims": [{"page": 3, "text": "An invented unsupported answer."}],
                "supported": True,
                "complete": True,
            }

    result = local_answer(
        "Who created the sample?",
        Index([Page(3, "The sample is a synthetic measurement.")], "test"),
        UnreliableModel(),
        Trace(tmp_path, "test"),
        2,
    )
    assert result["route"] == "evidence_only"
    assert all("invented" not in c["text"] for c in result["claims"])
    assert result["verification"] == "exact_excerpt_only"


def test_followup_without_subject_requests_clarification(tmp_path):
    result = local_answer(
        "Who invented it?", Index([], "test"), None, Trace(tmp_path, "test"), 2
    )
    assert result["route"] == "ask_clarification"


def test_explicit_arithmetic_uses_bounded_tool():
    assert calculate("calculate 100 * (1 - 2.5 / 100)")["value"] == 97.5
    with pytest.raises(ZeroDivisionError):
        calculate("calculate 1 / 0")
    with pytest.raises(ValueError):
        calculate("calculate __import__('os')")


def test_table_caption_header_rows_and_note_remain_together():
    page = Page(
        5,
        "Table 2.1 Synthetic records\n\n"
        "Item       Quantity\n\nA          2\n\nB          3\n\n"
        "Source: fictional illustration, not field data.",
    )
    passages = Index([page], "synthetic").passages
    assert len(passages) == 1
    assert passages[0].kind == "table"
    assert "Quantity" in passages[0].text
    assert "not field data" in passages[0].text
    assert passages[0].text in " ".join(page.text.split())


def test_conflicting_definitions_request_clarification(tmp_path):
    index = Index(
        [
            Page(1, "1.1 Incident\nAn incident is a test event."),
            Page(2, "2.1 Incident\nAn incident is a production alarm."),
        ],
        "synthetic",
    )
    result = local_answer("What is an incident?", index, None, Trace(tmp_path, "test"), 2)
    assert result["route"] == "ask_clarification"


def test_external_evaluation_catches_a_fabricated_quote(tmp_path, monkeypatch):
    suite = tmp_path / "cases.json"
    suite.write_text(
        json.dumps(
            [
                {
                    "id": "fabrication",
                    "question": "Explain the sample.",
                    "routes": ["evidence_only"],
                    "pages": [2],
                }
            ]
        ),
        encoding="utf-8",
    )
    document = tmp_path / "source.pdf"
    document.write_bytes(b"synthetic")
    import hashlib

    fingerprint = hashlib.sha256(b"synthetic").hexdigest()
    monkeypatch.setattr(
        "evidence_guard.evaluation.load_pdf",
        lambda *args: (
            Index([Page(2, "A fictional sample.")], fingerprint),
            False,
        ),
    )

    def ask(*args, **kwargs):
        return {
            "route": "evidence_only",
            "claims": [
                {
                    "page": 2,
                    "text": "Fabricated source.",
                    "quote": "Fabricated source.",
                }
            ],
            "events": [{"kind": "finish", "elapsed_ms": 1}],
        }

    result = evaluate(suite, tmp_path, ask, document)
    assert result["summary"]["passed"] == 0
    assert result["summary"]["provenance_failures"] == 1
    assert not result["manifest"]["document_changed_during_run"]


def test_offline_demo_has_no_network_attempts(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Offline demo attempted a network request")

    monkeypatch.setattr("evidence_guard.demo.chat", fail)
    run_demo(tmp_path, "cascade", live=False)
    assert (
        summarize(tmp_path, 10, 0, None)["by_mode"]["cascade_offline"]["calls_started"]
        == 0
    )
