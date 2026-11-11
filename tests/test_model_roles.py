"""CLI-to-workflow routing for default and explicitly reversed model roles."""

import json

import pytest
from test_documents import pdf_bytes

from evidence_guard import cli
from evidence_guard.groq_verifier import GroqVerifier
from evidence_guard.models import LocalModel, ModelError


def test_hosted_verification_keeps_selection_local_and_blocks_reverse(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "synthetic.pdf"
    pdf.write_bytes(pdf_bytes("Prepared for Mira Patel."))
    key = tmp_path / "key.txt"
    key.write_text("gsk_" + "0" * 40)
    observed = []

    def answer(question, index, model, *args, verifier=None, **kwargs):
        assert model.model == "qwen2.5-1.5b"
        assert verifier.hosted
        return {"route": "beta_flag", "claims": []}

    def warmup(*args, **kwargs):
        assert kwargs["verifier"] is None

    def serve(port, ask, *args, **kwargs):
        assert kwargs["hosted_verification"]
        assert "reversed" not in kwargs["model_roles"]
        document_id = kwargs["documents"].default_id
        observed.append(ask("Who is this for?", document_id=document_id))
        with pytest.raises(ValueError, match="Hosted verification"):
            ask("Who is this for?", document_id=document_id, role_assignment="reversed")

    monkeypatch.setattr(
        "sys.argv",
        [
            "evidence-guard",
            "--data-dir",
            str(tmp_path / "runtime"),
            "ui",
            "--pdf",
            str(pdf),
            "--verifier-provider",
            "groq",
            "--groq-key-file",
            str(key),
        ],
    )
    monkeypatch.setattr(cli, "local_answer", answer)
    monkeypatch.setattr(cli, "prepare", warmup)
    monkeypatch.setattr(cli, "serve", serve)
    cli.main()
    assert len(observed) == 1


@pytest.mark.parametrize("swap_default", [False, True])
def test_console_role_defaults_and_per_question_override(
    tmp_path, monkeypatch, swap_default
):
    pdf = tmp_path / "synthetic.pdf"
    pdf.write_bytes(pdf_bytes("This document was prepared for Mira Patel."))
    data = tmp_path / "runtime"
    calls = []

    def answer(question, index, model, *args, verifier=None, **kwargs):
        calls.append((model.model, verifier.model))
        return {"route": "human_review", "claims": [], "draft_answer": []}

    def serve(port, ask, *args, model_roles=None, **kwargs):
        assert model_roles["default"] == ("reversed" if swap_default else "primary")
        document_id = kwargs["documents"].default_id
        first = ask("Who is the recipient?", document_id=document_id)
        assert first["role_assignment"] == model_roles["default"]
        ask("Who is the recipient?", document_id=document_id, role_assignment="primary")
        ask("Who is the recipient?", document_id=document_id, role_assignment="reversed")

    argv = [
        "evidence-guard",
        "--data-dir",
        str(data),
        "ui",
        "--pdf",
        str(pdf),
        "--local-url",
        "http://127.0.0.1:18089/v1",
    ]
    if swap_default:
        argv.append("--swap-model-roles")
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(cli, "local_answer", answer)
    monkeypatch.setattr(cli, "prepare", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "serve", serve)
    cli.main()
    primary = ("qwen2.5-1.5b", "gemma-2-2b-it")
    assert calls == [primary[::-1] if swap_default else primary, primary, primary[::-1]]
    records = [json.loads(p.read_text()) for p in (data / "query-records").glob("*.json")]
    assert len(records) == 3
    for record in records:
        assignment = record["role_assignment"]
        expected = primary if assignment == "primary" else primary[::-1]
        assert record["model_roles"]["answer"]["model"] == expected[0]
        assert record["model_roles"]["verifier"]["model"] == expected[1]


@pytest.mark.parametrize("local_failure", [False, True])
def test_confidential_request_never_uses_hosted_adapter_or_cache(
    tmp_path, monkeypatch, local_failure
):
    pdf = tmp_path / "synthetic.pdf"
    pdf.write_bytes(pdf_bytes("The Cedar event is run by Birch Foundation."))
    key = tmp_path / "key.txt"
    key.write_text("gsk_" + "0" * 40)
    data = tmp_path / "runtime"
    local_calls, hosted_calls = [], []

    def local_run(self, instruction, payload, *args, schema=None, **kwargs):
        local_calls.append(self.model)
        if "intent" in schema["properties"]:
            return {
                "query": "Cedar organizer",
                "intent": "identify",
                "needs_clarification": False,
                "reason": "event role",
            }
        if "ids" in schema["properties"]:
            return {"ids": ["s1"], "reason": "Explicit role"}
        if local_failure:
            raise ModelError("local_unavailable")
        return {
            "decision": "accept",
            "reason": "Explicit event role",
            "recovery_query": "",
        }

    def hosted_run(self, instruction, payload, *args, **kwargs):
        hosted_calls.append(payload)
        return {
            "decision": "accept",
            "reason": "Explicit role",
            "recovery_query": "",
            "passage_indices": ["1"],
        }

    def serve(port, ask, *args, **kwargs):
        assert kwargs["local_verification"]["verifier"] == "gemma-2-2b-it"
        doc = kwargs["documents"].default_id
        result = ask("Who is the organizer of Cedar?", document_id=doc, local_only=True)
        assert not hosted_calls
        assert result["local_only"] is True
        assert result["verification_provider"] == "local"
        assert result["model_roles"]["verifier"]["model"] == "gemma-2-2b-it"
        assert result["review_packet"]["hosted_calls"] == 0
        assert "less reliable" in result["verification_notice"]
        assert result["review_packet"]["verification_provider"] == "local"
        assert result["route"] == ("beta_flag" if local_failure else "source_excerpt")
        assert local_calls[-1] == "gemma-2-2b-it"
        # This later request explicitly re-enables the configured hosted mode.
        hosted_result = ask(
            "Who is the organizer of Cedar?", document_id=doc, local_only=False
        )
        assert len(hosted_calls) == 1
        assert hosted_result["verification_provider"] == "groq"
        assert "verification_notice" not in hosted_result
        for invalid in ("true", "false", 1, None, [], {}):
            with pytest.raises(ValueError, match="true or false"):
                ask("Who is the organizer?", document_id=doc, local_only=invalid)

    monkeypatch.setattr(
        "sys.argv",
        [
            "evidence-guard",
            "--data-dir",
            str(data),
            "ui",
            "--pdf",
            str(pdf),
            "--verifier-provider",
            "groq",
            "--groq-key-file",
            str(key),
        ],
    )
    monkeypatch.setattr(LocalModel, "run", local_run)
    monkeypatch.setattr(GroqVerifier, "run", hosted_run)
    monkeypatch.setattr(cli, "prepare", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "serve", serve)
    cli.main()
    records = [json.loads(p.read_text()) for p in (data / "query-records").glob("*.json")]
    private = next(r for r in records if r["local_only"])
    assert private["verification_provider"] == "local"
    assert private["result"]["review_packet"]["verification_provider"] == "local"
    assert all(e.get("kind") != "hosted_model" for e in private["result"]["events"])
