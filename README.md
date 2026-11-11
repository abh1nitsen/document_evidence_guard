# Drishti Evidence Guard

A local-first, source-only PDF assistant. A small local model prepares a question
and selects evidence. Application code verifies and copies the original PDF text;
a configurable local or Groq verifier assesses relevance to the original request.

**Current release: 0.11.0 beta.** Exact source text does not guarantee a correct or
complete answer. The application can miss evidence, reject answerable requests,
or misjudge relevance. Review the linked source pages.

## Documentation

- [Technical implementation](docs/TECHNICAL_IMPLEMENTATION.md): architecture,
  algorithms, contracts, API, storage, privacy, failure handling and operations.
- [Contributing](CONTRIBUTING.md): development, verification and release process.
- [Security](SECURITY.md): trust boundaries, private-data handling and reporting.
- [Changelog](CHANGELOG.md): current release behavior.

## Requirements

- Python 3.11 or newer; Python 3.12 is used for the reference development environment.
- [uv](https://docs.astral.sh/uv/) for the locked dependency environment.
- Existing OpenAI-compatible model servers on loopback. The application does not
  bundle, download or launch model weights.
- A searchable PDF text layer. OCR, visual figure interpretation and arbitrary
  table/layout reconstruction are not implemented.

## Run locally

Install the application dependencies:

```sh
uv sync --locked
```

Start the console after starting your local model servers:

```sh
uv run --locked evidence-guard --data-dir "EXTERNAL_PRIVATE_DIR" ui --workflow extractive --local-url http://127.0.0.1:18089/v1 --local-model qwen2.5-1.5b --verifier-url http://127.0.0.1:18090/v1 --verifier-model gemma-2-2b-it
```

Replace `EXTERNAL_PRIVATE_DIR` with a directory outside every Git checkout.
Open [the loopback console](http://127.0.0.1:8787/), upload a PDF, select it and ask
a question. Uploads are limited to 25 MiB and 500 pages. All model calls in this
configuration remain local. Keep the service bound to loopback.

## Optional Groq verification

Explicitly enable hosted verification with external key files:

```sh
uv run --locked evidence-guard --data-dir "EXTERNAL_PRIVATE_DIR" ui --workflow extractive --local-url http://127.0.0.1:18089/v1 --verifier-provider groq --groq-model openai/gpt-oss-120b --groq-key-file "EXTERNAL_KEY_FILE_1" --groq-key-file "EXTERNAL_KEY_FILE_2" --groq-daily-calls 200
```

This sends the original question, supplied context and selected source excerpts to
Groq. It does not upload the PDF file. The local selector retains source selection;
the hosted judge cannot generate the displayed answer. The daily cap counts network
attempts across configured credentials for that runtime directory, per UTC day. It
is not a token, cost or provider-quota guarantee. Separate keys do not prove separate
organizations or independent quotas.

For a confidential request, check **Confidential document — use local verification
only** before searching. The
request then uses the configured local verifier, bypasses Groq and its cache, and
never falls back to Groq after a local failure. The choice stays checked across
reloads and document changes in the same browser tab until unchecked. It is not a
permanent document classification or a preference shared across tabs.

The interface warns: **Local verification may be less reliable than the larger
Groq verifier. It can miss errors or reject valid evidence. Review the source
excerpts.** Earlier hosted transmissions are unaffected by a later local-only choice.

## Output and evidence

The default `extractive` workflow uses contract `source-selection-v1`. Models
select passage IDs; application code verifies page-character spans and copies the
original text with document/page links. Unapproved excerpts are kept separately in
a review panel. Missing evidence, unresolved context, invalid selections, provider
errors and failed checks produce a beta flag.

The original question is retained throughout preparation, retrieval and judging.
Retrieval supports bounded relationship phrases and spelling equivalents, with one
controller-owned recovery attempt. These search hypotheses are not permission to
substitute one role for another or approve an inferred answer.

Runtime PDFs, indexes, raw prompts/responses, query records, caches and telemetry
belong outside Git. The repository contains application code, synthetic tests and
generic documentation only. Do not attach confidential review packets to public
issues or pull requests.

## Verification

```sh
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests
uv run --locked pytest
node --check src/evidence_guard/static/app.js
uv run --locked python scripts/check_public_tree.py
uv build
```

The 0.11.0 application baseline passed 162 automated tests using synthetic/mocked
fixtures. That count is not an accuracy benchmark. Tests do not require private
PDFs, model weights or provider credentials. Independent semantic evaluation and
untouched document holdouts remain necessary before claiming broad generalization.

The Python compatibility API retains a legacy default: callers must explicitly
request `workflow="extractive"` and provide `document_path` for this contract.
Retained `working` and `legacy` workflows are diagnostic alternatives, not equivalent
source-only implementations. Optional semantic tools are not used by the active
extractive retrieval path. A reuse license has not yet been selected.
