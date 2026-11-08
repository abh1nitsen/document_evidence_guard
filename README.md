# Drishti Evidence Guard

Version 0.4.0 adds local request interpretation, requirement-based retrieval and
source sentence coverage checks to the HTML console. Exact definitions and bounded
arithmetic retain their direct paths. Every response records its build and routing reason.

## Run the console

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run from this checkout:

```sh
uv sync --locked --python 3.12
uv run --locked ruff check src tests
uv run --locked pytest
uv build
```

Keep the PDF, model weights, extracted index, credentials, evaluation labels,
and run databases outside all Git checkouts. This repository contains the
application, lockfile, synthetic regression tests and scripted synthetic demo.

Start a [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) server using an
instruction-tuned GGUF stored outside the checkout. The Windows validation uses
build `b6760` and
[Qwen2.5 1.5B Instruct Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9).
The GGUF SHA-256 is
`6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.

```sh
llama-server -m "EXTERNAL_MODEL.gguf" --host 127.0.0.1 --port 8080 -c 8192 -np 1 --alias qwen2.5-1.5b
uv run --locked evidence-guard --data-dir "EXTERNAL_PRIVATE_DIR" ui --pdf "EXTERNAL_REPORT.pdf"
```

Use `llama-server.exe` on Windows. The application commands are identical on
Windows, macOS and Linux. Windows is tested locally and Linux in CI. Actual
macOS execution has not been tested.

Open [the local console](http://127.0.0.1:8787/). Agent activity streams actions,
operational reasons, page IDs and status changes while a question runs. The roles
execute sequentially with an evidence selector and an optional separate query
planner. Route history lives in the current
tab. Metadata persists in the external SQLite database without questions, answers,
excerpts or credentials. Source links open the configured PDF in the same local
server. Extracted text and returned evidence remain confidential local data.

For a command-line question, omit the positional question to enter it interactively:

```sh
uv run --locked evidence-guard --data-dir "EXTERNAL_PRIVATE_DIR" ask --pdf "EXTERNAL_REPORT.pdf"
```

## Local interpretation and coverage

Enable `--planner-url` to use the new workflow with a separate Gemma planner.
The tested Windows model is Gemma-2-2B-Instruct Q4_K_M, SHA-256
`e0aee85060f168f0f2d8473d7ea41ce2f3230c1bc1374847505ea599288a7787`.
Keep it outside Git and run a second loopback server:

```sh
llama-server -m "EXTERNAL_GEMMA.gguf" --host 127.0.0.1 --port 18090 -c 4096 -np 1 --alias gemma-2-2b-it
uv run --locked evidence-guard --data-dir "EXTERNAL_PRIVATE_DIR" ui --pdf "EXTERNAL_REPORT.pdf" --planner-url http://127.0.0.1:18090/v1
```

The `request-plan-v3` planner receives the original request, the previous supported
subject (when supplied), up to 14 chapter headings, six candidate headings and two
100-character preliminary previews. It proposes a subject, standalone normalized
request, evidence requirements, up to two searches, and optional clarification
choices. These are hypotheses, not independently verified interpretations.
Gemma-2/3 instructions are merged into a user turn for their chat template.

The harness validates the schema, retains the original question, and repairs
explicit number or negation loss by restoring the original to the search plan.
This mechanical guard cannot establish semantic equivalence or detect every lost
qualifier. Several requirements can accompany one request. A definition request
with additional requirements cannot use the canonical definition promotion path.
Two proposed interpretations produce a focused choice before evidence assessment.

Retrieval fuses the original and normalized questions, complementary queries and
queries for missing evidence requirements. English Snowball stemming connects
inflections such as score/scoring. Nearby section prose can be included across one
page boundary. Up to two batches contain at most four unseen passages and 6,000
source characters each. The assessor sees the first 700 characters per passage,
split into source spans of at most 320 characters and labeled with integer IDs.
It selects one or two IDs per supported requirement; the harness copies the text.
It cannot supply generated answer claims or invent a quotation. A shared sentence
cannot silently complete distinct requirements: the harness retains one use and
requires separate support for the others. This is conservative and can withhold
valid multi-purpose sentences; it does not prove semantic independence.

Code checks IDs and provenance. Semantic coverage is model-assessed and can be
wrong. The console explicitly distinguishes the two. A relevant excerpt does not
complete a request when some required parts remain uncovered: missing parts drive
the second retrieval. Previously supported parts are retained. No unseen evidence,
two batches, model failure, or the remaining time budget stops the loop.

Planner calls use up to 16 seconds; assessment calls use up to 18 seconds within a
default 55-second question budget (`--seconds` changes it). Socket timeouts are not
a strict wall-clock deadline. Server speed and response buffering can add overhead.
The UI prepares instruction prefixes using two short synthetic local calls before
serving. These calls have an eight-output-token limit and separate warmup telemetry;
no report content enters warmup. Startup can take up to two 20-second call timeouts.

The router distinguishes ambiguity, retrieval gaps, failed interpretation, failed
execution and provisional model matches. Even a complete model match retains a
hosted-review candidate because the selector cannot independently approve its own
semantic judgments. An execution failure does not mean the
report lacks information. A potential Groq task is recorded only as a candidate;
**confidential report processing has no hosted execution path**. Policy-blocked
candidates are shown separately from actual hosted calls. Missing retrieved evidence
is not a reason to invent facts with a larger model. The fictional demo remains the
only live Groq path.

The UI shows the normalized request, required/covered/missing parts, citations,
clarification choices, operational reasons and build information. The previous
supported subject is carried in the current tab; this is not full conversational
memory. A failed or ambiguous result clears it. Historical evaluations from a
different source fingerprint are labeled accordingly. Prompt text, questions,
normalized queries and source excerpts are not persisted in SQLite telemetry.
Plans and results in explicit evaluation files remain outside Git.

Development checks found incorrect semantic matches despite valid source IDs,
including a standards paragraph selected for an experimental implementation request.
Automated routing/page/provenance pass rates therefore overestimate usefulness.
The model-matched indicator is diagnostic, not an accuracy score or permission to
release an interpreted answer. Broader held-out user testing remains necessary.

Without `--planner-url`, the compatibility path uses passage selection with source
provenance checks and a review label; it does not perform the new coverage loop.
This option is useful for comparison, not equivalent functionality.

## Release contracts

| Result | What the application establishes |
| --- | --- |
| Answered locally | A unique definition heading and its complete source paragraph match the term or glossary expansion. The paragraph is copied directly. |
| Calculated locally | The user supplied an explicit expression such as `calculate 100 * (1 - 2.5 / 100)`. A bounded arithmetic tool computes it. Document applicability requires review. |
| Source evidence | Displayed text is copied from identified source passages. Relevance, completeness and any interpretation require review. |
| Clarification needed | The subject, term or arithmetic input needs clarification. |
| Insufficient evidence | The reviewed passages did not establish an answer. |
| Local run failed | No useful outcome could be produced within the runtime contract. Failures retain retrieved material for inspection and identify the failed stage. |

The source index keeps stable passage IDs, numbered headings, paragraphs, table
headers and rows, and figure captions. Retrieval uses glossary aliases, English
Snowball stemming, BM25-style ranking and heading boosts. Short table captions are
joined with their adjacent column blocks. Large blocks are bounded passages.
Figures are represented by extracted text and captions. The PDF source link
allows the user to inspect visual content.

Definitions support articles, conversational wording, ordinary terms, acronyms,
and multiple words. A close unmatched heading prompts clarification.
Generic document-scope wording such as "in this context" is separated from the
definition term. Domain restrictions and additional questions remain part of the
request and cannot be answered by promoting a general definition.
Follow-ups can use the previous supported subject in the current tab.
Reasoning over a complete conversation remains outside the contract.

General responses remain source evidence requiring review, even if every requested
part has model-assessed support. Partial coverage and model failures display their
specific limitations and any retrieved material; the application does not claim
that unassessed or missing evidence proves absence from the report.

## Evaluate locally

Store a labeled JSON array outside Git, with entries shaped like this example:

```json
[
  {
    "id": "sample_definition",
    "category": "definition",
    "question": "What is a sample?",
    "routes": ["answer_local"],
    "contains": ["required reference phrase"],
    "forbidden": ["known incorrect statement"],
    "pages": [4]
  }
]
```

The optional `topic` field supplies an established subject for a follow-up.
A string `intent` label checks the semantic planner's classification when required.
Optional `coverage_required` and `coverage_missing` arrays check requirement labels;
`routing_reason` checks the final reason code. `need_contains` maps a requirement
to required content fragments, so a quoted formula cannot pass a steps label merely
by pointing at an allowed page. Coverage labels still need human
review for relevance; model-assessed support does not establish correctness.
A numeric `calculation` field specifies an expected arithmetic result.
Run the suite through the same entry point used by the console:

```sh
uv run --locked evidence-guard --data-dir "EXTERNAL_PRIVATE_DIR" evaluate --pdf "EXTERNAL_REPORT.pdf" --suite "EXTERNAL_LABELS.json"
```

The external report includes every result, failed checks by case, category
counts, route counts, median and p95 latency, model failures and source-contract
checks. It fingerprints the source code, suite and PDF and records Python,
pypdf, package and model alias. A source change during evaluation is disclosed.
The server alias alone does not prove the deployed model artifact. Reproduce
the model revision, hash and llama.cpp build as well.

These checks measure routing, required content, provenance and the extractive
contract. They do not independently measure semantic accuracy. Human review
of relevance, completeness and supported interpretations remains necessary.
Evaluation files contain report-derived content and stay outside Git.
The evaluation command exits with a failure status when checks fail or the
source, document or labels change during the run.

## Measure hosted calls and scale scenarios

Groq is available only to the built-in fictional demo. Offline execution is
the default:

```sh
uv run --locked evidence-guard --data-dir "EXTERNAL_PRIVATE_DIR" demo
```

Set `GROQ_API_KEY` from a local secret store. The live cascade makes a hosted call
for the scripted interpretation case, while the baseline calls Groq for each
fixture. Use a fresh external sample directory for each matched comparison:

```sh
uv run --locked evidence-guard --data-dir "EXTERNAL_SAMPLE_DIR" demo --mode cascade --live
uv run --locked evidence-guard --data-dir "EXTERNAL_SAMPLE_DIR" demo --mode baseline --live
uv run --locked evidence-guard --data-dir "EXTERNAL_SAMPLE_DIR" report --monthly-questions 100000 --expected-frontier-rate 0.10 --local-cost-per-question 0.00001
```

To expose the live synthetic button in the console, start the server with
`--enable-live-demo` and the environment key. Confidential report questions
have no hosted execution path.

Each hosted attempt is recorded before dispatch, followed by status, elapsed
time, provider request ID, input/output/cached tokens, and estimated list cost
when usage is available. Unknown usage is counted separately and blocks a
cost comparison. Local call timings and token usage are recorded separately.
Version 2 report traffic has its own telemetry mode.

Scale projections use the measured mean hosted call costs and an explicitly
assumed escalation rate and local cost. Rates come from the dated
[Groq model card](https://console.groq.com/docs/model/openai/gpt-oss-120b) snapshot.
The small scripted demo verifies instrumentation. A representative evaluated
workload is required to estimate production quality, coverage and cost per
correctly resolved question.
