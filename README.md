# Drishti Evidence Guard

Version 0.3.0 provides a local HTML console, exact document definitions, bounded
arithmetic, and a local SLM that selects evidence passages. Every report response
has an explicit release contract and a trace of each workflow role.

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

## Optional local semantic planner

The optional Gemma stage interprets conversational requests before retrieval.
Exact definitions and arithmetic bypass this stage. Other requests send only the
question and up to ten short document headings to the local planner. Its bounded
JSON output contains an intent, a term, up to two retrieval queries and a reason
code. The console displays this interpretation and each model's activity.

The existing Windows model used for validation is Gemma-2-2B-Instruct Q4_K_M,
SHA-256 `e0aee85060f168f0f2d8473d7ea41ce2f3230c1bc1374847505ea599288a7787`.
Keep it outside Git and run a second loopback server:

```sh
llama-server -m "EXTERNAL_GEMMA.gguf" --host 127.0.0.1 --port 18090 -c 4096 -np 1 --alias gemma-2-2b-it
uv run --locked evidence-guard --data-dir "EXTERNAL_PRIVATE_DIR" ui --pdf "EXTERNAL_REPORT.pdf" --planner-url http://127.0.0.1:18090/v1
```

Gemma-2/3 instructions are merged into a user turn for their chat template.
The original query remains in reciprocal-rank fusion and its top retrieval
candidate is retained. Invalid plans, timeouts, or dropped explicit numbers or
negation fall back to the original query. This validation does not establish
semantic equivalence of rewrites; general results still require source review.
Canonical definition promotion additionally requires that the original request
contains the term or glossary alias, has no unhandled material qualifiers, and
matches a unique source definition. The planner cannot generate answer claims.

Planner latency and tokens are logged separately from retrieval and selection.
Successful plans are transient browser/CLI output; persistent telemetry stores
only reason codes and counts. Evaluation reports may contain the plans and
remain outside Git. With page labels, evaluation compares original and fused
top-six reference-page ranks. A small development comparison measures behavior,
not general semantic accuracy or production savings.

The initial eight-case private development run passed seven automated checks.
All three conversational definitions were released from exact source definitions;
one limitations retrieval missed its labeled page. Across six page-labeled plans,
reference-page rank improved in two cases, stayed the same in three and remained
outside the top six in one. Three selector calls timed out and used labeled
retrieval fallbacks. Mixed-workload median was 9.40 seconds and p95 19.38 seconds.
These figures describe this Windows run and its small development suite only.
They do not establish production readiness or a reduction in hosted spend.

## Release contracts

| Result | What the application establishes |
| --- | --- |
| Answered locally | A unique definition heading and its complete source paragraph match the term or glossary expansion. The paragraph is copied directly. |
| Calculated locally | The user supplied an explicit expression such as `calculate 100 * (1 - 2.5 / 100)`. A bounded arithmetic tool computes it. Document applicability requires review. |
| Source evidence | Displayed text is copied from identified source passages. Relevance, completeness and any interpretation require review. |
| Clarification needed | The subject, term or arithmetic input needs clarification. |
| Insufficient evidence | The reviewed passages did not establish an answer. |
| Local run failed | No useful outcome could be produced within the runtime contract. Model failures with retrieved passages can return source evidence through a labeled lexical fallback. |

The source index keeps stable passage IDs, numbered headings, paragraphs, table
headers and rows, and figure captions. Retrieval uses glossary aliases, plural
normalization, BM25-style ranking and heading boosts. Short table captions are
joined with their adjacent column blocks. Large blocks are bounded passages.
Figures are represented by extracted text and captions. The PDF source link
allows the user to inspect visual content.

Definitions support articles, conversational wording, ordinary terms, acronyms,
and multiple words. A close unmatched heading prompts clarification.
Generic document-scope wording such as "in this context" is separated from the
definition term. Domain restrictions and additional questions remain part of the
request and cannot be answered by promoting a general definition.
Follow-ups can use the last definition's subject in the current tab. An unresolved pronoun
requests clarification. Multi-turn reasoning over earlier answer details remains
outside the conversation contract.

For other questions, the SLM selects IDs from schema-constrained candidates.
The harness rejects unknown or duplicate IDs and copies the corresponding
source text. It preserves the top lexical candidate as a retrieval anchor.
The model cannot publish a generated paraphrase or approve its own answer.
The UI labels these results as evidence requiring review.

The loop tries at most two batches of up to six passages with a 6,000-character
source budget. The SLM sees short, explicitly truncated previews and integer
selection IDs. Returned evidence retains the complete selected source passage.
Only unseen passages justify another retrieval. Model errors stop selection
and can fall back to retrieved evidence. Each model call has a 12-second ceiling
inside the overall question budget, which defaults to 35 seconds. Runtime
performance depends on hardware and model configuration.

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
