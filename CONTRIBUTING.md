# Contributing

Use Python 3.11+ and the checked-in `uv.lock`. Install with `uv sync --locked`.
Core tests use synthetic documents and mocked model/provider adapters; no model
server, private input or real credential should be necessary.

## Verify a change

```sh
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests
uv run --locked pytest
node --check src/evidence_guard/static/app.js
uv run --locked python scripts/check_public_tree.py
uv build
```

Use a fresh external pytest base directory when the platform's temporary directory
requires it. Pytest can clear its base directory: never use an evidence directory
or a directory containing documents as `--basetemp`.

## Change requirements

- Preserve the original question, document binding, source offsets and data-transfer
  choice. Search expansion is a hypothesis, not a new user requirement.
- Keep the extractive answer path source-only. Do not introduce generated answer
  prose or bypass relevance checks to make a regression pass.
- Add meaningful boundary/contract tests for behavioral changes. Use synthetic
  fixtures with contrasting roles, qualifications, negation and no-answer cases.
- Confidential-mode changes require both success and failure tests proving the
  hosted adapter is never invoked, including cache access or fallback.
- Distinguish mocked tests, real model runs, semantic label review and untouched
  holdouts. Model agreement is not ground truth. Preserve failed runs and disputed
  labels outside Git rather than replacing them with later successes.
- Update architecture/API documentation and version/lockfile metadata when behavior
  changes. Keep PR descriptions focused on the problem, resulting behavior,
  verification and remaining limitations.

## Public-data policy

Commit application code, generic documentation and synthetic fixtures only. PDFs,
raw model calls, extracted content, question banks, labels, review packets, caches,
telemetry databases, model weights and credentials remain outside every checkout.
Do not attach them to public issues, pull requests, CI artifacts or releases.

The public-tree check detects common secret/file/path mistakes. Before publishing,
also run it over every reachable commit with an external private deny list:

```sh
uv run --locked python scripts/check_public_tree.py --revision HEAD --deny-file "EXTERNAL_DENY_LIST.json"
```

The deny file is a JSON array of nonempty strings to exclude. It must remain
outside Git. Matches report category/path/object identity without printing matched
secret or document text. This check supplements human review; it is not a guarantee
that arbitrary private material can be recognized automatically.

## Release procedure

1. Freeze the candidate source, configuration and synthetic/approved external cases.
2. Run verification and preserve failures in private immutable records.
3. Build the source distribution and wheel; compare package bytes and record hashes.
4. Audit the tip, complete reachable history, branches, tags and release metadata.
5. If private development ancestry exists, preserve it locally and publish an
   audited source snapshot descended only from approved public history. Push only
   the reviewed ref, never `--all` or `--mirror` from a private development clone.
6. Verify remote commit/tree, visibility, full fresh clone and CI results. Do not
   describe a push, public visibility or CI outcome as completed before checking it.

The local console is a single-user beta. Authentication, remote serving, parser
isolation, semantic calibration and production SLOs require a separate design and
review; they are not implied by a passing build.
