# Changelog

Only implemented application behavior is listed. Private evaluation material and
development call records are maintained separately.

## 0.11.0

- Added a per-search confidential local-only checkbox to the hosted-enabled console.
- Routed checked requests to the configured local verifier without hosted cache
  access or automatic hosted fallback.
- Added a visible local-verification reliability disclaimer, actual-provider
  result metadata and same-tab preference retention.
- Validated the override as a strict boolean at the API/controller boundary.

## 0.10.0

- Added bounded relationship phrase searches and explicit spelling equivalents.
- Reduced selected identity-question scaffolding in lexical search strings while
  preserving the original question for semantic checks.
- Added controller-owned recovery for empty/rejected selections, duplicate-bank
  detection and private retrieval match/provenance records.

## 0.9.0

- Added explicitly enabled Groq excerpt verification with strict subset approval.
- Added ordered credential handling, quota cooldowns, attempt limits and exact
  successful-response caching, with fail-closed errors.

## 0.8.0

- Introduced source-only selection and exact page-character-span rendering.
- Retained original requests, immutable source IDs and document/page links.
- Separated unapproved review excerpts from released answers.
