# Security and confidential data

Evidence Guard is a single-user loopback application. It does not implement user
authentication, tenant isolation, application-managed encryption at rest or
internet-facing deployment hardening. Keep the service and local model endpoints
on loopback and protect the external runtime with appropriate local access controls.

Local verification is the default CLI configuration. Explicit Groq configuration
transmits the original question, supplied context and selected excerpts to Groq.
For a confidential search, select the local-only checkbox **before** submitting;
local failure does not fall back to hosted judging. The preference belongs to the
current browser tab and is not permanent document metadata. Earlier transmissions
cannot be undone by changing it later.

Raw query/call records, source PDFs and hosted caches can contain confidential text.
Keep all of them outside Git and public issue/release attachments. Model weights
and credentials must also remain external. Never paste keys into shell arguments,
logs or a public report. Ignore rules do not remove already tracked objects from
Git history.

Exact-span checking establishes source provenance, not semantic correctness.
Untrusted documents can still influence source selection or auditing. Review source
pages for consequential decisions; the beta does not claim calibrated accuracy.

## Reporting

Do not disclose credentials, confidential documents or exploitable details in a
public issue. Use GitHub's private vulnerability reporting channel when it is
enabled. Otherwise request a private contact method without including sensitive
details. Reports should identify the affected release, environment, minimal
synthetic reproduction and observed behavior. Do not submit private runtime files.

The implemented controls, residual risks and production gaps are described in
[the technical implementation](docs/TECHNICAL_IMPLEMENTATION.md#9-security-and-privacy-threat-model).
