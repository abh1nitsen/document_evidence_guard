"use strict";

const byId = (id) => document.getElementById(id);
const runs = [];
let selected = null;
let activeRun = null;
let conversationTopic = null;
let currentDocument = null;
let documentLibrary = [];
let requestInProgress = false;
let verificationConfig = null;
try { byId("local-only").checked = sessionStorage.getItem("evidence-guard-local-only") === "true"; } catch {}
let maxUploadBytes = 25 * 1024 * 1024;
const roleNames = {
  planner: "Planner", retriever: "Retriever", context: "Context",
  writer: "Writer", verifier: "Verifier", router: "Router",
};
const routeLabels = {
  source_excerpt: ["Original source excerpts · beta", "local"],
  beta_flag: ["Beta flag · review needed", "open"],
  answer_local: ["Answered locally", "local"],
  answer_frontier: ["Answered via Groq", "frontier"],
  would_call: ["Would call Groq", "frontier"],
  partial: ["Partial answer", "open"],
  insufficient_evidence: ["Insufficient evidence", "open"],
  provider_error: ["Groq failed", "open"],
  frontier_unverified: ["Groq answer withheld", "open"],
  local_error: ["Local run failed", "open"],
  running: ["Running", "local"],
  evidence_only: ["Source evidence · review needed", "open"],
  human_review: ["Draft · human review needed", "open"],
  ask_clarification: ["Clarification needed", "open"],
  calculation_local: ["Calculated locally", "local"],
};
const reasons = {
  exact_source_definition: "The report explicitly defines the requested term. The source definition was copied directly.",
  deterministic_arithmetic: "An explicit arithmetic expression was evaluated locally.",
  interpretation_requires_review: "The source passages are available, but a complete interpretation requires review.",
  missing_conversation_subject: "The follow-up does not have an established subject.",
  no_established_answer: "The reviewed passages did not establish an answer.",
  ambiguous_term: "The term needs clarification before answering.",
  local_verified: "The local evidence checks passed.",
  missing_source_fact: "The source does not contain the requested fact.",
  cross_section_interpretation_unresolved: "The cross-section interpretation remained open.",
  confidential_document_local_only: "Report questions use the local model only.",
  offline_demo: "The demo simulated this route without making a Groq call.",
};
const plain = (value) => String(value ?? "").replaceAll("_", " ");
const milliseconds = (value) => value >= 1000
  ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(1)} ms`;
const eventOf = (run, kind) => run.events.find((event) => event.kind === kind);
const sourceURL = (run, page) => `/api/document${run.document?.id ? `?document_id=${encodeURIComponent(run.document.id)}` : ""}#page=${page}`;

function chooseDocument(id) {
  const next = documentLibrary.find((doc) => doc.id === id) || null;
  if (next?.id !== currentDocument?.id) {
    conversationTopic = null;
    byId("context").value = "";
  }
  currentDocument = next;
  byId("document-select").value = next?.id || "";
  byId("search-button").disabled = requestInProgress || !next;
  byId("document-status").textContent = next
    ? `${next.name} · ${next.searchable_pages}/${next.pages} searchable pages. ${next.warning || ""}`
    : "Upload or choose a PDF before asking a question.";
}

function drawDocuments(selectedId) {
  const select = byId("document-select");
  const placeholder = element("option", "Choose or upload a PDF");
  placeholder.value = "";
  select.replaceChildren(placeholder);
  for (const doc of documentLibrary) {
    const option = element("option", doc.name);
    option.value = doc.id;
    select.append(option);
  }
  chooseDocument(selectedId);
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function setBadge(node, run) {
  const [label, tone] = routeLabels[run.route] || [plain(run.route), "open"];
  node.textContent = label;
  node.className = `badge ${tone}`;
  node.hidden = false;
}

function describe(event) {
  const pages = event.pages?.length ? ` · pages ${event.pages.join(", ")}` : "";
  const duration = Number.isFinite(event.duration_ms)
    ? ` · took ${milliseconds(event.duration_ms)}` : "";
  switch (event.kind) {
    case "activity": return `${roleNames[event.role] || event.role} · ${event.action} · ${event.reason}`;
    case "index_load": return `Report index ${event.cache_hit ? "reused" : "built"}${duration}`;
    case "fetch": return `Fetch ${event.attempt} · ${plain(event.strategy || "synthetic")}${event.adjacent_context ? " · surrounding prose included" : ""}${pages}${duration}`;
    case "local_model": return `Local model · ${event.model || "selector"} · ${plain(event.stage)}${duration}`;
    case "hosted_model": return `Hosted verifier · ${event.model}${duration}`;
    case "work_tool": return `Document tool · ${plain(event.tool)} · ${event.status}${pages}${duration}`;
    case "work_state": return `Workspace · ${event.requirements} requirements · ${event.read_passages} read passages · ${event.gaps} gaps`;
    case "work_check": return `Answer checks · ${plain(event.basis)} · ${event.passed ? "passed within stated scope" : "recovery or review required"}`;
    case "query_understanding": return `Query planner · ${plain(event.status)} · ${plain(event.reason)} · original question retained`;
    case "rewrite_guard": return `Rewrite repaired · ${event.reasons.map(plain).join(", ")}`;
    case "coverage_guard": return `Coverage held back · ${plain(event.reason)} · ${event.needs.map(plain).join(", ")}`;
    case "coverage": return `Coverage · ${event.covered_needs.map(plain).join(", ") || "none"} supported · missing ${event.missing_needs.map(plain).join(", ") || "none"}`;
    case "escalation_decision": return `Routing · ${plain(event.reason)} · ${plain(event.action)}`;
    case "definition_promotion": return `Definition release · ${plain(event.reason)}${pages}`;
    case "local_issue": return `Local issue · ${plain(event.code)}`;
    case "first_verified": return `First evidence verified${pages}`;
    case "route": {
      const explanation = reasons[event.reason] || plain(event.reason);
      if (event.would_call_reason) {
        const why = reasons[event.would_call_reason] || plain(event.would_call_reason);
        return `Would call Groq · ${why} ${explanation}`;
      }
      const decision = event.decision === "call" ? "Call Groq" : "Skip Groq";
      return `${decision} · ${explanation}`;
    }
    case "frontier_start": return `Groq request · ${event.model}`;
    case "frontier_end": {
      const tokens = Number.isInteger(event.prompt_tokens) && Number.isInteger(event.completion_tokens)
        ? ` · ${event.prompt_tokens} input / ${event.completion_tokens} output tokens` : "";
      const cost = Number.isFinite(event.list_cost_usd)
        ? ` · $${event.list_cost_usd.toFixed(6)} estimated` : "";
      return `Groq ${plain(event.status)}${event.code ? ` (${plain(event.code)})` : ""}${duration}${tokens}${cost}`;
    }
    case "frontier_verify": return event.passed ? "Groq answer verified" : "Groq answer withheld";
    case "finish": return "Run complete";
    default: return null;
  }
}

function drawAgents(run) {
  const container = byId("agents");
  container.replaceChildren();
  byId("activity-state").textContent = !run ? "Ready"
    : run.running ? "Live" : "Recorded run";
  for (const [role, label] of Object.entries(roleNames)) {
    const events = (run?.events || []).filter((event) => event.kind === "activity" && event.role === role);
    const last = events.at(-1);
    let state = last?.status || (run && !run.running ? "Not used" : "Waiting");
    if (state === "running" && !run?.running) state = "Interrupted";
    const card = element("div", "", `agent ${state}`);
    const heading = element("div", "", "agent-head");
    heading.append(element("strong", label), element("span", plain(state), `badge ${state}`));
    card.append(heading);
    card.append(element("p", last?.action || (
      run?.case ? "Scripted synthetic fixture"
        : run && !run.running ? "This role was not needed for this run."
        : "Waiting for this role's turn."
    )));
    card.append(element("p", last?.reason || (
      run?.case ? "This demo records routing events without executing each model role."
        : run && !run.running ? "The route completed without invoking this role."
        : "The controller schedules this role when its input is ready."
    ), "why"));
    if (last?.model) card.append(element("p", `Model · ${last.model}`, "why"));
    if (last?.pages?.length) card.append(element("p", `Pages ${last.pages.join(", ")}`, "why"));
    if (last) {
      let timing = `Updated at ${milliseconds(last.elapsed_ms)}`;
      if (state === "running" && run?.startedAt) {
        timing = `Working for ${milliseconds(Math.max(0, performance.now() - run.startedAt - last.elapsed_ms))}`;
      }
      card.append(element("p", timing, "elapsed"));
    }
    container.append(card);
  }
}

function showRun(run) {
  selected = run.trace_id;
  drawAgents(run);
  const panel = byId("result");
  panel.classList.remove("empty");
  panel.replaceChildren(element("p", run.question, "answer"));
  if (run.model_roles) {
    panel.append(element("p", `Answer · ${run.model_roles.answer.model} | Verifier · ${run.model_roles.verifier.model}`, "summary"));
  }
  if (run.verification_provider) {
    panel.append(element("p", run.verification_provider === "local"
      ? "Verification: local only · no Groq check"
      : "Verification: Groq", "summary"));
  }
  if (run.verification_notice) panel.append(element("p", run.verification_notice, "proof"));
  setBadge(byId("route-badge"), run);
  const total = eventOf(run, "finish");
  const lane = run.case ? "Synthetic demo" : run.document?.name || "Confidential document";
  panel.append(element("p", `${lane} · ${total ? milliseconds(total.elapsed_ms) + " total" : "Run in progress"}`, "summary"));
  const verification = {
    exact_source_text_model_assessed_relevance: "Exact source text checked · relevance and completeness assessed by the configured verifier",
    exact_source_text_bounded_task_match: "Exact source text checked · bounded section-location or explicit-duration check passed",
    exact_source_definition: "Verification · explicit definition in source",
    exact_excerpt_only: "Verification · source provenance only; relevance and completeness require review",
    bounded_arithmetic: "Verification · arithmetic on user-provided values",
  };
  if (run.verification) panel.append(element("p", verification[run.verification] || plain(run.verification), "proof"));
  if (run.message) panel.append(element("p", run.message, "summary"));
  if (run.review_excerpts?.length) {
    const pending = element("details");
    pending.append(element("summary", "Inspect selected source text · not approved as an answer"));
    for (const excerpt of run.review_excerpts) {
      pending.append(element("blockquote", excerpt.text, "source-text"));
      const link = element("a", `PDF page ${excerpt.page} · ${excerpt.heading || "Source passage"}`);
      link.href = sourceURL(run, excerpt.page);
      link.target = "_blank";
      link.rel = "noopener";
      pending.append(link);
    }
    panel.append(pending);
  }
  if (run.draft_answer?.length) {
    panel.append(element("h3", "Provisional answer · review required"));
    for (const claim of run.draft_answer) {
      panel.append(element("p", claim.text, "answer"));
      for (const id of claim.citations) {
        const passage = run.evidence?.find((p) => p.passage_id === id);
        if (!passage) continue;
        const link = element("a", `PDF page ${passage.page} · ${passage.heading}`);
        link.href = sourceURL(run, passage.page);
        link.target = "_blank";
        link.rel = "noopener";
        panel.append(link);
      }
    }
  }
  if (run.working_state) {
    panel.append(element("h3", "Request requirements"));
    const needs = element("ol");
    for (const need of run.working_state.requirements) needs.append(element("li", need));
    panel.append(needs);
    for (const gap of run.working_state.gaps) panel.append(element("p", `Unresolved · ${gap}`, "summary"));
  }
  if (run.checks?.length) {
    const details = element("details");
    details.append(element("summary", "Checks and recovery history"));
    for (const check of run.checks) {
      details.append(element("p", `${plain(check.basis)} · ${JSON.stringify(check.checks || {})} · ${(check.issues || []).join("; ") || "no issues reported"}`, "summary"));
    }
    panel.append(details);
  }
  if (run.record) {
    const download = element("button", "Download local review packet", "secondary");
    download.type = "button";
    download.addEventListener("click", () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(run, null, 2)], {type: "application/json"}));
      const link = element("a");
      link.href = url;
      link.download = `review-${run.trace_id}.json`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    panel.append(download);
    panel.append(element("p", `Saved local record · SHA-256 ${run.record.sha256}`, "summary"));
  }
  panel.append(element("p", `Run build · ${run.build?.version || "not recorded"}`, "summary"));
  if (run.selection_method) panel.append(element("p", `Selection · ${plain(run.selection_method)}`, "summary"));
  if (run.query_plan) {
    const plan = run.query_plan;
    panel.append(element("h3", "Search interpretation"));
    panel.append(element("p", `${plan.model} · ${plain(plan.intent)} · ${plain(plan.reason)}`, "summary"));
    if (plan.term) panel.append(element("p", `Interpreted term · ${plan.term}`, "summary"));
    if (plan.subject) panel.append(element("p", `Subject · ${plan.subject}`, "summary"));
    if (plan.normalized_query) panel.append(element("p", `Interpreted request · ${plan.normalized_query}`, "summary"));
    const queries = element("ul");
    for (const query of plan.queries) queries.append(element("li", query));
    panel.append(queries, element("p", "The original question is retained in retrieval.", "summary"));
  }
  if (run.suggestions?.length) {
    for (const suggestion of run.suggestions) {
      const choice = element("button", suggestion, "secondary");
      choice.type = "button";
      choice.addEventListener("click", () => {
        byId("question").value = suggestion;
        request("/api/search", {question: suggestion, topic: null}, "Checking the selected interpretation…");
      });
      panel.append(choice);
    }
  }
  if (run.coverage) {
    const coverage = run.coverage;
    panel.append(element("h3", "Evidence coverage"));
    panel.append(element("p", `Model-assessed · ${plain(coverage.status)}. Exact quotations are checked against the source.`, "summary"));
    const list = element("ul");
    for (const need of coverage.required) {
      list.append(element("li", `${plain(need)} · ${coverage.covered.includes(need) ? "source quotation found" : "not established"}`));
    }
    panel.append(list);
  }
  if (run.routing) {
    panel.append(element("p", `Routing · ${plain(run.routing.reason)}. ${run.routing.hosted_candidate ? "A hosted reasoning task may help; report policy blocks transmission to Groq." : "Continue locally or review the stated limitation."}`, "summary"));
  }
  if (run.calculation) {
    panel.append(element("h3", "Calculation"));
    panel.append(element("p", `${run.calculation.expression} = ${run.calculation.value}`, "answer"));
  }
  if (run.claims?.length) {
    panel.append(element("h3", run.route === "source_excerpt" ? "Selected source text" : "Evidence"));
    for (const claim of run.claims) {
      if (claim.need) panel.append(element("h3", plain(claim.need)));
      panel.append(element(run.route === "source_excerpt" ? "blockquote" : "p", claim.text, run.route === "source_excerpt" ? "source-text" : "answer"));
      const source = element("details");
      source.append(element("summary", `PDF page ${claim.page}`), element("blockquote", claim.quote));
      const openSource = element("a", "Open source page");
      openSource.href = sourceURL(run, claim.page);
      openSource.target = "_blank";
      openSource.rel = "noopener";
      source.append(openSource);
      if (run.route === "source_excerpt") {
        openSource.textContent = `PDF page ${claim.page} · ${claim.heading || "Source passage"}`;
        panel.append(openSource);
      } else panel.append(source);
    }
  } else if (!run.case && !run.running && !run.message) {
    panel.append(element("p", "The local run could not verify an answer.", "summary"));
  }
  if (run.evidence?.length) {
    const evidence = element("details");
    evidence.append(element("summary", "Inspect retrieved passages"));
    for (const passage of run.evidence) {
      evidence.append(element("h3", `Page ${passage.page} · ${passage.heading || passage.kind}`));
      evidence.append(element("blockquote", passage.text));
    }
    panel.append(evidence);
  }
  if (run.reason) panel.append(element("p", `Stop reason · ${plain(run.reason)}`, "summary"));
  panel.append(element("h3", "Decision timeline"));
  const timeline = element("ol", "", "timeline");
  for (const event of run.events) {
    const description = describe(event);
    if (!description) continue;
    const item = element("li");
    item.append(element("time", milliseconds(event.elapsed_ms)), element("span", description));
    timeline.append(item);
  }
  panel.append(timeline);
  drawHistory();
}

function drawHistory() {
  byId("empty-history").hidden = runs.length > 0;
  byId("total").textContent = runs.length;
  byId("local").textContent = runs.filter((run) => ["source_excerpt", "answer_local", "calculation_local"].includes(run.route) && !run.review_packet?.hosted_calls).length;
  byId("frontier").textContent = runs.filter((run) => eventOf(run, "frontier_start") || run.review_packet?.hosted_calls > 0).length;
  byId("open").textContent = runs.filter((run) => ["beta_flag", "human_review", "partial", "insufficient_evidence", "evidence_only", "ask_clarification", "local_error"].includes(run.route)).length;
  const list = byId("runs");
  list.replaceChildren();
  for (const run of runs) {
    const item = element("li");
    const button = element("button", "", "run");
    button.type = "button";
    button.setAttribute("aria-pressed", String(run.trace_id === selected));
    button.append(element("span", run.question, "run-title"));
    const badge = element("span");
    setBadge(badge, run);
    button.append(badge, element("span", run.case ? "Demo" : run.document?.name || "Document", "run-meta"));
    button.addEventListener("click", () => {
      showRun(run);
      list.children[runs.indexOf(run)]?.firstElementChild.focus();
    });
    item.append(button);
    list.append(item);
  }
}

async function request(path, payload, message) {
  if (path === "/api/search") payload = {...payload, local_only: byId("local-only").checked};
  if (path === "/api/search" && !byId("model-role-controls").hidden) {
    payload = {...payload, role_assignment: byId("model-roles").value};
  }
  const buttons = ["search-button", "demo", "live", "upload-button", "document-select", "pdf-file", "model-roles", "local-only"].map(byId);
  requestInProgress = true;
  const status = byId("status");
  status.hidden = false;
  status.classList.remove("error");
  status.textContent = message;
  buttons.forEach((button) => { button.disabled = true; });
  let pending = null;
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || "Run failed.");
    }
    if (path === "/api/search") {
      pending = {
        question: payload.question, route: "running", events: [],
        trace_id: "pending", running: true, startedAt: performance.now(), document: currentDocument,
        local_only: payload.local_only,
      };
      activeRun = pending;
      runs.unshift(pending);
      showRun(pending);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;
      function consume(line) {
        if (!line.trim()) return;
        const item = JSON.parse(line);
        if (item.type === "error") throw new Error(item.error);
        if (item.type === "event") {
          pending.trace_id = item.event.trace_id;
          pending.events.push(item.event);
        } else if (item.type === "result") {
          Object.assign(pending, item.result, {running: false});
          conversationTopic = item.result.topic || null;
          finished = true;
        }
        showRun(pending);
      }
      while (true) {
        const {value, done} = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) consume(line);
        if (done) break;
      }
      if (buffer.trim()) consume(buffer);
      if (!finished) throw new Error("The local connection ended before the run finished.");
      status.textContent = "Search complete.";
    } else {
      const data = await response.json();
      runs.unshift(...data);
      showRun(data.find((run) => ["would_call", "answer_frontier"].includes(run.route)) || data[0]);
      status.textContent = "Demo routes added to history.";
    }
  } catch (error) {
    if (pending) {
      pending.running = false;
      pending.route = "local_error";
      showRun(pending);
    }
    status.textContent = error.message || "Could not connect to the local server.";
    status.classList.add("error");
  } finally {
    activeRun = null;
    requestInProgress = false;
    buttons.forEach((button) => { button.disabled = false; });
    byId("search-button").disabled = !currentDocument;
  }
}

setInterval(() => {
  if (activeRun && selected === activeRun.trace_id) drawAgents(activeRun);
}, 1000);
drawAgents(null);

byId("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = byId("question").value.trim();
  const context = byId("context").value.trim() || null;
  if (question && currentDocument) request("/api/search", {question, topic: conversationTopic, context, document_id: currentDocument.id}, "Searching and checking source evidence…");
});
byId("document-select").addEventListener("change", (event) => chooseDocument(event.target.value));
byId("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = byId("pdf-file").files[0];
  if (!file || requestInProgress) return;
  if (file.size > maxUploadBytes) {
    byId("document-status").textContent = "PDF uploads must be 25 MB or smaller.";
    return;
  }
  const controls = ["search-button", "upload-button", "document-select", "pdf-file", "demo", "live"].map(byId);
  requestInProgress = true;
  controls.forEach((control) => { control.disabled = true; });
  byId("document-status").textContent = "Uploading and indexing the PDF locally…";
  try {
    const response = await fetch(`/api/documents?name=${encodeURIComponent(file.name)}`, {
      method: "POST", headers: {"Content-Type": "application/pdf"}, body: file,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The PDF could not be uploaded.");
    documentLibrary = documentLibrary.filter((doc) => doc.id !== data.document.id);
    documentLibrary.push(data.document);
    drawDocuments(data.document.id);
    byId("document-status").textContent += ` ${data.duplicate ? "Existing PDF selected." : "Ready."} Indexing took ${milliseconds(data.index_ms)}.`;
    byId("pdf-file").value = "";
  } catch (error) {
    byId("document-status").textContent = error.message;
  } finally {
    requestInProgress = false;
    controls.forEach((control) => { control.disabled = false; });
    byId("search-button").disabled = !currentDocument;
  }
});
byId("demo").addEventListener("click", () => {
  request("/api/demo", {live: false}, "Running the offline synthetic demo…");
});
byId("live").addEventListener("click", () => {
  request("/api/demo", {live: true}, "Running the live synthetic demo with Groq…");
});
function updateVerificationMode() {
  const config = verificationConfig;
  if (!config) return;
  const localOnly = byId("local-only").checked;
  const local = localOnly || !config.hosted_verification;
  byId("privacy-note").textContent = local
    ? "Local verification only. Your question, context and excerpts stay on this device. No Groq check."
    : "Selected excerpts, your question and supplied context are sent to Groq for verification. PDFs remain stored locally.";
  byId("local-verification-warning").textContent = config.local_verification_warning || "Local verification may be less reliable. Review the source excerpts.";
  byId("local-verification-warning").hidden = !local || config.contract !== "source-selection-v1";
  byId("live").hidden = !config.live_enabled || localOnly;
  if (config.hosted_verification && config.local_verification) {
    const roles = localOnly ? config.local_verification : config.model_roles.primary;
    const option = byId("model-roles").querySelector('option[value="primary"]');
    if (option) option.textContent = `Select: ${roles.answer} · Verify: ${roles.verifier}`;
  }
}
byId("local-only").addEventListener("change", () => {
  try { sessionStorage.setItem("evidence-guard-local-only", String(byId("local-only").checked)); } catch {}
  updateVerificationMode();
});
async function refreshConfig() {
  const config = await fetch("/api/config").then((response) => response.json());
  verificationConfig = config;
  byId("confidential-controls").hidden = !config.hosted_verification || !config.local_verification;
  byId("build-info").textContent = `v${config.version} · ${config.contract}`;
  if (config.contract === "source-selection-v1") {
    roleNames.planner = "Query preparation";
    roleNames.writer = "Source selector";
  }
  if (config.model_roles && !byId("model-roles").options.length) {
    for (const key of ["primary", "reversed"]) {
      const roles = config.model_roles[key];
      if (!roles) continue;
      const option = element("option", `Select: ${roles.answer} · Verify: ${roles.verifier}`);
      option.value = key;
      byId("model-roles").append(option);
    }
    byId("model-roles").value = config.model_roles.default;
    byId("model-role-controls").hidden = false;
  }
  updateVerificationMode();
  maxUploadBytes = config.max_upload_bytes || maxUploadBytes;
  if (!requestInProgress && Array.isArray(config.documents)) {
    documentLibrary = config.documents;
    const selectedId = currentDocument?.id || config.default_document_id || documentLibrary[0]?.id;
    drawDocuments(selectedId);
  }
  if (config.evaluation) {
    const scope = config.evaluation_info?.current_source ? "Current source evaluation" : `Historical evaluation (v${config.evaluation_info?.version || "unrecorded"}; different source)`;
    byId("evaluation-summary").textContent = `${scope}: ${config.evaluation.passed}/${config.evaluation.total} automated checks passed. Median ${milliseconds(config.evaluation.median_ms)} · p95 ${milliseconds(config.evaluation.p95_ms)}. Cases with model failures: ${config.evaluation.model_failure_cases ?? "unrecorded"}; planner fallbacks: ${config.evaluation.planner_fallback_cases ?? "unrecorded"}. Relevance requires separate review.`;
    byId("evaluation-summary").hidden = false;
  }
  }
refreshConfig().catch(() => {});
window.addEventListener("focus", () => refreshConfig().catch(() => {}));
