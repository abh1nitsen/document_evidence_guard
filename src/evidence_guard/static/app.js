"use strict";

const byId = (id) => document.getElementById(id);
const runs = [];
let selected = null;
let activeRun = null;
let conversationTopic = null;
const roleNames = {
  planner: "Planner", retriever: "Retriever", context: "Context",
  writer: "Writer", verifier: "Verifier", router: "Router",
};
const routeLabels = {
  answer_local: ["Answered locally", "local"],
  answer_frontier: ["Answered via Groq", "frontier"],
  would_call: ["Would call Groq", "frontier"],
  partial: ["Partial answer", "open"],
  insufficient_evidence: ["Insufficient evidence", "open"],
  provider_error: ["Groq failed", "open"],
  frontier_unverified: ["Groq answer withheld", "open"],
  local_error: ["Local run failed", "open"],
  running: ["Running locally", "local"],
  evidence_only: ["Source evidence · review needed", "open"],
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
    case "fetch": return `Fetch ${event.attempt} · ${plain(event.strategy || "synthetic")}${pages}${duration}`;
    case "local_model": return `Local model · ${event.model || "selector"} · ${plain(event.stage)}${duration}`;
    case "query_understanding": return `Query planner · ${plain(event.status)} · ${plain(event.reason)} · original question retained`;
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
  setBadge(byId("route-badge"), run);
  const total = eventOf(run, "finish");
  const lane = run.case ? "Synthetic demo" : "Confidential report";
  panel.append(element("p", `${lane} · ${total ? milliseconds(total.elapsed_ms) + " total" : "Run in progress"}`, "summary"));
  const verification = {
    exact_source_definition: "Verification · explicit definition in source",
    exact_excerpt_only: "Verification · source provenance only; relevance and completeness require review",
    bounded_arithmetic: "Verification · arithmetic on user-provided values",
  };
  if (run.verification) panel.append(element("p", verification[run.verification] || plain(run.verification), "proof"));
  if (run.message) panel.append(element("p", run.message, "summary"));
  if (run.selection_method) panel.append(element("p", `Selection · ${plain(run.selection_method)}`, "summary"));
  if (run.query_plan) {
    const plan = run.query_plan;
    panel.append(element("h3", "Search interpretation"));
    panel.append(element("p", `${plan.model} · ${plain(plan.intent)} · ${plain(plan.reason)}`, "summary"));
    if (plan.term) panel.append(element("p", `Interpreted term · ${plan.term}`, "summary"));
    const queries = element("ul");
    for (const query of plan.queries) queries.append(element("li", query));
    panel.append(queries, element("p", "The original question is retained in retrieval.", "summary"));
  }
  if (run.calculation) {
    panel.append(element("h3", "Calculation"));
    panel.append(element("p", `${run.calculation.expression} = ${run.calculation.value}`, "answer"));
  }
  if (run.claims?.length) {
    panel.append(element("h3", "Evidence"));
    for (const claim of run.claims) {
      panel.append(element("p", claim.text, "answer"));
      const source = element("details");
      source.append(element("summary", `PDF page ${claim.page}`), element("blockquote", claim.quote));
      const openSource = element("a", "Open source page");
      openSource.href = `/api/document#page=${claim.page}`;
      openSource.target = "_blank";
      openSource.rel = "noopener";
      source.append(openSource);
      panel.append(source);
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
  byId("local").textContent = runs.filter((run) => ["answer_local", "calculation_local"].includes(run.route)).length;
  byId("frontier").textContent = runs.filter((run) => eventOf(run, "frontier_start")).length;
  byId("open").textContent = runs.filter((run) => ["partial", "insufficient_evidence", "evidence_only", "ask_clarification", "local_error"].includes(run.route)).length;
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
    button.append(badge, element("span", run.case ? "Demo" : "Report", "run-meta"));
    button.addEventListener("click", () => {
      showRun(run);
      list.children[runs.indexOf(run)]?.firstElementChild.focus();
    });
    item.append(button);
    list.append(item);
  }
}

async function request(path, payload, message) {
  const buttons = ["search-button", "demo", "live"].map(byId);
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
        trace_id: "pending", running: true, startedAt: performance.now(),
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
          if (item.result.topic) conversationTopic = item.result.topic;
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
    buttons.forEach((button) => { button.disabled = false; });
  }
}

setInterval(() => {
  if (activeRun && selected === activeRun.trace_id) drawAgents(activeRun);
}, 1000);
drawAgents(null);

byId("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = byId("question").value.trim();
  if (question) request("/api/search", {question, topic: conversationTopic}, "Searching and checking source evidence locally…");
});
byId("demo").addEventListener("click", () => {
  request("/api/demo", {live: false}, "Running the offline synthetic demo…");
});
byId("live").addEventListener("click", () => {
  request("/api/demo", {live: true}, "Running the live synthetic demo with Groq…");
});
fetch("/api/config").then((response) => response.json()).then((config) => {
  byId("live").hidden = !config.live_enabled;
  byId("build-info").textContent = `v${config.version} · ${config.contract}`;
  if (config.evaluation) {
    byId("evaluation-summary").textContent = `Latest local evaluation: ${config.evaluation.passed}/${config.evaluation.total} automated checks passed. Median ${milliseconds(config.evaluation.median_ms)} · p95 ${milliseconds(config.evaluation.p95_ms)}. Cases with model failures: ${config.evaluation.model_failure_cases ?? "unrecorded"}. Relevance requires separate review.`;
    byId("evaluation-summary").hidden = false;
  }
}).catch(() => {});
