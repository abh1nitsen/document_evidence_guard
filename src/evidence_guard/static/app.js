"use strict";

const byId = (id) => document.getElementById(id);
const runs = [];
let selected = null;
const routeLabels = {
  answer_local: ["Answered locally", "local"],
  answer_frontier: ["Answered via Groq", "frontier"],
  would_call: ["Would call Groq", "frontier"],
  partial: ["Partial answer", "open"],
  insufficient_evidence: ["Insufficient evidence", "open"],
  provider_error: ["Groq failed", "open"],
  frontier_unverified: ["Groq answer withheld", "open"],
};
const reasons = {
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
    case "index_load": return `Report index ${event.cache_hit ? "reused" : "built"}${duration}`;
    case "fetch": return `Fetch ${event.attempt}${pages}${duration}`;
    case "local_model": return `Local model · ${plain(event.stage)}${duration}`;
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

function showRun(run) {
  selected = run.trace_id;
  const panel = byId("result");
  panel.classList.remove("empty");
  panel.replaceChildren(element("p", run.question, "answer"));
  setBadge(byId("route-badge"), run);
  const total = eventOf(run, "finish");
  const lane = run.case ? "Synthetic demo" : "Confidential report";
  panel.append(element("p", `${lane} · ${milliseconds(total.elapsed_ms)} total`, "summary"));
  if (run.claims?.length) {
    panel.append(element("h3", "Evidence"));
    for (const claim of run.claims) {
      panel.append(element("p", claim.text, "answer"));
      const source = element("details");
      source.append(element("summary", `PDF page ${claim.page}`), element("blockquote", claim.quote));
      panel.append(source);
    }
  } else if (!run.case) {
    panel.append(element("p", "The local run could not verify an answer.", "summary"));
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
  byId("local").textContent = runs.filter((run) => run.route === "answer_local").length;
  byId("frontier").textContent = runs.filter((run) => eventOf(run, "frontier_start")).length;
  byId("open").textContent = runs.filter((run) => ["partial", "insufficient_evidence"].includes(run.route)).length;
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
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Run failed.");
    const newRuns = Array.isArray(data) ? data : [{...data, question: payload.question}];
    runs.unshift(...newRuns);
    showRun(newRuns.find((run) => ["would_call", "answer_frontier"].includes(run.route)) || newRuns[0]);
    status.textContent = Array.isArray(data) ? "Demo routes added to history." : "Search complete.";
  } catch (error) {
    status.textContent = error.message || "Could not connect to the local server.";
    status.classList.add("error");
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
  }
}

byId("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = byId("question").value.trim();
  if (question) request("/api/search", {question}, "Searching and verifying locally…");
});
byId("demo").addEventListener("click", () => {
  request("/api/demo", {live: false}, "Running the offline synthetic demo…");
});
byId("live").addEventListener("click", () => {
  request("/api/demo", {live: true}, "Running the live synthetic demo with Groq…");
});
fetch("/api/config").then((response) => response.json()).then((config) => {
  byId("live").hidden = !config.live_enabled;
}).catch(() => {});
