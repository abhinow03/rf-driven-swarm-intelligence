// Defense demo web interface. Talks to scripts/demo_web.py's Flask API.
// Act 1/2 data is fetched once (pre-computed, no GPU). Act 3 hits the live
// backend on submit. All model-generated text is inserted via textContent,
// never innerHTML, so raw LLM output can never be interpreted as markup.

const $ = (sel, root = document) => root.querySelector(sel);
const $all = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, opts = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(opts)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) node.appendChild(child);
  return node;
}

// --- tab switching ---
$all(".act-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    $all(".act-tab").forEach(t => t.classList.remove("active"));
    $all(".act-panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    $("#" + tab.dataset.act).classList.add("active");
  });
});

// --- status bar ---
async function loadStatus() {
  const res = await fetch("/api/status");
  const s = await res.json();
  const bar = $("#status-bar");
  bar.innerHTML = "";

  const adapterName = s.active_adapter.split("/").filter(Boolean).pop();
  bar.appendChild(el("span", { text: `ACTIVE_ADAPTER: ${adapterName}` }));

  const ckBadge = el("span", {
    class: `badge ${s.checkpoint_ok ? "ok" : "fail"}`,
    text: s.checkpoint_ok ? "checkpoint verified" : "checkpoint FAILED",
  });
  bar.appendChild(ckBadge);

  const stale = s.webdata_active_adapter !== s.active_adapter;
  bar.appendChild(el("span", {
    class: `badge ${stale ? "stale" : "ok"}`,
    text: stale ? "Act 1/2 cache STALE -- rebuild it" : "Act 1/2 cache fresh",
  }));

  if (s.webdata_generated_at_utc) {
    bar.appendChild(el("span", { text: `cached: ${s.webdata_generated_at_utc}` }));
  }

  return s;
}

// --- Act 1 ---
function renderAssessmentTable(assessment) {
  const table = el("table", { class: "assessment-table" });
  const fields = ["threat_level", "likely_intent", "recommended_action", "confidence_in_assessment"];
  for (const f of fields) {
    const tr = el("tr", {}, [
      el("td", { text: f }),
      el("td", { text: assessment ? String(assessment[f] ?? "—") : "—" }),
    ]);
    table.appendChild(tr);
  }
  return table;
}

let act1Data = null;

async function loadAct1() {
  const res = await fetch("/api/act1");
  act1Data = await res.json();

  const scenarioCard = $("#act1-scenario");
  scenarioCard.innerHTML = "";
  scenarioCard.appendChild(el("div", {
    class: "scenario-title",
    text: `Scenario ${act1Data.case_name} (${act1Data.pair[0]} -> ${act1Data.pair[1]}, expected_threat=${act1Data.expected_threat})`,
  }));
  scenarioCard.appendChild(el("div", { class: "ctx-block", text: act1Data.ctx }));

  const btnRow = $("#act1-buttons");
  btnRow.innerHTML = "";
  act1Data.systems.forEach(sys => {
    const btn = el("button", { text: sys.label, "data-key": sys.key });
    btn.addEventListener("click", () => spotlightSystem(sys.key));
    btnRow.appendChild(btn);
  });

  renderSystems();
}

function renderSystems(spotlightKey = null) {
  const row = $("#act1-systems");
  row.innerHTML = "";
  act1Data.systems.forEach(sys => {
    const card = el("div", { class: "system-card" });
    if (spotlightKey) {
      card.classList.add(sys.key === spotlightKey ? "spotlight" : "dimmed");
    }
    card.appendChild(el("h3", { text: sys.label }));

    const verdictClass = sys.error ? "verdict-error"
      : sys.verdict === "correct" ? "verdict-correct"
      : sys.verdict === "incorrect" ? "verdict-incorrect" : "verdict-abstained";
    card.appendChild(el("span", { class: `verdict-badge ${verdictClass}`, text: sys.verdict }));

    card.appendChild(el("div", { class: "fact-line", text: sys.fact }));

    if (sys.error) {
      card.appendChild(el("div", { class: "ctx-block", text: `GENERATION FAILED: ${sys.error}` }));
    } else {
      card.appendChild(renderAssessmentTable(sys.parsed));
      const raw = el("details", {});
      raw.appendChild(el("summary", { text: "raw output" }));
      raw.appendChild(el("div", { class: "ctx-block", text: sys.raw || "" }));
      card.appendChild(raw);
    }

    row.appendChild(card);
  });
}

function spotlightSystem(key) {
  $all("#act1-buttons button").forEach(b => b.classList.toggle("selected", b.dataset.key === key));
  renderSystems(key);
}

// --- Act 2 ---
const LAYER_CLASS = {
  layer1_deterministic: "layer-1",
  layer2_guard: "layer-2",
  layer3_llm: "layer-3",
};
const LAYER_TEXT = {
  layer1_deterministic: "LAYER 1 -- deterministic rule-table lookup",
  layer2_guard: "LAYER 2 -- guard abstention (no model call)",
  layer3_llm: "LAYER 3 -- LLM judgment (no RULES entry exists)",
};

function renderRoutingDetail(c) {
  if (c.layer === "layer1_deterministic") {
    let text = `rule-table key: ${JSON.stringify(c.detail.rules_key)}`;
    if (c.detail.llm_deviation && Object.keys(c.detail.llm_deviation).length) {
      text += ` (narrator tried to deviate on ${Object.keys(c.detail.llm_deviation).join(", ")} -- overwritten)`;
    }
    return text;
  }
  if (c.layer === "layer2_guard") {
    return `guard reason(s): ${c.detail.guard_reasons.join("; ")}`;
  }
  if (c.layer === "layer3_llm") {
    let text = `subtype: ${c.detail.subtype} -- ACTIVE_ADAPTER's judgment is load-bearing here`;
    if (c.detail.correction && c.detail.correction.applied) {
      text += ` (prior-corrected threat_level: ${c.detail.correction.corrected_argmax})`;
    }
    return text;
  }
  return "";
}

async function loadAct2() {
  const res = await fetch("/api/act2");
  const cases = await res.json();

  const col = $("#act2-cases");
  col.innerHTML = "";
  cases.forEach(c => {
    const card = el("div", { class: "case-card" });
    card.appendChild(el("span", { class: `layer-badge ${LAYER_CLASS[c.layer] || ""}`, text: LAYER_TEXT[c.layer] || c.layer || "FAILED" }));
    card.appendChild(el("div", { class: "case-title", text: `Case ${c.case_name}` }));
    card.appendChild(el("div", {
      class: "case-meta",
      text: `pair=${JSON.stringify(c.pair)}  has_ground_truth=${c.has_ground_truth}`,
    }));
    card.appendChild(el("div", { class: "ctx-block", text: c.ctx }));

    if (c.error) {
      card.appendChild(el("div", { class: "routing-detail", text: `PIPELINE FAILED: ${c.error}` }));
    } else {
      card.appendChild(el("div", { class: "routing-detail", text: renderRoutingDetail(c) }));
      card.appendChild(renderAssessmentTable(c.assessment));
    }
    col.appendChild(card);
  });
}

// --- Act 3 ---
async function loadAct3Disclosure() {
  const res = await fetch("/api/act3_disclosure");
  const data = await res.json();
  $("#act3-disclosure").textContent = data.text;
}

function populateFormationSelects(formations) {
  const a = $("#form-a"), b = $("#form-b");
  formations.forEach(f => {
    a.appendChild(el("option", { value: f, text: f }));
    b.appendChild(el("option", { value: f, text: f }));
  });
  if (formations.length > 1) b.selectedIndex = 1;
}

async function runAct3(formA, formB) {
  const resultDiv = $("#act3-result");
  resultDiv.innerHTML = "";
  resultDiv.appendChild(el("div", { class: "spinner", text: "running live generation…" }));

  const res = await fetch("/api/act3", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ form_a: formA, form_b: formB }),
  });
  const data = await res.json();

  resultDiv.innerHTML = "";
  const card = el("div", { class: "act3-result-card" });

  if (data.error) {
    card.classList.add("error");
    card.appendChild(el("div", { class: "case-title", text: `${formA} -> ${formB}` }));
    card.appendChild(el("div", { text: data.error }));
    if (data.ctx) card.appendChild(el("div", { class: "ctx-block", text: data.ctx }));
    resultDiv.appendChild(card);
    return;
  }

  card.appendChild(el("div", { class: "case-title", text: `${formA} -> ${formB}` }));
  card.appendChild(el("div", { class: "ctx-block", text: data.ctx }));
  card.appendChild(el("span", { class: `layer-badge ${LAYER_CLASS[data.layer] || ""}`, text: LAYER_TEXT[data.layer] || data.layer }));
  card.appendChild(el("div", { class: "routing-detail", text: renderRoutingDetail(data) }));
  card.appendChild(renderAssessmentTable(data.assessment));
  resultDiv.appendChild(card);
}

$("#act3-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const formA = $("#form-a").value, formB = $("#form-b").value;
  runAct3(formA, formB);
});

// --- boot ---
(async function init() {
  const status = await loadStatus();
  populateFormationSelects(status.formations);
  await Promise.all([loadAct1(), loadAct2(), loadAct3Disclosure()]);
})();
