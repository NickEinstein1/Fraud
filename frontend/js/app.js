import {
  API_BASE,
  getArchitecture,
  getCsvRow,
  getSystemInfo,
  openRealtimeSocket,
  scoreFromPath,
  scoreFromUpload,
  scoreTransactions,
} from "./api.js";

const RING_C = 327;
let systemInfo = null;
let lastBatchCsv = null;
let streamSocket = null;
let probHistory = [];

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const PAGE_COPY = {
  home: { title: "Overview", subtitle: "Platform health & AI architecture" },
  score: { title: "Score transaction", subtitle: "Single-transaction fraud analysis" },
  batch: { title: "Batch analysis", subtitle: "File path or CSV upload" },
  live: { title: "Live stream", subtitle: "Real-time AI + drift monitoring" },
  system: { title: "System", subtitle: "Model registry & feature schema" },
};

function toast(message, type = "info") {
  const stack = $("#toast-stack");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

function navigate(view) {
  $$(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === view));
  $$(".nav-link").forEach((l) => l.classList.toggle("active", l.dataset.nav === view));
  const copy = PAGE_COPY[view];
  if (copy) {
    $("#page-title").textContent = copy.title;
    $("#page-subtitle").textContent = copy.subtitle;
  }
  location.hash = view;
}

function haversineKm(lat1, lon1, lat2, lon2) {
  const r = 6371;
  const toRad = (d) => (d * Math.PI) / 180;
  const a1 = toRad(lat1);
  const a2 = toRad(lat2);
  const dlat = toRad(lat2 - lat1);
  const dlon = toRad(lon2 - lon1);
  const a =
    Math.sin(dlat / 2) ** 2 +
    Math.cos(a1) * Math.cos(a2) * Math.sin(dlon / 2) ** 2;
  return 2 * r * Math.asin(Math.min(1, Math.sqrt(a)));
}

function updateDistancePreview(form) {
  const el = $("#distance-preview");
  if (!el || !form) return;
  const lat = parseFloat(form.elements.lat?.value);
  const lon = parseFloat(form.elements.long?.value);
  const mlat = parseFloat(form.elements.merch_lat?.value);
  const mlon = parseFloat(form.elements.merch_long?.value);
  if ([lat, lon, mlat, mlon].some((v) => Number.isNaN(v))) {
    el.textContent = "";
    return;
  }
  const km = haversineKm(lat, lon, mlat, mlon);
  el.textContent = `Computed distance_km ≈ ${km.toFixed(2)} km (Haversine; sent with your request)`;
}

function buildManualForm(schema, defaults) {
  const form = $("#form-manual");
  form.innerHTML = "";

  const fields = schema?.length
    ? schema.filter((f) => !f.hidden)
    : (defaults ? Object.keys(defaults) : []).map((name) => ({
        name,
        type: "number",
        label: name,
      }));

  const core = fields.filter((f) => f.group !== "location");
  const location = fields.filter((f) => f.group === "location");

  const coreGrid = document.createElement("div");
  coreGrid.className = "form-grid";
  core.forEach((field) => coreGrid.appendChild(makeFieldEl(field, defaults)));
  form.appendChild(coreGrid);

  if (location.length) {
    const details = document.createElement("details");
    details.className = "geo-optional";
    details.innerHTML =
      "<summary>Location (optional — leave blank to use typical training values)</summary>";
    const geoGrid = document.createElement("div");
    geoGrid.className = "form-grid";
    location.forEach((field) => geoGrid.appendChild(makeFieldEl(field, defaults, true)));
    details.appendChild(geoGrid);
    form.appendChild(details);
  }

  ["lat", "long", "merch_lat", "merch_long"].forEach((n) => {
    const input = form.elements[n];
    if (input) input.addEventListener("input", () => updateDistancePreview(form));
  });
  updateDistancePreview(form);
}

function makeFieldEl(field, defaults, optional = false) {
  const name = field.name;
  const label = document.createElement("label");
  label.className = "field";
  const title = field.description ? ` title="${field.description.replace(/"/g, "&quot;")}"` : "";
  const val = defaults[name] ?? (field.type === "categorical" ? field.options?.[0] : "");
  const optAttr = optional ? ' placeholder="optional"' : "";

  if (field.type === "categorical" && field.options?.length) {
    const opts = field.options
      .map((o) => `<option value="${o}"${String(val) === String(o) ? " selected" : ""}>${o}</option>`)
      .join("");
    label.innerHTML = `<span${title}>${field.label || name}</span><select name="${name}">${opts}</select>`;
  } else {
    const v = val === "" ? "" : val;
    label.innerHTML = `<span${title}>${field.label || name}</span><input name="${name}" type="number" step="any" value="${v}"${optAttr} />`;
  }
  return label;
}

function fillManualForm(row) {
  const form = $("#form-manual");
  if (!form) return;
  Object.entries(row).forEach(([key, val]) => {
    if (key.startsWith("is_fraud") || key === "note" || key === "path" || key === "row_index") return;
    const el = form.elements[key];
    if (!el) return;
    el.value = val;
  });
  updateDistancePreview(form);
  const hint = $("#import-label-hint");
  if (hint && row.is_fraud_actual !== undefined) {
    hint.textContent = `Loaded row ${row.row_index}: CSV is_fraud_actual = ${row.is_fraud_actual} (ground truth)`;
    hint.classList.remove("hidden");
  }
}

function renderLayers(layers, container) {
  container.innerHTML = layers
    .map(
      (l) => `
    <article class="layer-tile">
      <span class="layer-id">${l.id}</span>
      <h3>${l.name}</h3>
      <p>${l.responsibility}</p>
    </article>`
    )
    .join("");
}

function updateHealthUI(health, prod) {
  const sidebar = $("#sidebar-health");
  const status = health?.status || "offline";
  sidebar.classList.toggle("online", status === "healthy" || status === "warning");
  sidebar.classList.toggle("offline", status === "offline" || status === "degraded");
  $(".status-label", sidebar).textContent =
    status === "healthy" ? "System online" : status.toUpperCase();

  $("#metric-health").textContent = status;
  $("#metric-model").textContent = prod?.version || "Not trained";
  $("#metric-gov").textContent = health?.governance_action || "—";
}

function showVerdict(prob, gov, featuresSample, actualLabel) {
  $("#verdict-empty").classList.add("hidden");
  const filled = $("#verdict-filled");
  filled.classList.remove("hidden");
  const cmp = $("#verdict-compare");
  if (cmp) {
    if (actualLabel !== undefined && actualLabel !== null) {
      const modelPred = prob >= 0.5 ? 1 : 0;
      cmp.classList.remove("hidden");
      cmp.innerHTML = `<p><strong>CSV ground truth:</strong> is_fraud = ${actualLabel} · <strong>Model:</strong> ${modelPred} (prob ${prob.toFixed(4)})${
        Number(actualLabel) !== modelPred
          ? " — <em>These differ; the model is not 100% accurate on this row.</em>"
          : " — label and model agree."
      }</p>`;
    } else {
      cmp.classList.add("hidden");
    }
  }
  const featEl = $("#verdict-features");
  if (featEl && featuresSample?.length) {
    const row = featuresSample[0];
    const lines = Object.entries(row)
      .map(([k, v]) => `<li><code>${k}</code> = ${typeof v === "number" ? Number(v).toFixed(4) : v}</li>`)
      .join("");
    featEl.innerHTML = `<p class="muted">Model inputs (after featurization):</p><ul class="feat-list">${lines}</ul>`;
    featEl.classList.remove("hidden");
  } else if (featEl) {
    featEl.classList.add("hidden");
  }

  const pct = Math.min(1, Math.max(0, prob));
  const ring = $("#prob-ring");
  const fg = $("#ring-fg");
  ring.classList.toggle("fraud", pct >= 0.5);
  fg.style.strokeDashoffset = String(RING_C * (1 - pct));
  $("#prob-value").textContent = pct.toFixed(4);
  $("#verdict-gov").textContent = gov || "—";

  const pill = $("#verdict-pill");
  const isFraud = pct >= 0.5;
  pill.textContent = isFraud ? "FRAUD DETECTED" : "LEGITIMATE";
  pill.classList.toggle("fraud", isFraud);
}

function renderBatchResults(data, rows) {
  const preds = data.predictions || [];
  const summary = data.batch_summary || {};
  const fraudCount = preds.filter((p) => p.fraud_prediction === 1).length;
  const meanProb =
    preds.length > 0 ? preds.reduce((s, p) => s + p.fraud_probability, 0) / preds.length : 0;

  const gov = data.governance_action || "continue";
  $("#batch-summary-text").textContent = `Analyzed ${rows} transactions · governance: ${gov}`;
  if (gov === "alert" || gov === "retrain_recommended") {
    toast(`Drift advisory: ${gov} — ${(summary.drifted_features || data.batch_summary?.drifted_features || []).length} features`, "error");
  }
  const metrics = $("#batch-metrics");
  metrics.classList.remove("hidden");
  metrics.innerHTML = `
    <div class="mini-metric"><span>Rows</span><strong>${rows}</strong></div>
    <div class="mini-metric"><span>Fraud flags</span><strong>${fraudCount}</strong></div>
    <div class="mini-metric"><span>Mean prob</span><strong>${meanProb.toFixed(4)}</strong></div>
    <div class="mini-metric"><span>Governance</span><strong>${data.governance_action}</strong></div>`;

  const hasLabels = preds.some((p) => p.is_fraud_actual !== undefined);
  if (summary.has_ground_truth) {
    const recall = summary.recall_on_labeled_fraud;
    const extra = recall !== undefined
      ? ` · recall on labeled fraud: ${(recall * 100).toFixed(1)}%`
      : "";
    $("#batch-summary-text").textContent += ` · CSV fraud: ${summary.actual_fraud_in_batch ?? "—"} · model flagged: ${summary.model_flagged_count ?? "—"}${extra}`;
  }

  const tbody = $("#results-table tbody");
  tbody.innerHTML = preds
    .slice(0, 100)
    .map((p, i) => {
      const fraud = p.fraud_prediction === 1;
      const actual =
        p.is_fraud_actual !== undefined
          ? `<span class="tag ${p.is_fraud_actual === 1 ? "fraud" : "legit"}">${p.is_fraud_actual === 1 ? "FRAUD" : "LEGIT"}</span>`
          : "—";
      const match =
        p.is_fraud_actual !== undefined
          ? p.model_agrees_with_label
            ? "✓"
            : "≠"
          : "—";
      return `<tr class="${fraud ? "high-risk" : ""}">
        <td>${i + 1}</td>
        <td>${p.fraud_probability.toFixed(4)}</td>
        <td><span class="tag ${fraud ? "fraud" : "legit"}">${fraud ? "FRAUD" : "LEGIT"}</span></td>
        <td>${hasLabels ? actual : "—"}</td>
        <td>${hasLabels ? match : "—"}</td>
      </tr>`;
    })
    .join("");

  $("#table-wrap").classList.remove("hidden");
  $("#btn-download-csv").classList.remove("hidden");

  const header = hasLabels
    ? "index,fraud_probability,fraud_prediction,is_fraud_actual,model_agrees_with_label"
    : "index,fraud_probability,fraud_prediction";
  lastBatchCsv = [header].concat(
    preds.map((p, i) => {
      const base = `${i},${p.fraud_probability},${p.fraud_prediction}`;
      if (!hasLabels) return base;
      return `${base},${p.is_fraud_actual ?? ""},${p.model_agrees_with_label ?? ""}`;
    })
  ).join("\n");
}

function drawChart() {
  const canvas = $("#chart-prob");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (probHistory.length < 2) return;

  const max = Math.max(0.01, ...probHistory, 0.5);
  ctx.strokeStyle = "rgba(61, 255, 168, 0.9)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  probHistory.forEach((v, i) => {
    const x = (i / (probHistory.length - 1)) * (w - 20) + 10;
    const y = h - 10 - (v / max) * (h - 20);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function appendFeed(snap) {
  const feed = $("#live-feed");
  const li = document.createElement("li");
  li.innerHTML = `<strong>Batch ${snap.batch_id}</strong> · prob ${snap.mean_fraud_probability.toFixed(3)} · flagged ${snap.fraud_detected} · drift ${snap.drifted_features?.length || 0}`;
  feed.prepend(li);
  while (feed.children.length > 40) feed.lastChild.remove();
}

function handleLiveSnapshot(snap) {
  $("#live-processed").textContent = snap.total_processed.toLocaleString();
  $("#live-prob").textContent = snap.mean_fraud_probability.toFixed(4);
  $("#live-flagged").textContent = snap.fraud_detected;
  $("#live-gov").textContent = snap.governance_action;

  probHistory.push(snap.mean_fraud_probability);
  if (probHistory.length > 60) probHistory.shift();
  drawChart();
  appendFeed(snap);

  if (snap.blocked) {
    toast("Governance blocked serving — drift threshold exceeded", "error");
    stopStream();
  }
}

function stopStream() {
  if (streamSocket) {
    streamSocket.close();
    streamSocket = null;
  }
  $("#btn-stream-start").disabled = false;
  $("#btn-stream-stop").disabled = true;
}

function startStream() {
  const max = parseInt($("#stream-max").value, 10) || 80;
  stopStream();
  probHistory = [];
  $("#btn-stream-start").disabled = true;
  $("#btn-stream-stop").disabled = false;

  try {
    streamSocket = openRealtimeSocket(max);
    streamSocket.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.event === "governance_block") return;
        handleLiveSnapshot(data);
      } catch {
        /* ignore */
      }
    };
    streamSocket.onclose = () => {
      $("#btn-stream-start").disabled = false;
      $("#btn-stream-stop").disabled = true;
      toast("Live stream ended");
    };
    streamSocket.onerror = () => toast("WebSocket connection failed", "error");
    toast("Live stream connected", "success");
  } catch (err) {
    toast(err.message, "error");
    stopStream();
  }
}

async function init() {
  $("#api-base").textContent = API_BASE;

  $$("[data-nav]").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (el.dataset.nav) {
        e.preventDefault();
        navigate(el.dataset.nav);
      }
    });
  });

  $$("[data-goto]").forEach((btn) => {
    btn.addEventListener("click", () => navigate(btn.dataset.goto));
  });

  const hash = location.hash.replace("#", "");
  if (PAGE_COPY[hash]) navigate(hash);

  $("#menu-toggle")?.addEventListener("click", () => {
    $("#sidebar").classList.toggle("open");
  });

  $$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.batchTab;
      $("#batch-path").classList.toggle("active", tab === "path");
      $("#batch-upload").classList.toggle("active", tab === "upload");
    });
  });

  const dz = $("#dropzone");
  if (dz) {
    ["dragenter", "dragover"].forEach((ev) => {
      dz.addEventListener(ev, (e) => {
        e.preventDefault();
        dz.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((ev) => {
      dz.addEventListener(ev, (e) => {
        e.preventDefault();
        dz.classList.remove("dragover");
      });
    });
  }

  $("#form-manual")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const rec = {};
    const schema = systemInfo?.feature_schema || [];
    const schemaByName = Object.fromEntries(schema.map((f) => [f.name, f]));

    (systemInfo?.feature_columns || []).forEach((col) => {
      const raw = fd.get(col);
      if (raw === null || raw === "") return;
      const meta = schemaByName[col];
      if (meta?.type === "categorical") {
        rec[col] = String(raw);
      } else {
        const num = parseFloat(raw);
        if (!Number.isNaN(num)) rec[col] = num;
      }
    });
    const actualHint = $("#import-label-hint")?.textContent?.match(/is_fraud_actual = (\d)/);
    const actualLabel = actualHint ? parseInt(actualHint[1], 10) : undefined;
    try {
      const data = await scoreTransactions([rec]);
      const p = data.predictions[0];
      const actual = p.is_fraud_actual ?? actualLabel;
      showVerdict(
        p.fraud_probability,
        data.governance_action,
        data.batch_summary?.features_used_sample,
        actual
      );
      toast("Analysis complete", "success");
    } catch (err) {
      toast(err.message, "error");
    }
  });

  $("#form-path")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    try {
      const data = await scoreFromPath(fd.get("path"), parseInt(fd.get("max_rows"), 10));
      renderBatchResults(data, data.rows_scored || 0);
      toast("Batch analysis complete", "success");
    } catch (err) {
      toast(err.message, "error");
    }
  });

  $("#form-upload")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    try {
      const data = await scoreFromUpload(fd.get("file"), parseInt(fd.get("max_rows"), 10));
      renderBatchResults(data, data.rows_scored || 0);
      toast("Upload analyzed", "success");
    } catch (err) {
      toast(err.message, "error");
    }
  });

  $("#btn-download-csv")?.addEventListener("click", () => {
    if (!lastBatchCsv) return;
    const blob = new Blob([lastBatchCsv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "sentinel_fraud_scores.csv";
    a.click();
  });

  $("#btn-stream-start")?.addEventListener("click", startStream);
  $("#btn-stream-stop")?.addEventListener("click", stopStream);

  $("#btn-import-row")?.addEventListener("click", async () => {
    const path = $("#import-row-path")?.value || "data/credit_dt/fraudTest.csv";
    const rowIndex = parseInt($("#import-row-index")?.value, 10) || 0;
    try {
      const row = await getCsvRow(path, rowIndex);
      fillManualForm(row);
      toast(`Loaded row ${rowIndex} from CSV`, "success");
    } catch (err) {
      toast(err.message, "error");
    }
  });

  try {
    const [info, arch] = await Promise.all([getSystemInfo(), getArchitecture()]);
    systemInfo = info;

    $("#dataset-badge").textContent = info.dataset;
    $("#metric-dataset").textContent = info.dataset;
    updateHealthUI(info.health, info.production_model);

    buildManualForm(info.feature_schema, info.feature_defaults || {});

    const pipelineEl = $("#scoring-pipeline-hint");
    const stepsEl = $("#pipeline-steps");
    if (pipelineEl && stepsEl && info.scoring_pipeline?.length) {
      pipelineEl.hidden = false;
      stepsEl.innerHTML = info.scoring_pipeline.map((s) => `<li>${s}</li>`).join("");
    }

    const pathInput = $('input[name="path"]', $("#form-path"));
    if (pathInput && info.paths?.test) pathInput.value = info.paths.test;

    $("#stream-path-label").textContent = `Stream source: ${info.paths?.stream || "—"}`;

    renderLayers(arch.layers, $("#layer-grid"));
    renderLayers(arch.layers, $("#system-layers"));

    $("#system-model").textContent = JSON.stringify(info.production_model || {}, null, 2);
    $("#system-features").innerHTML = (info.feature_schema || info.feature_columns.map((f) => ({ name: f })))
      .map(
        (f) =>
          `<li><strong>${f.label || f.name}</strong>${f.description ? ` — ${f.description}` : ""}${
            f.options ? ` <em>(${f.options.length} labels)</em>` : ""
          }</li>`
      )
      .join("");
  } catch (err) {
    toast(`API offline: ${err.message}. Run: python main.py --api`, "error");
    updateHealthUI({ status: "offline" }, null);
  }

  const canvas = $("#chart-prob");
  if (canvas) {
    canvas.width = canvas.parentElement?.clientWidth || 400;
    canvas.height = 120;
  }
}

document.addEventListener("DOMContentLoaded", init);
