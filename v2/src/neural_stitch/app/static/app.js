"use strict";

const $ = (sel) => document.querySelector(sel);
const api = (path, opts) => fetch(path, opts).then(async (r) => {
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `Request failed (${r.status})`);
  return body;
});

let picked = [];      // File objects chosen by the user
let job = null;       // latest job state from the server
let events = null;    // EventSource
let losses = [];      // [iter, loss] for the chart
let maxIters = 6000;
let resultFile = "panorama";
let info = null;      // server settings from /api/info
let resBuiltFor = null;

const fileUrl = (name, download = false) =>
  `/api/jobs/${job.id}/files/${name}${download ? (name.includes("?") ? "&" : "?") + "download=true" : ""}`;

function showError(msg) {
  const el = $("#error");
  el.textContent = msg || "";
  el.hidden = !msg;
}

// ---- 1. Upload ---------------------------------------------------------------------------------

function renderPicked() {
  const list = $("#picked");
  list.replaceChildren(...picked.map((f, k) => {
    const li = document.createElement("li");
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    img.onload = () => URL.revokeObjectURL(img.src);
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = f.name;
    const rm = document.createElement("button");
    rm.className = "remove";
    rm.textContent = "×";
    rm.title = "Remove";
    rm.onclick = (e) => { e.preventDefault(); picked.splice(k, 1); renderPicked(); };
    li.append(img, name, rm);
    return li;
  }));
  $("#check-btn").disabled = picked.length < 2;
  $("#upload-msg").textContent = picked.length === 1 ? "Add at least one more image." : "";
}

function addFiles(files) {
  picked.push(...[...files].filter((f) => f.type.startsWith("image/") || /\.(png|jpe?g|tiff?|bmp|webp)$/i.test(f.name)));
  renderPicked();
}

const drop = $("#drop");
$("#file-input").onchange = (e) => { addFiles(e.target.files); e.target.value = ""; };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files); };

$("#check-btn").onclick = async () => {
  showError("");
  $("#check-btn").disabled = true;
  $("#upload-msg").textContent = "Uploading…";
  const form = new FormData();
  picked.forEach((f) => form.append("files", f, f.name));
  form.append("speed", $("#opt-speed").value);
  try {
    job = await api("/api/jobs", { method: "POST", body: form });
    $("#upload-msg").textContent = "";
    $("#check").hidden = false;
    setProgress("#check-progress", null, "Starting…");
    render();
    listen();
  } catch (err) {
    showError(err.message);
    $("#upload-msg").textContent = "";
    $("#check-btn").disabled = false;
  }
};

// ---- Live updates ------------------------------------------------------------------------------

function listen() {
  if (events) events.close();
  events = new EventSource(`/api/jobs/${job.id}/events`);
  events.addEventListener("state", refresh);
  events.addEventListener("progress", (e) => onProgress(JSON.parse(e.data)));
}

async function refresh() {
  try {
    job = await api(`/api/jobs/${job.id}`);
    render();
  } catch (err) {
    showError(err.message);
  }
}

function setProgress(sel, fraction, text) {
  const block = $(sel);
  block.hidden = false;
  const bar = block.querySelector(".bar");
  bar.classList.toggle("indeterminate", fraction == null);
  bar.firstElementChild.style.width = fraction == null ? "" : `${Math.round(fraction * 100)}%`;
  block.querySelector("span").textContent = text;
}

const STAGES = { features: "Finding keypoints", matching: "Matching pairs", bundle: "Least-squares alignment",
  neural: "Neural refinement", render: "Rendering" };

function onProgress(p) {
  if (!job) return;
  if (["features", "matching"].includes(p.stage) || (p.stage === "bundle" && job.state === "checking")) {
    setProgress("#check-progress", p.fraction, `${STAGES[p.stage]}: ${p.message}`);
    return;
  }
  if (p.stage === "neural") {
    const d = p.data;
    losses.push([d.iter, d.loss]);
    drawChart();
    const parts = [`Step ${d.iter} / ${maxIters}`, `loss ${d.loss.toFixed(5)}`];
    if (d.rel_improvement != null) parts.push(`improvement ${(d.rel_improvement * 100).toFixed(2)}%`);
    parts.push(`homographies moved ${d.corner_motion_px.toFixed(3)} px`);
    setProgress("#run-progress", p.fraction, parts.join(" · "));
  } else {
    setProgress("#run-progress", p.fraction, `${STAGES[p.stage] || p.stage}: ${p.message}`);
  }
  if (p.preview && job.state === "running") {
    const img = $("#live-preview");
    img.src = fileUrl(p.preview);
    img.hidden = false;
  }
}

function drawChart() {
  const line = $("#chart polyline");
  if (losses.length < 2) { line.setAttribute("points", ""); return; }
  const xs = losses.map((p) => p[0]), ys = losses.map((p) => Math.log(p[1]));
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const pts = losses.map(([x], k) =>
    `${((x - x0) / (x1 - x0 || 1)) * 300},${95 - ((ys[k] - y0) / (y1 - y0 || 1)) * 90}`);
  line.setAttribute("points", pts.join(" "));
}

// ---- 2-4. Render job state ---------------------------------------------------------------------

function render() {
  const s = job.state;
  $("#upload").querySelectorAll("button, input, select").forEach((el) => { el.disabled = true; });
  drop.style.pointerEvents = "none";

  if (s === "checking") return;
  $("#check-progress").hidden = true;

  const r = job.report;
  if (r) {
    const msg = $("#check-msg");
    msg.textContent = r.message;
    msg.classList.toggle("bad", !r.ok);
    $("#report").replaceChildren(...r.images.map((im) => {
      const li = document.createElement("li");
      li.className = im.reference ? "ref" : im.used ? "used" : "out";
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = im.reference ? "Reference" : im.used ? "Used" : "Left out";
      const img = document.createElement("img");
      img.src = fileUrl(`thumb_${im.index}.jpg`);
      const name = document.createElement("div");
      name.className = "name";
      name.textContent = im.name;
      li.append(badge, img, name);
      if (im.reason) {
        const why = document.createElement("div");
        why.className = "reason";
        why.textContent = im.reason;
        li.append(why);
      }
      return li;
    }));
    const names = job.images.map((im) => im.name);
    $("#pairs").hidden = false;
    $("#pairs tbody").replaceChildren(...r.pairs.map((p) => {
      const tr = document.createElement("tr");
      [`${names[p.i]} ↔ ${names[p.j]}`, p.matches, p.inliers, p.ok ? "✓ match" : p.reason].forEach((v) => {
        const td = document.createElement("td");
        td.textContent = v;
        tr.append(td);
      });
      return tr;
    }));
  }
  if (job.files.check_preview) {
    $("#check-preview").hidden = false;
    $("#check-preview img").src = fileUrl(job.files.check_preview);
  }
  if (s === "not_alignable") { $("#restart-inline")?.remove(); addRestart("#check"); return; }

  $("#run").hidden = false;
  const running = s === "queued" || s === "running";
  $("#start-btn").disabled = running;
  $("#opt-neural").disabled = running;
  $("#opt-iters").disabled = running;
  $("#opt-size").disabled = running;
  $("#opt-res").disabled = running;
  if (resBuiltFor !== job.id) buildResolutions();
  $("#cancel-btn").hidden = !running;
  if (s === "queued") setProgress("#run-progress", null, "Waiting for the worker…");
  if (s === "cancelled") setProgress("#run-progress", 0, "Cancelled.");
  if (s === "failed") { setProgress("#run-progress", 0, "Failed."); showError(job.error); }

  if (s === "done") {
    $("#run-progress").hidden = true;
    $("#result").hidden = false;
    const neural = job.metrics.neural_iterations != null;
    $(".tabs").hidden = !neural;
    $("#dl-pano").hidden = !neural;
    if (!neural) resultFile = "classic";
    showResult();
    $("#dl-pano").href = fileUrl(job.files.panorama, true);
    $("#dl-classic").href = fileUrl(job.files.classic, true);
    $("#dl-homs").href = fileUrl(job.files.homographies, true);
    renderMetrics();
  }
}

function showResult() {
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.file === resultFile)));
  $("#result-img").src = `${fileUrl(job.files[resultFile])}?t=${Date.now()}`;
}
document.querySelectorAll(".tabs button").forEach((b) => { b.onclick = () => { resultFile = b.dataset.file; showResult(); }; });

const METRIC_LABELS = {
  output_resolution: "Resolution",
  output_size: "Output size (px)",
  output_note: "Note",
  keypoint_rms_ransac_px: "Keypoint error after RANSAC (px)",
  keypoint_rms_bundle_px: "Keypoint error after least squares (px)",
  keypoint_rms_neural_px: "Keypoint error after neural refinement (px)",
  neural_iterations: "Neural steps",
  neural_stop_reason: "Stopped because",
  time_features_s: "Time: keypoints (s)",
  time_matching_s: "Time: matching (s)",
  time_bundle_s: "Time: least squares (s)",
  time_neural_s: "Time: neural (s)",
  time_render_s: "Time: rendering (s)",
};

function renderMetrics() {
  $("#metrics tbody").replaceChildren(...Object.entries(METRIC_LABELS)
    .filter(([k]) => job.metrics[k] != null)
    .map(([k, label]) => {
      const tr = document.createElement("tr");
      const v = job.metrics[k];
      [label, typeof v === "number" && !Number.isInteger(v) ? v.toFixed(3) : String(v).replace("_", " ")]
        .forEach((t) => { const td = document.createElement("td"); td.textContent = t; tr.append(td); });
      return tr;
    }));
}

// ---- Output resolution -------------------------------------------------------------------------

function presetSize(W0, H0, n) {
  return W0 >= H0 ? [Math.round(W0 * n / H0), n] : [n, Math.round(H0 * n / W0)];
}

function buildResolutions() {
  if (!info || !job.metrics.native_size) return;
  resBuiltFor = job.id;
  const [W0, H0] = job.metrics.native_size;
  const maxPx = info.output.max_megapixels * 1e6;
  const capNote = (w, h) => (w * h > maxPx ? ` (limited to ${info.output.max_megapixels} MP)` : "");
  const opts = [];
  const add = (value, text, disabled = false) => {
    const o = document.createElement("option");
    o.value = value; o.textContent = text; o.disabled = disabled;
    opts.push(o);
  };
  add("native", `Native — ${W0} × ${H0}${capNote(W0, H0)}`);
  let best = "native";
  for (const r of info.resolutions) {
    const name = r.label ? `${r.name} ${r.label}` : r.name;
    if (r.short_side > Math.min(W0, H0)) {
      add(r.name, `${name} — needs larger images`, true);
      continue;
    }
    const [w, h] = presetSize(W0, H0, r.short_side);
    add(r.name, `${name} — ${w} × ${h}${capNote(w, h)}`);
    if (r.short_side <= 1080) best = r.name;
  }
  add("custom", "Custom long side…");
  const sel = $("#opt-res");
  sel.replaceChildren(...opts);
  sel.value = best;
  $("#custom-size").hidden = true;
}

$("#opt-res").onchange = () => { $("#custom-size").hidden = $("#opt-res").value !== "custom"; };

function resolutionChoice() {
  const v = $("#opt-res").value;
  if (v === "custom") return { resolution: "native", max_side: Number($("#opt-size").value) || null };
  if (v === "native") return { resolution: "native", max_side: 50000 };  // full size; the MP limit still applies
  return { resolution: v };
}

function addRestart(sel) {
  const b = document.createElement("button");
  b.id = "restart-inline";
  b.textContent = "Try other images";
  b.onclick = restart;
  const row = document.createElement("div");
  row.className = "row";
  row.append(b);
  $(sel).append(row);
}

// ---- Actions -----------------------------------------------------------------------------------

$("#start-btn").onclick = async () => {
  showError("");
  losses = [];
  drawChart();
  $("#result").hidden = true;
  $("#live-preview").hidden = true;
  maxIters = Number($("#opt-iters").value) || maxIters;
  try {
    job = await api(`/api/jobs/${job.id}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ neural: $("#opt-neural").checked, max_iters: maxIters, ...resolutionChoice() }),
    });
    render();
  } catch (err) {
    showError(err.message);
  }
};

$("#cancel-btn").onclick = () => api(`/api/jobs/${job.id}/cancel`, { method: "POST" }).catch((e) => showError(e.message));

function restart() {
  if (events) events.close();
  if (job) fetch(`/api/jobs/${job.id}`, { method: "DELETE" });
  job = null; picked = []; losses = [];
  ["#check", "#run", "#result"].forEach((s) => { $(s).hidden = true; });
  $("#restart-inline")?.remove();
  $("#check-preview").hidden = true;
  $("#pairs").hidden = true;
  $("#check-msg").textContent = "";
  $("#report").replaceChildren();
  $("#upload").querySelectorAll("button, input, select").forEach((el) => { el.disabled = false; });
  drop.style.pointerEvents = "";
  showError("");
  renderPicked();
  window.scrollTo({ top: 0, behavior: "smooth" });
}
$("#restart-btn").onclick = restart;

// ---- Init --------------------------------------------------------------------------------------

api("/api/info").then((i) => {
  info = i;
  $("#device").textContent = `running on ${info.device} · ${info.backend}`;
  maxIters = info.neural.max_iters;
  $("#opt-iters").value = maxIters;
  $("#opt-size").value = info.output.max_side;
  $("#opt-neural").checked = info.neural.enabled;
}).catch(() => {});
renderPicked();
