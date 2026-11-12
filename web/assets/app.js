const API_BASE = (() => {
  const configured = (window.RAG_API_BASE || "").replace(/\/$/, "");
  if (configured) return configured;
  if (["localhost", "127.0.0.1"].includes(window.location.hostname)) return "http://127.0.0.1:8001";
  return "";
})();

const WORKSPACE_ID_KEY = "rag-workspace-id";
const WORKSPACE_TOKEN_KEY = "rag-workspace-token";
const state = { workspaceId: "", token: "", documents: [], formats: null, busy: false };
const $ = (selector) => document.querySelector(selector);

const apiStatus = $("#apiStatus");
const fileInput = $("#fileInput");
const dropzone = $("#dropzone");
const uploadProgress = $("#uploadProgress");
const uploadStatus = $("#uploadStatus");
const uploadError = $("#uploadError");
const askForm = $("#askForm");
const questionInput = $("#question");
const askButton = $("#askButton");
const askError = $("#askError");
const toast = $("#toast");

function setStatus(kind, label) {
  apiStatus.className = `status-pill ${kind}`;
  apiStatus.lastChild.textContent = label;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  window.setTimeout(() => toast.classList.remove("visible"), 3200);
}

async function api(path, options = {}) {
  if (!API_BASE) throw new Error("The API address is not configured. Set VITE_API_BASE_URL in the Vercel project.");
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    const detail = payload?.detail;
    if (typeof detail === "string") throw new Error(detail);
    if (detail?.message) {
      const failures = (detail.errors || []).map((item) => `${item.filename}: ${item.error}`).join("\n");
      throw new Error([detail.message, failures].filter(Boolean).join("\n"));
    }
    throw new Error(`Request failed (${response.status}).`);
  }
  return payload;
}

function authHeaders() {
  return state.token ? { Authorization: `Bearer ${state.token}` } : {};
}

function saveSession() {
  sessionStorage.setItem(WORKSPACE_ID_KEY, state.workspaceId);
  sessionStorage.setItem(WORKSPACE_TOKEN_KEY, state.token);
}

function clearSession() {
  sessionStorage.removeItem(WORKSPACE_ID_KEY);
  sessionStorage.removeItem(WORKSPACE_TOKEN_KEY);
  state.workspaceId = "";
  state.token = "";
  state.documents = [];
  renderDocuments();
}

async function ensureWorkspace() {
  if (state.workspaceId && state.token) return;
  const created = await api("/v1/workspaces", { method: "POST" });
  state.workspaceId = created.workspace_id;
  state.token = created.access_token;
  saveSession();
}

function showError(element, message) {
  element.textContent = message;
  element.hidden = !message;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function renderDocuments() {
  const list = $("#fileList");
  list.replaceChildren();
  $("#documentCount").textContent = String(state.documents.length);
  for (const document of state.documents) {
    const item = document.createElement("li");
    item.className = "file-item";
    const icon = document.createElement("span");
    icon.className = "file-icon";
    icon.textContent = document.filename.split(".").pop()?.slice(0, 4) || "FILE";
    icon.setAttribute("aria-hidden", "true");
    const meta = document.createElement("span");
    meta.className = "file-meta";
    const name = document.createElement("span");
    name.className = "file-name";
    name.textContent = document.filename;
    const size = document.createElement("span");
    size.className = "file-size";
    size.textContent = formatSize(document.size_bytes);
    meta.append(name, size);
    item.append(icon, meta);
    list.append(item);
  }
}

async function loadDocuments() {
  if (!state.workspaceId || !state.token) return;
  try {
    const result = await api(`/v1/workspaces/${encodeURIComponent(state.workspaceId)}/documents`, {
      headers: authHeaders(),
    });
    state.documents = result.documents || [];
    renderDocuments();
  } catch {
    clearSession();
  }
}

async function initialize() {
  try {
    const [health, formats] = await Promise.all([api("/v1/health"), api("/v1/formats")]);
    if (health.status === "ok") setStatus("online", "API ready");
    state.formats = formats;
    const limits = formats.limits;
    $("#formatHint").textContent = `Documents, code, images, readable text · ${formatSize(limits.per_file_bytes)} / file · ${formatSize(limits.per_workspace_bytes)} total`;
  } catch (error) {
    setStatus("offline", "API unavailable");
    showError(uploadError, error.message);
  }

  state.workspaceId = sessionStorage.getItem(WORKSPACE_ID_KEY) || "";
  state.token = sessionStorage.getItem(WORKSPACE_TOKEN_KEY) || "";
  await loadDocuments();
}

async function uploadFiles(files) {
  if (!files.length || state.busy) return;
  showError(uploadError, "");
  if (!state.formats) {
    showError(uploadError, "The backend is unavailable. Try again after it is online.");
    return;
  }
  const maxFile = state.formats.limits.per_file_bytes;
  const tooLarge = files.filter((file) => file.size > maxFile);
  if (tooLarge.length) {
    showError(uploadError, `${tooLarge.map((file) => file.name).join(", ")} exceed the ${formatSize(maxFile)} file limit.`);
    return;
  }
  state.busy = true;
  uploadProgress.hidden = false;
  uploadStatus.textContent = "Preparing a private workspace…";
  dropzone.setAttribute("aria-busy", "true");
  try {
    await ensureWorkspace();
    const form = new FormData();
    for (const file of files) form.append("files", file, file.webkitRelativePath || file.name);
    uploadStatus.textContent = `Extracting text from ${files.length} file${files.length === 1 ? "" : "s"}…`;
    const result = await api(`/v1/workspaces/${encodeURIComponent(state.workspaceId)}/documents`, {
      method: "POST",
      headers: authHeaders(),
      body: form,
    });
    state.documents = result.documents || [];
    renderDocuments();
    const rejected = (result.errors || []).map((item) => `${item.filename}: ${item.error}`);
    showError(uploadError, rejected.join("\n"));
    showToast(`${files.length - rejected.length} file(s) added · ${result.chunk_count} searchable passages`);
  } catch (error) {
    showError(uploadError, error.message);
  } finally {
    state.busy = false;
    uploadProgress.hidden = true;
    dropzone.removeAttribute("aria-busy");
    fileInput.value = "";
    $("#folderInput").value = "";
  }
}

function selectedMode() {
  return document.querySelector('input[name="answerMode"]:checked')?.value || "documents";
}

function renderAnswerText(answer, citationsEnabled) {
  const container = $("#answerText");
  container.replaceChildren();
  if (!citationsEnabled) {
    container.textContent = answer;
    return;
  }
  const pattern = /\[\s*(\d+)\s*\]/g;
  let start = 0;
  let match;
  while ((match = pattern.exec(answer)) !== null) {
    container.append(document.createTextNode(answer.slice(start, match.index)));
    const button = document.createElement("button");
    button.type = "button";
    button.className = "citation-link";
    button.textContent = `[${match[1]}]`;
    button.setAttribute("aria-label", `Jump to source passage ${match[1]}`);
    button.addEventListener("click", () => document.getElementById(`evidence-${match[1]}`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
    container.append(button);
    start = pattern.lastIndex;
  }
  container.append(document.createTextNode(answer.slice(start)));
}

function renderEvidence(citations) {
  const section = $("#evidenceSection");
  const list = $("#evidenceList");
  list.replaceChildren();
  section.hidden = !citations?.length;
  for (const citation of citations || []) {
    const card = document.createElement("article");
    card.className = `evidence-card ${citation.was_cited ? "cited" : ""}`;
    card.id = `evidence-${citation.index}`;
    const header = document.createElement("div");
    header.className = "evidence-card-header";
    const sourceWrap = document.createElement("div");
    const source = document.createElement("div");
    source.className = "evidence-source";
    source.textContent = `[${citation.index}] ${citation.source_file}`;
    const heading = document.createElement("div");
    heading.className = "evidence-heading";
    heading.textContent = citation.heading_path || "Document passage";
    sourceWrap.append(source, heading);
    const tag = document.createElement("span");
    tag.className = "evidence-tag";
    tag.textContent = citation.was_cited ? "CITED" : "RETRIEVED";
    header.append(sourceWrap, tag);
    const excerpt = document.createElement("p");
    excerpt.className = "evidence-text";
    excerpt.textContent = citation.text_excerpt || "No excerpt was returned.";
    card.append(header, excerpt);
    if (citation.verified_claims?.length) {
      const checks = document.createElement("div");
      checks.className = "verification";
      for (const detail of citation.verified_claims) {
        const line = document.createElement("div");
        line.className = "verification-item";
        const verdict = document.createElement("span");
        verdict.className = `verdict ${String(detail.verdict).toLowerCase()}`;
        verdict.textContent = `${detail.verdict} · `;
        line.append(verdict, document.createTextNode(detail.claim));
        checks.append(line);
      }
      card.append(checks);
    }
    list.append(card);
  }
}

function renderTrace(stages) {
  const details = $("#traceDetails");
  const grid = $("#traceGrid");
  grid.replaceChildren();
  details.hidden = !stages;
  if (!stages) return;
  const groups = [
    ["Dense search", stages.dense_top || []],
    ["BM25 search", stages.sparse_top || []],
    ["RRF fusion", stages.fused_top || []],
  ];
  for (const [title, hits] of groups) {
    const column = document.createElement("div");
    column.className = "trace-column";
    const heading = document.createElement("h4");
    heading.textContent = title;
    column.append(heading);
    for (const hit of hits) {
      const row = document.createElement("div");
      row.className = "trace-hit";
      row.append(document.createTextNode(hit.source_file));
      const score = document.createElement("span");
      score.textContent = `${hit.heading_path} · ${Number(hit.score).toFixed(4)}`;
      row.append(score);
      column.append(row);
    }
    grid.append(column);
  }
}

function renderMetrics(result) {
  const metrics = $("#answerMetrics");
  metrics.replaceChildren();
  if (result.citation_accuracy !== null && result.citation_verification_available) {
    const accuracy = document.createElement("span");
    accuracy.className = "metric-chip";
    accuracy.textContent = `Citation check · ${Math.round(result.citation_accuracy * 100)}% supported`;
    metrics.append(accuracy);
  } else if (result.mode === "documents") {
    const unavailable = document.createElement("span");
    unavailable.className = "metric-chip";
    unavailable.textContent = "Citation verification unavailable";
    metrics.append(unavailable);
  }
  if (result.retrieval_score !== null && result.retrieval_score !== undefined) {
    const score = document.createElement("span");
    score.className = "metric-chip";
    score.textContent = `Top passage score · ${Number(result.retrieval_score).toFixed(4)} (not a probability)`;
    metrics.append(score);
  }
}

async function askQuestion(event) {
  event.preventDefault();
  if (state.busy) return;
  const question = questionInput.value.trim();
  const mode = selectedMode();
  if (!question) return;
  if (mode === "documents" && !state.documents.length) {
    showError(askError, "Upload at least one document for a grounded answer, or choose General AI.");
    return;
  }
  showError(askError, "");
  state.busy = true;
  askButton.disabled = true;
  askButton.querySelector("span").textContent = "Thinking…";
  try {
    await ensureWorkspace();
    const result = await api(`/v1/workspaces/${encodeURIComponent(state.workspaceId)}/ask`, {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode }),
    });
    $("#answerArea").hidden = false;
    const notice = $("#answerNotice");
    notice.textContent = result.notice;
    notice.className = `answer-notice ${result.mode === "general" ? "general" : ""}`;
    renderAnswerText(result.answer, result.mode === "documents");
    renderEvidence(result.citations);
    renderMetrics(result);
    renderTrace(result.pipeline_stages);
    $("#answerArea").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(askError, error.message);
  } finally {
    state.busy = false;
    askButton.disabled = false;
    askButton.querySelector("span").textContent = "Ask";
  }
}

async function deleteWorkspace() {
  if (state.workspaceId && state.token) {
    try {
      await api(`/v1/workspaces/${encodeURIComponent(state.workspaceId)}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
    } catch {
      // Clearing the local bearer token still makes this browser unable to access the workspace.
    }
  }
  clearSession();
  $("#answerArea").hidden = true;
  showError(uploadError, "");
  showError(askError, "");
  showToast("Workspace cleared from this session.");
}

fileInput.addEventListener("change", () => uploadFiles([...fileInput.files]));
$("#folderInput").addEventListener("change", (event) => uploadFiles([...event.target.files]));
askForm.addEventListener("submit", askQuestion);
$("#newWorkspace").addEventListener("click", deleteWorkspace);
$("#copyAnswer").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("#answerText").textContent);
    showToast("Answer copied.");
  } catch {
    showToast("Clipboard access is unavailable in this browser.");
  }
});
questionInput.addEventListener("input", () => {
  $("#characterCount").textContent = `${questionInput.value.length} / 4000`;
});
document.querySelectorAll('input[name="answerMode"]').forEach((input) => {
  input.addEventListener("change", () => {
    document.querySelectorAll(".mode-option").forEach((item) => item.classList.remove("selected"));
    input.closest(".mode-option").classList.add("selected");
  });
});
for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); });
}
dropzone.addEventListener("drop", (event) => uploadFiles([...event.dataTransfer.files]));

initialize();
