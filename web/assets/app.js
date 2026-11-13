const API_BASE = (() => {
  const configured = (window.RAG_API_BASE || "").replace(/\/$/, "");
  if (configured) return configured;
  if (["localhost", "127.0.0.1"].includes(window.location.hostname)) return "http://127.0.0.1:8001";
  return "";
})();

const WORKSPACE_ID_KEY = "rag-workspace-id";
const WORKSPACE_TOKEN_KEY = "rag-workspace-token";
const state = { workspaceId: "", token: "", documents: [], chunkCount: 0, formats: null, busy: false };
const $ = (selector) => document.querySelector(selector);

const apiStatus = $("#apiStatus");
const fileInput = $("#fileInput");
const dropzone = $("#dropzone");
const uploadProgress = $("#uploadProgress");
const uploadStatus = $("#uploadStatus");
const uploadSummary = $("#uploadSummary");
const uploadError = $("#uploadError");
const askForm = $("#askForm");
const questionInput = $("#question");
const askButton = $("#askButton");
const answerProgress = $("#answerProgress");
const answerProgressText = $("#answerProgressText");
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
    let message;
    if (typeof detail === "string") message = detail;
    if (detail?.message) {
      const failures = (detail.errors || []).map((item) => `${item.filename}: ${item.error}`).join("\n");
      message = [detail.message, failures].filter(Boolean).join("\n");
    }
    const error = new Error(message || `Request failed (${response.status}).`);
    error.status = response.status;
    throw error;
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
  state.chunkCount = 0;
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

function showUploadSummary(message, kind = "success") {
  uploadSummary.textContent = message;
  uploadSummary.className = `upload-summary ${kind}`;
  uploadSummary.hidden = !message;
}

function setInteractionBusy(busy) {
  state.busy = busy;
  askButton.disabled = busy;
  $("#newWorkspace").disabled = busy;
  fileInput.disabled = busy;
  $("#folderInput").disabled = busy;
  questionInput.disabled = busy;
  document.querySelectorAll('input[name="answerMode"]').forEach((input) => { input.disabled = busy; });
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
  $("#emptyDocuments").hidden = state.documents.length > 0;
  for (const doc of state.documents) {
    const item = document.createElement("li");
    item.className = "file-item";
    const icon = document.createElement("span");
    icon.className = "file-icon";
    icon.textContent = doc.filename.split(".").pop()?.slice(0, 4) || "FILE";
    icon.setAttribute("aria-hidden", "true");
    const meta = document.createElement("span");
    meta.className = "file-meta";
    const name = document.createElement("span");
    name.className = "file-name";
    name.textContent = doc.filename;
    const size = document.createElement("span");
    size.className = "file-size";
    size.textContent = `${formatSize(doc.size_bytes)} · indexed`;
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
    state.chunkCount = result.chunk_count || 0;
    renderDocuments();
    if (state.documents.length) {
      showUploadSummary(`Workspace restored · ${state.documents.length} document${state.documents.length === 1 ? "" : "s"} indexed · ${state.chunkCount} searchable passage${state.chunkCount === 1 ? "" : "s"} ready.`, "success");
    }
  } catch (error) {
    if (error.status === 401 || error.status === 404) {
      clearSession();
      showUploadSummary("This temporary workspace expired. Upload your files again to continue.", "warning");
    } else {
      showUploadSummary("Could not restore this workspace yet. Your session has been kept; try again after the API reconnects.", "warning");
      showError(uploadError, error.message);
    }
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
  showUploadSummary("");
  if (!state.formats) {
    showError(uploadError, "The backend is unavailable. Try again after it is online.");
    showUploadSummary("Upload could not start because the API is not ready. Wait for the API status to turn green, then try again.", "error");
    fileInput.value = "";
    $("#folderInput").value = "";
    return;
  }
  const limits = state.formats.limits;
  const currentBytes = state.documents.reduce((total, doc) => total + (doc.size_bytes || 0), 0);
  const filesToUpload = [];
  const localRejections = [];
  let reservedBytes = currentBytes;
  let reservedFiles = state.documents.length;
  for (const file of files) {
    const filename = file.webkitRelativePath || file.name;
    if (file.size > limits.per_file_bytes) {
      localRejections.push(`${filename}: exceeds the ${formatSize(limits.per_file_bytes)} per-file limit.`);
    } else if (reservedFiles >= limits.files_per_workspace) {
      localRejections.push(`${filename}: the workspace file limit (${limits.files_per_workspace}) has been reached.`);
    } else if (reservedBytes + file.size > limits.per_workspace_bytes) {
      localRejections.push(`${filename}: exceeds the ${formatSize(limits.per_workspace_bytes)} workspace total.`);
    } else {
      filesToUpload.push(file);
      reservedBytes += file.size;
      reservedFiles += 1;
    }
  }
  if (!filesToUpload.length) {
    const message = localRejections.length ? localRejections.join("\n") : "No files were selected.";
    showError(uploadError, message);
    showUploadSummary(`No files were added. ${localRejections.length} file${localRejections.length === 1 ? " was" : "s were"} outside the current workspace limits.`, "error");
    fileInput.value = "";
    $("#folderInput").value = "";
    return;
  }
  setInteractionBusy(true);
  uploadProgress.hidden = false;
  uploadStatus.textContent = "Waking the API and preparing your private workspace…";
  dropzone.setAttribute("aria-busy", "true");
  try {
    await ensureWorkspace();
    const form = new FormData();
    for (const file of filesToUpload) form.append("files", file, file.webkitRelativePath || file.name);
    uploadStatus.textContent = `Uploading ${filesToUpload.length} file${filesToUpload.length === 1 ? "" : "s"} and building the search index…`;
    const previousCount = state.documents.length;
    const result = await api(`/v1/workspaces/${encodeURIComponent(state.workspaceId)}/documents`, {
      method: "POST",
      headers: authHeaders(),
      body: form,
    });
    state.documents = result.documents || [];
    state.chunkCount = result.chunk_count || 0;
    renderDocuments();
    const serverRejections = (result.errors || []).map((item) => `${item.filename}: ${item.error}`);
    const rejections = [...localRejections, ...serverRejections];
    const addedCount = Math.max(0, state.documents.length - previousCount);
    showError(uploadError, rejections.join("\n"));
    if (addedCount) {
      const partial = rejections.length ? " Some selected files were skipped; see the details above." : "";
      showUploadSummary(`Upload complete · ${addedCount} new document${addedCount === 1 ? "" : "s"} indexed · ${state.chunkCount} searchable passage${state.chunkCount === 1 ? "" : "s"} ready. Ask a question in Grounded mode.${partial}`, rejections.length ? "warning" : "success");
    } else {
      showUploadSummary("No new documents were added. Check the details above, then choose different files.", "warning");
    }
  } catch (error) {
    showError(uploadError, error.message);
    showUploadSummary("Upload did not finish. Your existing documents are still in this workspace; check the error and try again.", "error");
  } finally {
    setInteractionBusy(false);
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
  const citedCount = (citations || []).filter((citation) => citation.was_cited).length;
  $("#evidenceHelp").textContent = citedCount
    ? "Green outlined passages were cited in the answer. Other passages were retrieved candidates only. Click a citation such as [1] to jump to its source."
    : "These passages were retrieved for your question, but none was cited in a verifiable answer. The app shows them so you can inspect what it searched.";
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
  const addMetric = (label) => {
    const chip = document.createElement("span");
    chip.className = "metric-chip";
    chip.textContent = label;
    metrics.append(chip);
  };
  if (result.mode === "general") {
    addMetric("No document sources used");
    return;
  }
  const citations = result.citations || [];
  const citedCount = citations.filter((citation) => citation.was_cited).length;
  if (result.grounded && citedCount) {
    addMetric(`Answer cites ${citedCount} of ${citations.length} retrieved passage${citations.length === 1 ? "" : "s"}`);
  } else if (citations.length) {
    addMetric(`${citations.length} passage${citations.length === 1 ? "" : "s"} retrieved · no grounded answer returned`);
  } else {
    addMetric("No relevant passages found");
  }
  if (result.citation_verification_available && result.citation_accuracy !== null) {
    addMetric(`Citation check · ${Math.round(result.citation_accuracy * 100)}% of cited passages supported`);
  } else if (result.grounded) {
    addMetric(result.notice?.toLowerCase().includes("skipped")
      ? "Automated citation check skipped"
      : "Automated citation check unavailable");
  }
}

function renderAnswerMethod(result) {
  const summary = $("#methodSummary");
  const steps = $("#methodSteps");
  steps.replaceChildren();
  const method = result.mode === "general"
    ? {
        summary: "General AI answers directly from the question. It does not search your documents or the live web.",
        steps: [
          ["Question", "Your prompt is sent to the configured AI provider."],
          ["Answer", "The model returns a general response without document citations or web verification."],
        ],
      }
    : {
        summary: result.grounded
          ? "Grounded mode searches your files first and returns an answer linked to the passages it used."
          : result.citations?.length
            ? "Likely passages were retrieved, but the AI did not provide a verifiable answer from them. The app withheld the answer instead of guessing."
            : "No relevant passages were found in your files, so the app withheld an answer instead of guessing.",
        steps: [
          ["Search", "Meaning-based search and keyword search find candidate passages in your uploaded files."],
          ["Rank", result.citations?.length
            ? `Reciprocal rank fusion combined the results; ${result.citations.length} passage${result.citations.length === 1 ? " was" : "s were"} retrieved for review.`
            : "The search did not find a passage relevant enough to answer safely."],
          ["Answer", result.grounded
            ? "The AI writes from the retrieved passages and attaches source markers to factual claims."
            : "When the passages are insufficient or a claim cannot be supported, the app withholds the answer."],
          ["Check", result.citation_verification_available
            ? result.grounded
              ? "A separate citation check compares cited claims with their source passages."
              : "The citation checker found an unsupported claim, so the answer was withheld."
            : result.grounded
              ? "Automated citation checking was unavailable for this answer; review the source passages below."
              : result.citations?.length
                ? "No citation check ran because the AI draft did not contain valid source markers."
                : "No retrieved source passages were available to check."],
        ],
      };
  summary.textContent = method.summary;
  for (const [title, description] of method.steps) {
    const item = document.createElement("li");
    const heading = document.createElement("strong");
    heading.textContent = title;
    const detail = document.createElement("span");
    detail.textContent = description;
    item.append(heading, detail);
    steps.append(item);
  }
}

function updateModeExplanation() {
  $("#modeExplanation").textContent = selectedMode() === "documents"
    ? "Searches your uploaded files and shows the passages used. If the files do not support an answer, the app says so instead of guessing."
    : "Sends your question to the AI provider without searching your files or the live web. The answer will not have document citations.";
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
  $("#answerArea").hidden = true;
  setInteractionBusy(true);
  answerProgress.hidden = false;
  answerProgressText.textContent = mode === "documents"
    ? "Searching your documents, writing an evidence-based answer, and checking citations…"
    : "Sending your question to the AI provider. Your files and the web are not being searched…";
  askButton.querySelector("span").textContent = mode === "documents" ? "Searching…" : "Answering…";
  try {
    await ensureWorkspace();
    const result = await api(`/v1/workspaces/${encodeURIComponent(state.workspaceId)}/ask`, {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode }),
    });
    $("#answerArea").hidden = false;
    $("#answeredQuestion").textContent = question;
    const notice = $("#answerNotice");
    notice.textContent = result.notice;
    notice.className = `answer-notice ${result.mode === "general" ? "general" : result.grounded ? "grounded" : "withheld"}`;
    renderAnswerText(result.answer, result.mode === "documents");
    renderEvidence(result.citations);
    renderMetrics(result);
    renderAnswerMethod(result);
    renderTrace(result.pipeline_stages);
    $("#answerArea").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(askError, error.message);
  } finally {
    setInteractionBusy(false);
    answerProgress.hidden = true;
    askButton.querySelector("span").textContent = "Ask";
  }
}

async function deleteWorkspace() {
  if (state.busy) return;
  setInteractionBusy(true);
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
  showUploadSummary("");
  setInteractionBusy(false);
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
    updateModeExplanation();
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
