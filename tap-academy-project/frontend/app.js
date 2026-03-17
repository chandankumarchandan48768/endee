/* ================================================================
   Endee RAG — Frontend Application Logic
   ================================================================ */

const API = "http://localhost:8000";

// ── State ─────────────────────────────────────────────────────

let isAsking = false;

// ── Init ──────────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  setInterval(checkHealth, 30000);
});

// ── Health check ──────────────────────────────────────────────

async function checkHealth() {
  const badge = document.getElementById("status-badge");
  const txt   = document.getElementById("status-text");
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    if (r.ok) {
      const d = await r.json();
      badge.className = "status-badge ok";
      txt.textContent = `Endee Connected · ${d.index || "documents"}`;
    } else {
      throw new Error(r.statusText);
    }
  } catch {
    badge.className = "status-badge error";
    txt.textContent = "Backend Offline";
  }
}

// ── Tab switching ─────────────────────────────────────────────

function switchTab(name) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`tab-${name}`).classList.add("active");
  document.getElementById(`nav-${name}`).classList.add("active");
  if (name === "docs") loadDocuments();
}

// ── Upload ────────────────────────────────────────────────────

function dragOver(e) {
  e.preventDefault();
  document.getElementById("upload-zone").classList.add("drag-over");
}
function dragLeave(e) {
  document.getElementById("upload-zone").classList.remove("drag-over");
}
function dropFile(e) {
  e.preventDefault();
  dragLeave(e);
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  if (!file) return;
  const allowed = [".pdf", ".txt", ".md"];
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!allowed.includes(ext)) {
    toast("Only PDF, TXT, and MD files are supported.", "error");
    return;
  }

  const progress = document.getElementById("upload-progress");
  const bar      = document.getElementById("progress-bar");
  const statusTxt = document.getElementById("upload-status-text");

  progress.classList.remove("hidden");
  bar.style.width = "0%";
  statusTxt.textContent = `Uploading ${file.name}…`;

  // animate progress bar
  let pct = 0;
  const tick = setInterval(() => {
    pct = Math.min(pct + 4, 85);
    bar.style.width = pct + "%";
  }, 120);

  try {
    const fd = new FormData();
    fd.append("file", file);

    const r = await fetch(`${API}/upload`, { method: "POST", body: fd });
    clearInterval(tick);
    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || r.statusText);
    }
    const data = await r.json();
    bar.style.width = "100%";
    statusTxt.textContent = `✅ ${data.chunks_indexed} chunks indexed`;
    toast(`"${file.name}" indexed: ${data.chunks_indexed} chunks`, "success");
    setTimeout(() => progress.classList.add("hidden"), 3000);
    // reset input
    document.getElementById("file-input").value = "";
  } catch (e) {
    clearInterval(tick);
    bar.style.width = "0%";
    statusTxt.textContent = `❌ ${e.message}`;
    toast(`Upload failed: ${e.message}`, "error");
  }
}

// ── Ingest samples ────────────────────────────────────────────

async function ingestSamples() {
  const btn = document.getElementById("btn-ingest");
  btn.textContent = "⏳ Loading…";
  btn.disabled = true;

  const sampleFiles = ["ai_overview.txt", "vector_databases_and_rag.txt"];
  let success = 0;

  for (const fname of sampleFiles) {
    try {
      // Fetch the sample file from the data directory via the static server
      // Since we're calling the API backend, use the ingest endpoint with a URL or
      // fetch from the static files served by FastAPI
      const fileUrl = `/static/../data/sample_docs/${fname}`;
      // Fetch txt content from backend's static mount won't work directly,
      // so we load from a known path
      const r = await fetch(`${API}/static/${fname}`).catch(() => null);

      // fallback: use the sample data folder served via /static
      // Actually embed a small version directly for demo reliability
      const sampleContent = await fetchSampleContent(fname);
      if (!sampleContent) continue;

      const blob = new Blob([sampleContent], { type: "text/plain" });
      const file = new File([blob], fname, { type: "text/plain" });
      const fd = new FormData();
      fd.append("file", file);

      const up = await fetch(`${API}/upload`, { method: "POST", body: fd });
      if (up.ok) success++;
    } catch (e) {
      console.warn(`Failed to ingest ${fname}:`, e);
    }
  }

  btn.textContent = "🗃️ Load Sample Docs";
  btn.disabled = false;
  if (success > 0) {
    toast(`${success} sample document(s) loaded into Endee!`, "success");
  } else {
    toast("Could not load samples. Make sure the backend is running.", "error");
  }
}

async function fetchSampleContent(fname) {
  // Try fetching from backend static files
  try {
    const r = await fetch(`${API}/static/${fname}`, { signal: AbortSignal.timeout(5000) });
    if (r.ok) return await r.text();
  } catch {}
  return null;
}

// ── Chat Q&A ──────────────────────────────────────────────────

function setQuestion(q) {
  const input = document.getElementById("chat-input");
  input.value = q;
  autoResize(input);
  input.focus();
}

function chatKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

async function sendQuestion() {
  if (isAsking) return;
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question) return;

  const topK = parseInt(document.getElementById("top-k").value, 10);

  // Clear welcome screen on first message
  const welcome = document.querySelector(".chat-welcome");
  if (welcome) welcome.remove();

  input.value = "";
  autoResize(input);

  // Append user message
  appendMessage("user", "👤", question);

  // Append thinking indicator
  const thinkingId = `think-${Date.now()}`;
  appendThinking(thinkingId);

  isAsking = true;
  document.getElementById("chat-send").disabled = true;

  try {
    const r = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topK }),
    });

    removeThinking(thinkingId);

    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || r.statusText);
    }

    const data = await r.json();
    appendAnswerMessage(data);

  } catch (e) {
    removeThinking(thinkingId);
    appendMessage("assistant", "⚡", `❌ Error: ${e.message}`);
    toast(e.message, "error");
  } finally {
    isAsking = false;
    document.getElementById("chat-send").disabled = false;
  }
}

function appendMessage(role, avatar, text) {
  const msgs = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="msg-role">${role === "user" ? "You" : "AI Assistant"}</div>
      <div class="msg-text">${escHtml(text)}</div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function appendAnswerMessage(data) {
  const msgs = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = "msg assistant";

  const sourcesHtml = data.sources && data.sources.length > 0
    ? `<div class="msg-sources">
        <div class="msg-sources-title">📎 Sources (${data.sources.length})</div>
        ${data.sources.map(s => `
          <div class="source-chip">
            <div class="source-score">${(s.score * 100).toFixed(1)}%</div>
            <div class="source-info">
              <div class="source-file">📄 ${escHtml(s.source)}</div>
              <div class="source-preview">${escHtml(s.preview)}</div>
            </div>
          </div>`).join("")}
       </div>`
    : "";

  const modelBadge = data.model && data.model !== "none"
    ? `<span class="model-badge">🤖 ${escHtml(data.model)}</span>`
    : "";

  div.innerHTML = `
    <div class="msg-avatar">⚡</div>
    <div class="msg-body">
      <div class="msg-role">AI Assistant</div>
      <div class="msg-text">${escHtml(data.answer)}</div>
      ${modelBadge}
      ${sourcesHtml}
    </div>`;

  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function appendThinking(id) {
  const msgs = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = "msg assistant";
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">⚡</div>
    <div class="msg-body">
      <div class="msg-role">AI Assistant</div>
      <div class="thinking">
        Thinking
        <div class="thinking-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ── Semantic Search ───────────────────────────────────────────

async function runSearch() {
  const query = document.getElementById("search-input").value.trim();
  if (!query) return;

  const topK = parseInt(document.getElementById("top-k").value, 10);
  const container = document.getElementById("search-results");
  container.innerHTML = `<div class="empty-state"><span>⏳</span><p>Searching Endee vector index…</p></div>`;

  try {
    const r = await fetch(`${API}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();

    if (!data.results || data.results.length === 0) {
      container.innerHTML = `<div class="empty-state"><span>🤷</span><p>No results found.<br/>Upload documents first.</p></div>`;
      return;
    }

    container.innerHTML = data.results.map((res, i) => `
      <div class="result-card" style="animation-delay:${i * 0.05}s">
        <div class="result-header">
          <span class="result-source">📄 ${escHtml(res.source)}</span>
          <span class="result-score-badge">${(res.score * 100).toFixed(1)}% match</span>
        </div>
        <div class="result-text">${escHtml(res.text)}</div>
        <div class="result-meta">Chunk #${res.chunk_index} · Doc ID: ${res.doc_id}</div>
      </div>`).join("");

  } catch (e) {
    container.innerHTML = `<div class="empty-state"><span>❌</span><p>Error: ${escHtml(e.message)}</p></div>`;
    toast(e.message, "error");
  }
}

// ── Documents ─────────────────────────────────────────────────

async function loadDocuments() {
  const list = document.getElementById("docs-list");
  list.innerHTML = `<div class="empty-state"><span>⏳</span><p>Loading…</p></div>`;
  try {
    const r = await fetch(`${API}/documents`);
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();

    if (!data.documents || data.documents.length === 0) {
      list.innerHTML = `<div class="empty-state"><span>📂</span><p>No documents indexed yet.<br/>Upload a file or load sample docs.</p></div>`;
      return;
    }

    list.innerHTML = data.documents.map((doc, i) => `
      <div class="doc-card" style="animation-delay:${i * 0.05}s">
        <div class="doc-icon">📄</div>
        <div class="doc-info">
          <div class="doc-name">${escHtml(doc.source)}</div>
          <div class="doc-meta">ID: ${doc.doc_id}</div>
        </div>
        <span class="doc-badge">${doc.total_chunks} chunks</span>
        <button class="btn-danger" onclick="deleteDoc('${escHtml(doc.doc_id)}', this)">🗑️ Delete</button>
      </div>`).join("");

  } catch (e) {
    list.innerHTML = `<div class="empty-state"><span>❌</span><p>Error: ${escHtml(e.message)}</p></div>`;
    toast(e.message, "error");
  }
}

async function deleteDoc(docId, btn) {
  if (!confirm("Delete all chunks for this document?")) return;
  btn.disabled = true;
  btn.textContent = "…";
  try {
    const r = await fetch(`${API}/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();
    toast(`Deleted ${d.chunks_deleted} chunks.`, "success");
    loadDocuments();
  } catch (e) {
    toast(`Delete failed: ${e.message}`, "error");
    btn.disabled = false;
    btn.textContent = "🗑️ Delete";
  }
}

// ── Toast ─────────────────────────────────────────────────────

function toast(msg, type = "info") {
  const icons = { success: "✅", error: "❌", info: "ℹ️" };
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = `${icons[type] || ""} ${msg}`;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4500);
}

// ── Utils ─────────────────────────────────────────────────────

function escHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
