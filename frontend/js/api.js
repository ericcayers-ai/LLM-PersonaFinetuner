const API_BASE = "";

function parseErrorBody(body) {
  if (typeof body === "string") return body;
  if (body.message) {
    return body.hint ? `${body.message} ${body.hint}` : body.message;
  }
  if (body.detail) {
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  }
  return JSON.stringify(body);
}

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = parseErrorBody(body);
    } catch (_) {}
    throw new Error(detail);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  if (ct.startsWith("audio/")) return res.blob();
  if (res.status === 204) return null;
  return res.text();
}

export const api = {
  getGpu: () => request("/api/system/gpu"),
  getHealth: () => request("/api/system/health"),
  listPersonas: () => request("/api/personas"),
  getPersona: (id) => request(`/api/personas/${id}`),
  uploadDataset: async (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/datasets/upload", { method: "POST", body: form });
  },
  buildDataset: (body) =>
    request("/api/datasets/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  startTraining: (config) =>
    request("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  cancelTraining: (jobId) =>
    request(`/api/training/${jobId}/cancel`, { method: "POST" }),
  getLossHistory: (jobId) => request(`/api/training/${jobId}/loss`),
  chat: (body) =>
    request("/api/inference/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  getVoiceCapabilities: () => request("/api/voice/capabilities"),
  getVoiceProfile: (personaId) => request(`/api/voice/${personaId}/profile`),
  uploadVoiceProfile: (personaId, file, consentConfirmed, language = "en") => {
    const form = new FormData();
    form.append("file", file);
    form.append("consent_confirmed", String(consentConfirmed));
    form.append("language", language);
    return request(`/api/voice/${personaId}/profile`, { method: "POST", body: form });
  },
  deleteVoiceProfile: (personaId) =>
    request(`/api/voice/${personaId}/profile`, { method: "DELETE" }),
  transcribe: (file, language = "") => {
    const form = new FormData();
    form.append("file", file);
    if (language) form.append("language", language);
    return request("/api/voice/transcribe", { method: "POST", body: form });
  },
  synthesize: (personaId, text, language = "en") =>
    request(`/api/voice/${personaId}/synthesize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language }),
    }),
  exportGguf: (personaId) =>
    request(`/api/personas/${personaId}/export-gguf`, { method: "POST" }),
};

export function showToast(message, type = "info") {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = message;
  el.className = `toast show ${type}`;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => {
    el.classList.remove("show");
  }, 4500);
}

export function setLoading(btn, loading, label = "Loading…") {
  if (!btn) return;
  if (loading) {
    btn.dataset.prevText = btn.textContent;
    btn.disabled = true;
    btn.classList.add("loading");
    btn.textContent = label;
  } else {
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = btn.dataset.prevText || btn.textContent;
  }
}
