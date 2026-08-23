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
  exportGguf: (personaId) =>
    request(`/api/personas/${personaId}/export-gguf`, { method: "POST" }),

  getVoiceBackends: () => request("/api/voice/backends"),
  uploadVoiceReference: async (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/voice/reference", { method: "POST", body: form });
  },
  synthesizeVoice: async (body) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120000);
    let res;
    try {
      res = await fetch(`${API_BASE}/api/voice/synthesize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (e) {
      if (e.name === "AbortError") throw new Error("Voice synthesis timed out after 2 minutes.");
      throw e;
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) {
      let detail = res.statusText;
      try {
        detail = parseErrorBody(await res.json());
      } catch (_) {}
      throw new Error(detail);
    }
    return res.blob();
  },
  startVoiceLiveSession: (lockAfterSeconds = 12) =>
    request("/api/voice/live/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lock_after_seconds: lockAfterSeconds }),
    }),
  getVoiceLiveStatus: (sessionId) => request(`/api/voice/live/${sessionId}/status`),
  lockVoiceLiveSession: (sessionId) =>
    request(`/api/voice/live/${sessionId}/lock`, { method: "POST" }),
  dropVoiceLiveSession: (sessionId) =>
    request(`/api/voice/live/${sessionId}`, { method: "DELETE" }),
  voiceLiveStreamUrl: (sessionId) => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/api/voice/live/${sessionId}/stream`;
  },
};

export function showToast(message, type = "info") {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = message;
  el.className = `toast show ${type}`;
  clearTimeout(el._timer);
  const duration = type === "error" ? 6500 : 4200;
  el._timer = setTimeout(() => {
    el.classList.remove("show");
  }, duration);
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
