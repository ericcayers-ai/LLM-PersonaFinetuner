const API_BASE = "";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
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
};

export function showToast(message, type = "info") {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = message;
  el.className = `toast show ${type}`;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => {
    el.classList.remove("show");
  }, 4000);
}
