import { api, showToast } from "./api.js";
import { renderSource } from "./steps/source.js";
import { renderTrain, enableExport } from "./steps/train.js";
import { renderTest } from "./steps/test.js";

const state = {
  personaId: null,
  datasetId: null,
  jobId: null,
  uploads: [],
  personas: [],
  datasetReady: false,
  form: {
    personaName: "",
    systemPrompt: "",
    targetName: "",
    syntheticQa: false,
  },
};

const stepMounted = { source: false, train: false, test: false };
const stepPersonaId = { train: null, test: null };
let currentStep = "source";
let gpuInterval = null;

export function goToStep(step) {
  currentStep = step;
  document.querySelectorAll(".step-tab").forEach((t) => {
    const active = t.dataset.step === step;
    t.classList.toggle("active", active);
    t.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".step-panel").forEach((p) => {
    const active = p.id === `step-${step}`;
    p.classList.toggle("active", active);
    if (active) p.removeAttribute("hidden");
    else p.setAttribute("hidden", "");
  });
  updateStepIndicators();
  mountStep(step);
}

window.__pfGoToStep = goToStep;

async function init() {
  setupBridgeToggle();
  setupNewPersona();
  await loadGpu();
  await loadVersion();
  await loadPersonas();
  setupStepNav();
  mountStep("source");
  updateStepIndicators();
  gpuInterval = setInterval(loadGpu, 30000);
}

function setupBridgeToggle() {
  const btn = document.getElementById("toggle-bridge");
  const bridge = document.getElementById("voice-bridge");
  if (!btn || !bridge) return;
  btn.addEventListener("click", () => {
    const open = bridge.hasAttribute("hidden");
    if (open) bridge.removeAttribute("hidden");
    else bridge.setAttribute("hidden", "");
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    btn.classList.toggle("active", open);
  });
}

function setupNewPersona() {
  const btn = document.getElementById("new-persona-btn");
  if (!btn) return;
  btn.addEventListener("click", () => {
    state.personaId = null;
    state.datasetId = null;
    state.jobId = null;
    state.uploads = [];
    state.datasetReady = false;
    state.form = {
      personaName: "",
      systemPrompt: "",
      targetName: "",
      syntheticQa: false,
    };
    stepMounted.source = false;
    stepMounted.train = false;
    stepMounted.test = false;
    renderPersonaList();
    updateStepIndicators();
    goToStep("source");
    showToast("Ready for a new persona — upload samples in Source.", "info");
  });
}

async function loadVersion() {
  const chip = document.getElementById("version-chip");
  if (!chip) return;
  try {
    const health = await api.getHealth();
    if (health?.version) chip.textContent = `v${health.version}`;
  } catch (_) {
    /* keep static fallback */
  }
}

async function loadGpu() {
  const badge = document.getElementById("gpu-badge");
  if (!badge) return;
  try {
    const info = await api.getGpu();
    const text = badge.querySelector(".gpu-text");
    if (info.cuda_available) {
      badge.className = "gpu-badge ok";
      text.textContent = `GPU · ${info.device_name} · ${info.vram_free_gb}/${info.vram_total_gb} GB`;
      badge.title = info.warning || "CUDA available";
    } else {
      badge.className = "gpu-badge " + (info.warning ? "warn" : "error");
      text.textContent = info.warning || "No GPU detected";
      badge.title = "Training requires an NVIDIA GPU with CUDA";
    }
  } catch (_) {
    badge.className = "gpu-badge error";
    badge.querySelector(".gpu-text").textContent = "GPU check failed";
    badge.title = "Could not reach the server";
  }
}

async function loadPersonas() {
  try {
    state.personas = await api.listPersonas();
    renderPersonaList();
    updateStepIndicators();
  } catch (_) {
    state.personas = [];
  }
}

function statusPillClass(status) {
  if (status === "trained") return "trained";
  if (status === "training") return "training";
  if (status === "draft") return "ready";
  if (status === "failed") return "";
  return "";
}

function renderPersonaList() {
  const list = document.getElementById("persona-list");
  if (!state.personas.length) {
    list.innerHTML =
      '<li class="persona-empty">No personas yet. Upload writing samples in Source to begin.</li>';
    return;
  }
  list.innerHTML = state.personas
    .map(
      (p) => `
    <li class="persona-item ${p.id === state.personaId ? "active" : ""}" data-id="${p.id}" tabindex="0" role="button" aria-pressed="${p.id === state.personaId ? "true" : "false"}">
      <div class="persona-item-name">${escapeHtml(p.name)}</div>
      <div class="persona-item-meta">
        <span class="status-pill ${statusPillClass(p.status)}">${escapeHtml(p.status)}</span>
      </div>
    </li>`
    )
    .join("");

  list.querySelectorAll(".persona-item").forEach((el) => {
    const select = () => {
      state.personaId = el.dataset.id;
      const persona = state.personas.find((p) => p.id === state.personaId);
      if (persona?.dataset_id) {
        state.datasetId = persona.dataset_id;
        state.datasetReady = true;
      }
      renderPersonaList();
      updateStepIndicators();
      enableExport(state);
      if (currentStep === "train" || currentStep === "test") {
        remountStepIfNeeded(currentStep, true);
      }
    };
    el.addEventListener("click", select);
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        select();
      }
    });
  });
}

function updateStepIndicators() {
  const persona = state.personas.find((p) => p.id === state.personaId);
  const hasDataset = Boolean(state.datasetId || state.datasetReady || persona?.dataset_id);
  const isTrained = persona?.status === "trained";

  document.querySelectorAll(".step-tab").forEach((tab) => {
    const step = tab.dataset.step;
    let complete = false;
    if (step === "source") complete = hasDataset;
    if (step === "train") complete = isTrained;
    if (step === "test") complete = isTrained;
    tab.classList.toggle("complete", complete);
  });
}

function setupStepNav() {
  document.querySelectorAll(".step-tab").forEach((tab) => {
    tab.addEventListener("click", () => goToStep(tab.dataset.step));
    tab.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        goToStep(tab.dataset.step);
      }
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const order = ["source", "train", "test"];
        const idx = order.indexOf(tab.dataset.step);
        const next =
          e.key === "ArrowRight"
            ? order[(idx + 1) % order.length]
            : order[(idx - 1 + order.length) % order.length];
        goToStep(next);
        document.querySelector(`.step-tab[data-step="${next}"]`)?.focus();
      }
    });
  });
}

function isTrainingActive() {
  const cancelBtn = document.querySelector("#step-train #cancel-btn");
  return Boolean(cancelBtn && !cancelBtn.disabled);
}

function remountStepIfNeeded(step, force = false) {
  if (step === "source") return;
  if (step === "train" && isTrainingActive()) return;
  if (!force && stepMounted[step] && stepPersonaId[step] === state.personaId) return;
  stepMounted[step] = false;
  mountStep(step);
}

function mountStep(step) {
  const panel = document.getElementById(`step-${step}`);
  if (!panel) return;

  if (step === "source") {
    if (stepMounted.source) return;
    renderSource(panel, state, async (result) => {
      state.personaId = result.persona_id;
      state.datasetId = result.dataset_id;
      state.datasetReady = true;
      await loadPersonas();
      remountStepIfNeeded("train", true);
      remountStepIfNeeded("test", true);
      updateStepIndicators();
    });
    stepMounted.source = true;
    return;
  }

  if (stepMounted[step] && stepPersonaId[step] === state.personaId) return;
  if (step === "train" && isTrainingActive()) return;

  panel.innerHTML = "";

  if (step === "train") {
    renderTrain(panel, state, async () => {
      await loadPersonas();
      enableExport(state);
      updateStepIndicators();
    });
  } else if (step === "test") {
    renderTest(panel, state);
  }

  stepMounted[step] = true;
  stepPersonaId[step] = state.personaId;
}

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

init();
