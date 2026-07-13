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
};

let currentStep = "source";

async function init() {
  await loadGpu();
  await loadPersonas();
  setupStepNav();
  renderCurrentStep();
}

async function loadGpu() {
  const badge = document.getElementById("gpu-badge");
  try {
    const info = await api.getGpu();
    const text = badge.querySelector(".gpu-text");
    if (info.cuda_available) {
      badge.className = "gpu-badge ok";
      text.textContent = `GPU: ${info.device_name} · ${info.vram_free_gb}/${info.vram_total_gb} GB`;
    } else {
      badge.className = "gpu-badge " + (info.warning ? "warn" : "error");
      text.textContent = info.warning || "No GPU detected";
    }
  } catch (_) {
    badge.className = "gpu-badge error";
    badge.querySelector(".gpu-text").textContent = "GPU check failed";
  }
}

async function loadPersonas() {
  try {
    state.personas = await api.listPersonas();
    renderPersonaList();
  } catch (_) {
    state.personas = [];
  }
}

function renderPersonaList() {
  const list = document.getElementById("persona-list");
  if (!state.personas.length) {
    list.innerHTML = '<li class="persona-empty">No personas yet</li>';
    return;
  }
  list.innerHTML = state.personas
    .map(
      (p) => `
    <li class="persona-item ${p.id === state.personaId ? "active" : ""}" data-id="${p.id}">
      <div class="persona-item-name">${escapeHtml(p.name)}</div>
      <div class="persona-item-status">${p.status}</div>
    </li>`
    )
    .join("");

  list.querySelectorAll(".persona-item").forEach((el) => {
    el.addEventListener("click", () => {
      state.personaId = el.dataset.id;
      const persona = state.personas.find((p) => p.id === state.personaId);
      if (persona?.dataset_id) state.datasetId = persona.dataset_id;
      renderPersonaList();
      renderCurrentStep();
      if (persona?.status === "trained") enableExport(state);
    });
  });
}

function setupStepNav() {
  document.querySelectorAll(".step-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      currentStep = tab.dataset.step;
      document.querySelectorAll(".step-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      document.querySelectorAll(".step-panel").forEach((p) => p.classList.remove("active"));
      document.getElementById(`step-${currentStep}`).classList.add("active");
      renderCurrentStep();
    });
  });
}

function renderCurrentStep() {
  const panels = {
    source: document.getElementById("step-source"),
    train: document.getElementById("step-train"),
    test: document.getElementById("step-test"),
  };

  if (currentStep === "source") {
    renderSource(panels.source, state, async (result) => {
      state.personaId = result.persona_id;
      state.datasetId = result.dataset_id;
      await loadPersonas();
    });
  } else if (currentStep === "train") {
    renderTrain(panels.train, state, async () => {
      await loadPersonas();
      enableExport(state);
    });
  } else if (currentStep === "test") {
    renderTest(panels.test, state);
  }
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

init();
