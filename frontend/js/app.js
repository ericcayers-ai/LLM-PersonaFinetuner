import { api, showToast } from "./api.js";
import { renderSource } from "./steps/source.js";
import { renderTrain, enableExport } from "./steps/train.js";
import { renderTest } from "./steps/test.js";
import { renderVoice } from "./steps/voice.js";
import { renderCall } from "./steps/call.js";

const state = {
  personaId: null,
  datasetId: null,
  jobId: null,
  uploads: [],
  personas: [],
  form: {
    personaName: "",
    systemPrompt: "",
    targetName: "",
    syntheticQa: false,
  },
};

const stepMounted = { source: false, train: false, test: false, voice: false, call: false };
const stepDisposers = {};
let currentStep = "source";
let gpuInterval = null;

async function init() {
  await loadGpu();
  await loadPersonas();
  setupStepNav();
  mountStep("source");
  gpuInterval = setInterval(loadGpu, 30000);
}

async function loadGpu() {
  const badge = document.getElementById("gpu-badge");
  if (!badge) return;
  try {
    const info = await api.getGpu();
    const text = badge.querySelector(".gpu-text");
    if (info.cuda_available) {
      badge.className = "gpu-badge ok";
      text.textContent = `GPU: ${info.device_name} · ${info.vram_free_gb}/${info.vram_total_gb} GB`;
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
  } catch (_) {
    state.personas = [];
  }
}

function renderPersonaList() {
  const list = document.getElementById("persona-list");
  if (!state.personas.length) {
    list.innerHTML = '<li class="persona-empty">No personas yet — upload samples in Source to begin.</li>';
    return;
  }
  list.innerHTML = state.personas
    .map(
      (p) => `
    <li class="persona-item ${p.id === state.personaId ? "active" : ""}" data-id="${p.id}" tabindex="0" role="button">
      <div class="persona-item-name">${escapeHtml(p.name)}</div>
      <div class="persona-item-status">${p.status}</div>
    </li>`
    )
    .join("");

  list.querySelectorAll(".persona-item").forEach((el) => {
    const select = () => {
      state.personaId = el.dataset.id;
      const persona = state.personas.find((p) => p.id === state.personaId);
      if (persona?.dataset_id) state.datasetId = persona.dataset_id;
      renderPersonaList();
      if (persona?.status === "trained") enableExport(state);
      ["train", "test", "voice", "call"].forEach(unmountStep);
      mountStep(currentStep);
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

function setupStepNav() {
  document.querySelectorAll(".step-tab").forEach((tab) => {
    tab.addEventListener("click", () => goToStep(tab.dataset.step));
    tab.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        goToStep(tab.dataset.step);
      }
    });
  });
}

function goToStep(step) {
  currentStep = step;
  document.dispatchEvent(new CustomEvent("personafinetuner:stepchange", { detail: step }));
  document.querySelectorAll(".step-tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.step === step);
    t.setAttribute("aria-selected", t.dataset.step === step ? "true" : "false");
  });
  document.querySelectorAll(".step-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `step-${step}`);
  });
  mountStep(step);
}

function mountStep(step) {
  const panel = document.getElementById(`step-${step}`);
  if (stepMounted[step]) return;

  if (step === "source") {
    renderSource(panel, state, async (result) => {
      state.personaId = result.persona_id;
      state.datasetId = result.dataset_id;
      await loadPersonas();
    });
  } else if (step === "train") {
    renderTrain(panel, state, async () => {
      await loadPersonas();
      enableExport(state);
    });
  } else if (step === "test") {
    renderTest(panel, state);
  } else if (step === "voice") {
    stepDisposers.voice = renderVoice(panel, state, async () => {
      await loadPersonas();
      unmountStep("call");
    });
  } else if (step === "call") {
    stepDisposers.call = renderCall(panel, state);
  }
  stepMounted[step] = true;
}

function unmountStep(step) {
  stepDisposers[step]?.();
  delete stepDisposers[step];
  stepMounted[step] = false;
  const panel = document.getElementById(`step-${step}`);
  if (panel) panel.innerHTML = "";
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

init();
