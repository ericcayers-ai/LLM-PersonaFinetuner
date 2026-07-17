import { api, showToast, setLoading } from "../api.js";

let lossChart = null;
let lossPoints = [];

export function renderTrain(container, state, onTrained) {
  const hasPersona = Boolean(state.personaId);
  const persona = state.personas?.find((p) => p.id === state.personaId);
  const isTrained = persona?.status === "trained";

  container.innerHTML = `
    <h2 class="step-heading">Train</h2>
    <p class="step-desc">Fine-tune a LoRA adapter on your dataset. Requires an NVIDIA GPU with CUDA.</p>

    ${
      hasPersona
        ? ""
        : `<div class="empty-state">
            <strong>No dataset selected</strong>
            Build a dataset in Source first, then return here to train.
            <div class="btn-row mt-3">
              <button type="button" class="btn btn-secondary btn-sm" id="goto-source">Go to Source</button>
            </div>
          </div>`
    }

    <div class="section-card train-essentials">
      <h3 class="section-card-title">Model</h3>
      <div class="form-row train-model-row">
        <label for="base-model">Base model</label>
        <select id="base-model" ${hasPersona ? "" : "disabled"}>
          <option value="unsloth/Llama-3.2-3B-Instruct-bnb-4bit">Llama 3.2 3B (recommended)</option>
          <option value="unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit">Mistral 7B</option>
          <option value="unsloth/Qwen2.5-7B-Instruct-bnb-4bit">Qwen 2.5 7B</option>
        </select>
        <p class="field-hint">3B fits most 6–8 GB GPUs. 7B models need more VRAM.</p>
      </div>
    </div>

    <div class="advanced-block" id="advanced-block">
      <button type="button" class="advanced-toggle" id="advanced-toggle" aria-expanded="false" aria-controls="advanced-body">
        <span>Advanced settings</span>
        <span class="advanced-toggle-hint">epochs, LR, batch, LoRA</span>
        <span class="advanced-chevron" aria-hidden="true">▾</span>
      </button>
      <div class="advanced-body" id="advanced-body">
        <div class="hyper-grid">
          <div class="slider-row form-row">
            <label for="epochs">Epochs <span id="epochs-val">3</span></label>
            <input type="range" id="epochs" min="1" max="10" value="3" ${hasPersona ? "" : "disabled"} />
          </div>
          <div class="slider-row form-row">
            <label for="lr">Learning rate <span id="lr-val">2e-4</span></label>
            <input type="range" id="lr" min="1" max="5" value="2" ${hasPersona ? "" : "disabled"} />
          </div>
          <div class="slider-row form-row">
            <label for="batch">Batch size <span id="batch-val">2</span></label>
            <input type="range" id="batch" min="1" max="8" value="2" ${hasPersona ? "" : "disabled"} />
          </div>
          <div class="slider-row form-row">
            <label for="lora-r">LoRA rank <span id="lora-val">16</span></label>
            <input type="range" id="lora-r" min="4" max="64" step="4" value="16" ${hasPersona ? "" : "disabled"} />
          </div>
        </div>
      </div>
    </div>

    <div class="btn-row">
      <button type="button" class="btn btn-primary" id="start-btn" ${hasPersona ? "" : "disabled"}>Start training</button>
      <button type="button" class="btn btn-danger" id="cancel-btn" disabled>Cancel</button>
      <span class="spacer"></span>
      <button type="button" class="btn btn-secondary" id="export-btn" ${isTrained ? "" : "disabled"}>Export for Ollama</button>
    </div>

    <div class="train-status-panel section-card">
      <h3 class="section-card-title">Progress</h3>
      <div class="progress-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="progress-bar">
        <div class="progress-fill" id="progress-fill"></div>
      </div>
      <p class="progress-label" id="progress-label">${hasPersona ? "Ready to train." : "Waiting for a dataset…"}</p>
      <div class="loss-chart" id="loss-chart" hidden>
        <canvas id="loss-canvas" width="600" height="160" aria-label="Training loss chart"></canvas>
      </div>
      <div class="log-panel" id="log-panel" tabindex="0" aria-label="Training log">
        <div class="log-line">${hasPersona ? "Ready to train." : "Build a dataset in Source to unlock training."}</div>
      </div>
    </div>

    ${
      isTrained
        ? `<div class="callout callout-success mt-3">
            <strong>This persona is trained</strong>
            Chat with it in Test, or export the adapter for Ollama.
            <div class="btn-row mt-3">
              <button type="button" class="btn btn-primary btn-sm" id="goto-test">Continue to Test →</button>
            </div>
          </div>`
        : ""
    }
  `;

  container.querySelector("#goto-source")?.addEventListener("click", () => {
    window.__pfGoToStep?.("source");
  });
  container.querySelector("#goto-test")?.addEventListener("click", () => {
    window.__pfGoToStep?.("test");
  });

  const advancedBlock = container.querySelector("#advanced-block");
  const advancedToggle = container.querySelector("#advanced-toggle");
  advancedToggle.addEventListener("click", () => {
    const open = advancedBlock.classList.toggle("open");
    advancedToggle.setAttribute("aria-expanded", open ? "true" : "false");
  });

  const lrMap = { 1: 5e-5, 2: 2e-4, 3: 5e-4, 4: 1e-3, 5: 2e-3 };
  const epochsEl = container.querySelector("#epochs");
  const lrEl = container.querySelector("#lr");
  const batchEl = container.querySelector("#batch");
  const loraEl = container.querySelector("#lora-r");

  epochsEl.addEventListener("input", () => {
    container.querySelector("#epochs-val").textContent = epochsEl.value;
  });
  lrEl.addEventListener("input", () => {
    container.querySelector("#lr-val").textContent = lrMap[lrEl.value];
  });
  batchEl.addEventListener("input", () => {
    container.querySelector("#batch-val").textContent = batchEl.value;
  });
  loraEl.addEventListener("input", () => {
    container.querySelector("#lora-val").textContent = loraEl.value;
  });

  const logPanel = container.querySelector("#log-panel");
  const progressFill = container.querySelector("#progress-fill");
  const progressBar = container.querySelector("#progress-bar");
  const progressLabel = container.querySelector("#progress-label");
  const startBtn = container.querySelector("#start-btn");
  const cancelBtn = container.querySelector("#cancel-btn");
  const exportBtn = container.querySelector("#export-btn");
  const lossChartEl = container.querySelector("#loss-chart");
  const canvas = container.querySelector("#loss-canvas");
  lossChart = canvas.getContext("2d");
  lossPoints = [];

  function addLog(msg, cls = "") {
    const line = document.createElement("div");
    line.className = `log-line ${cls}`;
    line.textContent = msg;
    logPanel.appendChild(line);
    logPanel.scrollTop = logPanel.scrollHeight;
  }

  function drawLossChart() {
    const ctx = lossChart;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (lossPoints.length < 2) return;

    const losses = lossPoints.map((p) => p.loss);
    const minL = Math.min(...losses);
    const maxL = Math.max(...losses);
    const range = maxL - minL || 1;
    const pad = 20;

    ctx.strokeStyle = "#62d9e0";
    ctx.lineWidth = 2;
    ctx.beginPath();
    lossPoints.forEach((p, i) => {
      const x = pad + (i / (lossPoints.length - 1)) * (w - pad * 2);
      const y = h - pad - ((p.loss - minL) / range) * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  function updateProgress(progress) {
    if (progress.total_steps > 0) {
      const pct = (progress.step / progress.total_steps) * 100;
      progressFill.style.width = `${Math.min(pct, 100)}%`;
      progressBar.setAttribute("aria-valuenow", String(Math.round(Math.min(pct, 100))));
      progressLabel.textContent = `Epoch ${progress.epoch}/${progress.total_epochs || "?"} · Step ${progress.step}/${progress.total_steps}`;
    }
    if (progress.eta_seconds != null && progress.eta_seconds > 0) {
      const mins = Math.ceil(progress.eta_seconds / 60);
      progressLabel.textContent += ` · ~${mins} min left`;
    }
    if (progress.loss != null) {
      lossChartEl.hidden = false;
      lossPoints.push({ step: progress.step, loss: progress.loss });
      drawLossChart();
      addLog(`Step ${progress.step} — loss ${progress.loss.toFixed(4)}`, "loss");
    } else if (progress.message) {
      addLog(progress.message);
      progressLabel.textContent = progress.message;
    }
  }

  startBtn.addEventListener("click", async () => {
    if (!state.personaId) {
      showToast("Build a dataset first in the Source step.", "error");
      return;
    }

    startBtn.disabled = true;
    cancelBtn.disabled = false;
    lossPoints = [];
    lossChartEl.hidden = true;
    logPanel.innerHTML = "";
    addLog("Starting training…");
    progressFill.style.width = "0%";
    progressBar.setAttribute("aria-valuenow", "0");
    document.getElementById("voice-bridge")?.classList.add("training");

    try {
      const result = await api.startTraining({
        persona_id: state.personaId,
        base_model: container.querySelector("#base-model").value,
        epochs: parseInt(epochsEl.value, 10),
        learning_rate: lrMap[lrEl.value],
        batch_size: parseInt(batchEl.value, 10),
        lora_r: parseInt(loraEl.value, 10),
        lora_alpha: parseInt(loraEl.value, 10) * 2,
      });

      state.jobId = result.job_id;
      subscribeSSE(result.job_id, (progress) => {
        updateProgress(progress);
        if (["completed", "failed", "cancelled"].includes(progress.status)) {
          finishTraining(progress);
        }
      });
    } catch (e) {
      addLog(`Error: ${e.message}`, "error");
      showToast(e.message, "error");
      startBtn.disabled = false;
      cancelBtn.disabled = true;
      document.getElementById("voice-bridge")?.classList.remove("training");
    }
  });

  cancelBtn.addEventListener("click", async () => {
    if (!state.jobId) return;
    if (!window.confirm("Cancel training? Progress for this run will be lost.")) return;
    try {
      await api.cancelTraining(state.jobId);
      addLog("Cancellation requested…");
    } catch (e) {
      showToast(e.message, "error");
    }
  });

  exportBtn.addEventListener("click", async () => {
    if (!state.personaId) return;
    setLoading(exportBtn, true, "Exporting…");
    try {
      const result = await api.exportGguf(state.personaId);
      showToast(result.message, "success");
      addLog(`Export: ${result.message}`);
      if (result.manual_steps?.length) {
        result.manual_steps.forEach((s) => addLog(s));
      }
    } catch (e) {
      showToast(e.message, "error");
    } finally {
      setLoading(exportBtn, false);
      exportBtn.textContent = "Export for Ollama";
      enableExport(state);
    }
  });

  function finishTraining(progress) {
    document.getElementById("voice-bridge")?.classList.remove("training");
    cancelBtn.disabled = true;
    startBtn.disabled = false;

    if (progress.status === "completed") {
      addLog("Training complete!", "loss");
      progressLabel.textContent = "Training complete";
      progressFill.style.width = "100%";
      progressBar.setAttribute("aria-valuenow", "100");
      exportBtn.disabled = false;
      showToast("Training complete — try chatting in Test.", "success");
      if (onTrained) onTrained();
    } else if (progress.status === "failed") {
      addLog(progress.error || "Training failed", "error");
      progressLabel.textContent = "Training failed";
      showToast(progress.error || "Training failed", "error");
    } else {
      addLog("Training cancelled.");
      progressLabel.textContent = "Training cancelled";
    }
  }
}

function subscribeSSE(jobId, onProgress) {
  const source = new EventSource(`/api/training/${jobId}/stream`);
  source.addEventListener("progress", (e) => {
    const data = JSON.parse(e.data);
    onProgress(data);
    if (["completed", "failed", "cancelled"].includes(data.status)) {
      source.close();
    }
  });
  source.onerror = () => source.close();
}

export function enableExport(state) {
  const btn = document.querySelector("#export-btn");
  if (!btn) return;
  const persona = state.personas?.find((p) => p.id === state.personaId);
  btn.disabled = !(state.personaId && persona?.status === "trained");
}
