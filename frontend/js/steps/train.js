import { api, showToast } from "../api.js";

let lossChart = null;
let lossPoints = [];

export function renderTrain(container, state, onTrained) {
  container.innerHTML = `
    <h2 class="step-heading">Train</h2>
    <p class="step-desc">Fine-tune a LoRA adapter on your dataset. Requires an NVIDIA GPU.</p>

    <div class="form-row" style="margin-bottom:1rem">
      <label for="base-model">Base model</label>
      <select id="base-model">
        <option value="unsloth/Llama-3.2-3B-Instruct-bnb-4bit">Llama 3.2 3B (recommended)</option>
        <option value="unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit">Mistral 7B</option>
        <option value="unsloth/Qwen2.5-7B-Instruct-bnb-4bit">Qwen 2.5 7B</option>
      </select>
    </div>

    <div class="hyper-grid">
      <div class="slider-row form-row">
        <label>Epochs <span id="epochs-val">3</span></label>
        <input type="range" id="epochs" min="1" max="10" value="3" />
      </div>
      <div class="slider-row form-row">
        <label>Learning rate <span id="lr-val">2e-4</span></label>
        <input type="range" id="lr" min="1" max="5" value="2" />
      </div>
      <div class="slider-row form-row">
        <label>Batch size <span id="batch-val">2</span></label>
        <input type="range" id="batch" min="1" max="8" value="2" />
      </div>
      <div class="slider-row form-row">
        <label>LoRA rank <span id="lora-val">16</span></label>
        <input type="range" id="lora-r" min="4" max="64" step="4" value="16" />
      </div>
    </div>

    <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    <div class="loss-chart"><canvas id="loss-canvas" width="600" height="160"></canvas></div>
    <div class="log-panel" id="log-panel"><div class="log-line">Ready to train.</div></div>

    <div class="btn-row">
      <button type="button" class="btn btn-primary" id="start-btn" ${state.personaId ? "" : "disabled"}>Start training</button>
      <button type="button" class="btn btn-danger" id="cancel-btn" disabled>Cancel</button>
      <button type="button" class="btn btn-secondary" id="export-btn" disabled>Export for Ollama</button>
    </div>
  `;

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
  const startBtn = container.querySelector("#start-btn");
  const cancelBtn = container.querySelector("#cancel-btn");
  const exportBtn = container.querySelector("#export-btn");
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
    }
    if (progress.loss != null) {
      lossPoints.push({ step: progress.step, loss: progress.loss });
      drawLossChart();
      addLog(`Step ${progress.step} — loss ${progress.loss}`, "loss");
    } else if (progress.message) {
      addLog(progress.message);
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
    logPanel.innerHTML = "";
    addLog("Starting training…");
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
    try {
      await api.cancelTraining(state.jobId);
      addLog("Cancellation requested…");
    } catch (e) {
      showToast(e.message, "error");
    }
  });

  exportBtn.addEventListener("click", async () => {
    if (!state.personaId) return;
    try {
      const result = await api.exportGguf(state.personaId);
      showToast(result.message, "success");
      addLog(`Export: ${result.message}`);
      if (result.manual_steps?.length) {
        result.manual_steps.forEach((s) => addLog(s));
      }
    } catch (e) {
      showToast(e.message, "error");
    }
  });

  function finishTraining(progress) {
    document.getElementById("voice-bridge")?.classList.remove("training");
    cancelBtn.disabled = true;
    startBtn.disabled = false;

    if (progress.status === "completed") {
      addLog("Training complete!", "loss");
      exportBtn.disabled = false;
      showToast("Training complete!", "success");
      if (onTrained) onTrained();
    } else if (progress.status === "failed") {
      addLog(progress.error || "Training failed", "error");
      showToast("Training failed", "error");
    } else {
      addLog("Training cancelled.");
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
  if (btn) btn.disabled = !state.personaId;
}
