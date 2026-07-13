import { api, showToast } from "../api.js";

export function renderSource(container, state, onBuilt) {
  container.innerHTML = `
    <h2 class="step-heading">Source</h2>
    <p class="step-desc">Upload writing samples or chat exports. Supported: .txt, .jsonl, Discord/WhatsApp/Instagram exports.</p>

    <div class="drop-zone" id="drop-zone">
      <div class="drop-zone-icon">↑</div>
      <p>Drop files here or click to browse</p>
      <p class="drop-zone-hint">.txt · .jsonl · .json (Discord, Instagram)</p>
    </div>
    <input type="file" id="file-input" accept=".txt,.jsonl,.json" multiple hidden />

    <ul class="upload-list" id="upload-list"></ul>
    <div id="format-hint" class="format-hint" hidden></div>

    <div class="form-grid">
      <div class="form-row">
        <label for="persona-name">Persona name</label>
        <input type="text" id="persona-name" placeholder="e.g. Alex" />
      </div>
      <div class="form-row">
        <label for="system-prompt">System prompt</label>
        <textarea id="system-prompt" placeholder="You are Alex, a thoughtful writer who…"></textarea>
      </div>
      <div class="form-row" id="target-name-row" hidden>
        <label for="target-name">Target username / contact name</label>
        <input type="text" id="target-name" placeholder="Exact name from the chat export" />
      </div>
      <div class="checkbox-row">
        <input type="checkbox" id="synthetic-qa" />
        <label for="synthetic-qa">Generate synthetic Q&amp;A pairs (optional, uses GPU if available)</label>
      </div>
    </div>

    <div class="btn-row">
      <button type="button" class="btn btn-primary" id="build-btn" disabled>Build dataset</button>
    </div>

    <div id="preview-section" class="preview-section" hidden></div>
  `;

  const dropZone = container.querySelector("#drop-zone");
  const fileInput = container.querySelector("#file-input");
  const uploadList = container.querySelector("#upload-list");
  const formatHint = container.querySelector("#format-hint");
  const targetNameRow = container.querySelector("#target-name-row");
  const buildBtn = container.querySelector("#build-btn");

  function updateUploadUI() {
    uploadList.innerHTML = state.uploads
      .map(
        (u) => `
      <li class="upload-item">
        <span>${u.filename}</span>
        <span class="upload-format">${u.format_label || u.detected_format}</span>
      </li>`
      )
      .join("");

    const needsTarget = state.uploads.some((u) => u.requires_target_name);
    targetNameRow.hidden = !needsTarget;

    if (state.uploads.length > 0) {
      const last = state.uploads[state.uploads.length - 1];
      formatHint.hidden = false;
      formatHint.textContent = last.sample_hint || "";
    } else {
      formatHint.hidden = true;
    }

    buildBtn.disabled = state.uploads.length === 0;
  }

  async function handleFiles(files) {
    for (const file of files) {
      try {
        const result = await api.uploadDataset(file);
        state.uploads.push(result);
        showToast(`Uploaded ${file.name} (${result.format_label})`, "success");
      } catch (e) {
        showToast(`Upload failed: ${e.message}`, "error");
      }
    }
    updateUploadUI();
  }

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    handleFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener("change", () => {
    handleFiles(fileInput.files);
    fileInput.value = "";
  });

  buildBtn.addEventListener("click", async () => {
    const personaName = container.querySelector("#persona-name").value.trim();
    const systemPrompt = container.querySelector("#system-prompt").value.trim();
    const targetName = container.querySelector("#target-name").value.trim();
    const syntheticQa = container.querySelector("#synthetic-qa").checked;

    if (!personaName || !systemPrompt) {
      showToast("Persona name and system prompt are required.", "error");
      return;
    }

    const needsTarget = state.uploads.some((u) => u.requires_target_name);
    if (needsTarget && !targetName) {
      showToast("Target username/contact name is required for chat exports.", "error");
      return;
    }

    buildBtn.disabled = true;
    buildBtn.textContent = "Building…";

    try {
      const result = await api.buildDataset({
        upload_ids: state.uploads.map((u) => u.upload_id),
        persona_name: personaName,
        system_prompt: systemPrompt,
        target_name: targetName || null,
        generate_synthetic_qa: syntheticQa,
      });

      state.personaId = result.persona_id;
      state.datasetId = result.dataset_id;
      renderPreview(container.querySelector("#preview-section"), result);
      showToast(`Dataset built: ${result.total_examples} examples`, "success");
      if (onBuilt) onBuilt(result);
    } catch (e) {
      showToast(`Build failed: ${e.message}`, "error");
    } finally {
      buildBtn.disabled = false;
      buildBtn.textContent = "Build dataset";
    }
  });

  updateUploadUI();
}

function renderPreview(section, result) {
  section.hidden = false;
  const warnings =
    result.warnings?.length > 0
      ? `<ul class="warning-list">${result.warnings.map((w) => `<li>${w}</li>`).join("")}</ul>`
      : "";

  const rows = result.examples
    .map((ex) => {
      const cells = ex.messages
        .map(
          (m) =>
            `<td class="role-${m.role}"><strong>${m.role}</strong><br>${escapeHtml(m.content.slice(0, 200))}${m.content.length > 200 ? "…" : ""}</td>`
        )
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  section.innerHTML = `
    <h3 class="step-heading" style="font-size:1.1rem">Dataset preview</h3>
    <div class="preview-stats">
      <span class="stat-badge">Total: <strong>${result.total_examples}</strong></span>
      <span class="stat-badge">Train: <strong>${result.train_examples}</strong></span>
      <span class="stat-badge">Val: <strong>${result.val_examples}</strong></span>
    </div>
    ${warnings}
    <table class="preview-table">
      <tbody>${rows}</tbody>
    </table>
  `;
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
