import { api, showToast, setLoading } from "../api.js";

export function renderSource(container, state, onBuilt) {
  container.innerHTML = `
    <h2 class="step-heading">Source</h2>
    <p class="step-desc">Upload writing samples or chat exports. Supported: .txt, .jsonl, Discord, WhatsApp, and Instagram exports.</p>

    <div class="drop-zone" id="drop-zone" tabindex="0" role="button" aria-label="Upload files">
      <div class="drop-zone-icon" aria-hidden="true">↑</div>
      <p>Drop files here or click to browse</p>
      <p class="drop-zone-hint">.txt · .jsonl · .json (Discord, Instagram)</p>
    </div>
    <input type="file" id="file-input" accept=".txt,.jsonl,.json" multiple hidden />

    <ul class="upload-list" id="upload-list"></ul>
    <div id="format-hint" class="format-hint" hidden></div>

    <div class="form-grid">
      <div class="form-row">
        <label for="persona-name">Persona name</label>
        <input type="text" id="persona-name" placeholder="e.g. Alex" value="${escapeAttr(state.form.personaName)}" />
      </div>
      <div class="form-row">
        <label for="system-prompt">System prompt</label>
        <textarea id="system-prompt" placeholder="You are Alex, a thoughtful writer who…">${escapeHtml(state.form.systemPrompt)}</textarea>
      </div>
      <div class="form-row" id="target-name-row" hidden>
        <label for="target-name">Target username / contact name</label>
        <input type="text" id="target-name" placeholder="Exact name from the chat export" value="${escapeAttr(state.form.targetName)}" />
      </div>
      <div class="checkbox-row">
        <input type="checkbox" id="synthetic-qa" ${state.form.syntheticQa ? "checked" : ""} />
        <label for="synthetic-qa">Generate synthetic Q&amp;A pairs (optional)</label>
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

  container.querySelector("#persona-name").addEventListener("input", (e) => {
    state.form.personaName = e.target.value;
  });
  container.querySelector("#system-prompt").addEventListener("input", (e) => {
    state.form.systemPrompt = e.target.value;
  });
  container.querySelector("#target-name").addEventListener("input", (e) => {
    state.form.targetName = e.target.value;
  });
  container.querySelector("#synthetic-qa").addEventListener("change", (e) => {
    state.form.syntheticQa = e.target.checked;
  });

  function updateUploadUI() {
    if (!state.uploads.length) {
      uploadList.innerHTML = '<li class="upload-empty">No files uploaded yet.</li>';
    } else {
      uploadList.innerHTML = state.uploads
        .map(
          (u, i) => `
        <li class="upload-item">
          <span class="upload-name">${escapeHtml(u.filename)}</span>
          <span class="upload-format">${escapeHtml(u.format_label || u.detected_format)}</span>
          <button type="button" class="btn-icon remove-upload" data-index="${i}" aria-label="Remove ${escapeAttr(u.filename)}">×</button>
        </li>`
        )
        .join("");
      uploadList.querySelectorAll(".remove-upload").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.uploads.splice(Number(btn.dataset.index), 1);
          updateUploadUI();
        });
      });
    }

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
    dropZone.classList.add("uploading");
    for (const file of files) {
      try {
        const result = await api.uploadDataset(file);
        state.uploads.push(result);
        showToast(`Uploaded ${file.name} (${result.format_label})`, "success");
      } catch (e) {
        showToast(`Upload failed: ${e.message}`, "error");
      }
    }
    dropZone.classList.remove("uploading");
    updateUploadUI();
  }

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
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

    state.form.personaName = personaName;
    state.form.systemPrompt = systemPrompt;
    state.form.targetName = targetName;
    state.form.syntheticQa = syntheticQa;

    if (!personaName || !systemPrompt) {
      showToast("Persona name and system prompt are required.", "error");
      return;
    }

    const needsTarget = state.uploads.some((u) => u.requires_target_name);
    if (needsTarget && !targetName) {
      showToast("Target username/contact name is required for chat exports.", "error");
      return;
    }

    setLoading(buildBtn, true, "Building…");

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
      setLoading(buildBtn, false);
      buildBtn.textContent = "Build dataset";
      buildBtn.disabled = state.uploads.length === 0;
    }
  });

  updateUploadUI();
}

function renderPreview(section, result) {
  section.hidden = false;
  const warnings =
    result.warnings?.length > 0
      ? `<ul class="warning-list">${result.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>`
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
    <h3 class="step-heading preview-heading">Dataset preview</h3>
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
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;");
}
