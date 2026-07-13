import { api, setLoading, showToast } from "../api.js";
import { MicrophoneCapture } from "../audio.js";

export function renderVoice(container, state, onProfileChanged) {
  const hasPersona = Boolean(state.personaId);
  container.innerHTML = `
    <h2 class="step-heading">Voice</h2>
    <p class="step-desc">Clone a consented voice with Chatterbox and transcribe local audio with faster-whisper.</p>
    ${hasPersona ? "" : '<div class="empty-state">Select or create a persona before adding a voice.</div>'}

    <div class="speech-engine-status" id="speech-engine-status">Checking local speech engines…</div>
    <div class="voice-grid">
      <section class="voice-card">
        <h3>Voice reference</h3>
        <p class="voice-card-desc">Use one clean speaker, no music, and 8–15 seconds of natural speech.</p>
        <div id="voice-profile-status" class="profile-status">No voice reference saved.</div>
        <div class="form-row">
          <label for="voice-file">Reference audio</label>
          <input id="voice-file" type="file" accept=".wav,.mp3,.m4a,.flac,.ogg,.webm,audio/*" ${hasPersona ? "" : "disabled"} />
        </div>
        <div class="btn-row">
          <button type="button" class="btn btn-secondary" id="record-reference" ${hasPersona ? "" : "disabled"}>Record reference</button>
          <button type="button" class="btn btn-secondary btn-sm" id="remove-voice" hidden>Remove</button>
        </div>
        <audio id="reference-preview" class="audio-preview" controls hidden></audio>
        <div class="checkbox-row consent-row">
          <input type="checkbox" id="voice-consent" ${hasPersona ? "" : "disabled"} />
          <label for="voice-consent">I confirm the voice owner explicitly consented to cloning and use.</label>
        </div>
        <button type="button" class="btn btn-primary" id="save-voice" ${hasPersona ? "" : "disabled"}>Save cloned voice</button>
        <p class="responsible-note">Generated Chatterbox audio contains an imperceptible PerTh watermark.</p>
      </section>

      <section class="voice-card">
        <h3>Speech lab</h3>
        <p class="voice-card-desc">Test the same STT and TTS engines used during live calls.</p>
        <div class="form-row">
          <label for="speech-lab-text">Text to speak</label>
          <textarea id="speech-lab-text" maxlength="1200" ${hasPersona ? "" : "disabled"}>Hello — this is a preview of my cloned voice.</textarea>
        </div>
        <button type="button" class="btn btn-secondary" id="speak-preview" ${hasPersona ? "" : "disabled"}>Generate voice preview</button>
        <audio id="tts-preview" class="audio-preview" controls hidden></audio>
        <div class="speech-divider"><span>speech to text</span></div>
        <button type="button" class="btn btn-secondary" id="record-transcript">Record for transcription</button>
        <div id="transcript-result" class="transcript-result">Your transcript will appear here.</div>
      </section>
    </div>
  `;

  const fileInput = container.querySelector("#voice-file");
  const recordReferenceBtn = container.querySelector("#record-reference");
  const saveBtn = container.querySelector("#save-voice");
  const removeBtn = container.querySelector("#remove-voice");
  const referencePreview = container.querySelector("#reference-preview");
  const profileStatus = container.querySelector("#voice-profile-status");
  const speakBtn = container.querySelector("#speak-preview");
  const ttsPreview = container.querySelector("#tts-preview");
  const transcribeBtn = container.querySelector("#record-transcript");
  const transcriptResult = container.querySelector("#transcript-result");
  let capture = null;
  let referenceFile = null;
  let recordingPurpose = null;
  let recordingTimer = null;
  let referenceUrl = null;
  let ttsUrl = null;

  function showAudio(audio, blob, previousUrl) {
    if (previousUrl) URL.revokeObjectURL(previousUrl);
    const url = URL.createObjectURL(blob);
    audio.src = url;
    audio.hidden = false;
    return url;
  }

  async function getCapture() {
    if (!capture) capture = await MicrophoneCapture.open();
    await capture.resume();
    return capture;
  }

  function stopRecording() {
    if (!capture || !recordingPurpose) return;
    clearTimeout(recordingTimer);
    const purpose = recordingPurpose;
    recordingPurpose = null;
    const blob = capture.stopManual();
    recordReferenceBtn.textContent = "Record reference";
    transcribeBtn.textContent = "Record for transcription";
    if (!blob) return;

    if (purpose === "reference") {
      referenceFile = new File([blob], "voice-reference.wav", { type: "audio/wav" });
      referenceUrl = showAudio(referencePreview, blob, referenceUrl);
      showToast("Reference recording ready", "success");
    } else {
      transcribeBlob(blob);
    }
  }

  async function startRecording(purpose) {
    if (recordingPurpose) {
      stopRecording();
      return;
    }
    try {
      const mic = await getCapture();
      mic.startManual();
      recordingPurpose = purpose;
      const button = purpose === "reference" ? recordReferenceBtn : transcribeBtn;
      button.textContent = "Stop recording";
      recordingTimer = setTimeout(stopRecording, purpose === "reference" ? 15000 : 20000);
    } catch (error) {
      showToast(error.message, "error");
    }
  }

  async function transcribeBlob(blob) {
    setLoading(transcribeBtn, true, "Transcribing…");
    transcriptResult.textContent = "Transcribing locally…";
    try {
      const file = new File([blob], "dictation.wav", { type: "audio/wav" });
      const result = await api.transcribe(file);
      transcriptResult.textContent = result.text;
      showToast(`Detected ${result.language}`, "success");
    } catch (error) {
      transcriptResult.textContent = error.message;
      showToast(error.message, "error");
    } finally {
      setLoading(transcribeBtn, false);
      transcribeBtn.textContent = "Record for transcription";
    }
  }

  async function loadCapabilities() {
    const status = container.querySelector("#speech-engine-status");
    try {
      const info = await api.getVoiceCapabilities();
      status.innerHTML = `
        <span class="${info.tts_installed ? "engine-ok" : "engine-missing"}">TTS · ${escapeHtml(info.tts_model)}</span>
        <span class="${info.stt_installed ? "engine-ok" : "engine-missing"}">STT · ${escapeHtml(info.stt_model)}</span>
        <span>Transport · WebSocket</span>
      `;
    } catch (error) {
      status.textContent = error.message;
    }
  }

  async function loadProfile() {
    if (!state.personaId) return;
    try {
      const profile = await api.getVoiceProfile(state.personaId);
      profileStatus.textContent = `${profile.sample_filename} · ${formatBytes(profile.sample_size_bytes)} · ${profile.language}`;
      profileStatus.classList.add("ready");
      removeBtn.hidden = false;
    } catch (_) {
      profileStatus.textContent = "No voice reference saved.";
      profileStatus.classList.remove("ready");
      removeBtn.hidden = true;
    }
  }

  fileInput?.addEventListener("change", () => {
    referenceFile = fileInput.files?.[0] || null;
    if (referenceFile) referenceUrl = showAudio(referencePreview, referenceFile, referenceUrl);
  });
  recordReferenceBtn?.addEventListener("click", () => startRecording("reference"));
  transcribeBtn?.addEventListener("click", () => startRecording("transcribe"));

  saveBtn?.addEventListener("click", async () => {
    const file = referenceFile || fileInput.files?.[0];
    if (!file) {
      showToast("Choose or record a voice reference first.", "error");
      return;
    }
    if (!container.querySelector("#voice-consent").checked) {
      showToast("Voice-owner consent must be confirmed.", "error");
      return;
    }
    setLoading(saveBtn, true, "Saving…");
    try {
      await api.uploadVoiceProfile(state.personaId, file, true, "en");
      await loadProfile();
      await onProfileChanged?.();
      showToast("Cloned voice is ready", "success");
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      setLoading(saveBtn, false);
    }
  });

  removeBtn.addEventListener("click", async () => {
    try {
      await api.deleteVoiceProfile(state.personaId);
      await loadProfile();
      await onProfileChanged?.();
      showToast("Voice reference removed", "success");
    } catch (error) {
      showToast(error.message, "error");
    }
  });

  speakBtn?.addEventListener("click", async () => {
    const text = container.querySelector("#speech-lab-text").value.trim();
    if (!text) return;
    setLoading(speakBtn, true, "Generating…");
    try {
      const audio = await api.synthesize(state.personaId, text);
      ttsUrl = showAudio(ttsPreview, audio, ttsUrl);
      await ttsPreview.play();
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      setLoading(speakBtn, false);
    }
  });

  const onStepChange = (event) => {
    if (event.detail !== "voice" && capture) {
      capture.close();
      capture = null;
      recordingPurpose = null;
      clearTimeout(recordingTimer);
    }
  };
  document.addEventListener("personafinetuner:stepchange", onStepChange);
  loadCapabilities();
  loadProfile();
}

function formatBytes(bytes) {
  return bytes < 1024 * 1024
    ? `${Math.round(bytes / 1024)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
