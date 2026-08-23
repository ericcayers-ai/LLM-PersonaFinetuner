import { api, showToast } from "./api.js";

// Encode a buffer of Float32 mono PCM samples as a self-contained 16-bit WAV blob.
function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

const SEND_INTERVAL_MS = 2000;

export function createVoiceController() {
  let backends = [];
  let selectedBackend = "pocket";
  let referenceMode = null; // 'upload' | 'live' | null
  let voiceId = null;
  let sessionId = null;
  let liveLocked = false;
  let liveTotalSeconds = 0;
  let lockAfterSeconds = 12;

  let audioCtx = null;
  let processorNode = null;
  let sourceNode = null;
  let mediaStream = null;
  let ws = null;
  let pendingSamples = [];
  let sendTimer = null;
  let liveError = null;

  const audioEl = new Audio();
  let autoSpeak = false;

  let els = {};

  async function loadBackends() {
    try {
      const res = await api.getVoiceBackends();
      backends = res.backends;
      selectedBackend = res.default_backend || "pocket";
    } catch (e) {
      backends = [];
    }
  }

  function hasReference() {
    return (referenceMode === "upload" && voiceId) || (referenceMode === "live" && liveLocked && sessionId);
  }

  function stopLiveCapture() {
    if (sendTimer) clearInterval(sendTimer);
    sendTimer = null;
    if (processorNode) {
      processorNode.disconnect();
      processorNode = null;
    }
    if (sourceNode) {
      sourceNode.disconnect();
      sourceNode = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (audioCtx) {
      audioCtx.close().catch(() => {});
      audioCtx = null;
    }
    if (ws) {
      try {
        ws.close();
      } catch (_) {}
      ws = null;
    }
  }

  function flushPending() {
    if (!ws || ws.readyState !== WebSocket.OPEN || pendingSamples.length === 0) return;
    const total = pendingSamples.reduce((n, arr) => n + arr.length, 0);
    const merged = new Float32Array(total);
    let offset = 0;
    for (const arr of pendingSamples) {
      merged.set(arr, offset);
      offset += arr.length;
    }
    pendingSamples = [];
    const blob = encodeWav(merged, audioCtx.sampleRate);
    blob.arrayBuffer().then((buf) => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(buf);
    });
  }

  async function startLiveCapture(render) {
    try {
      liveError = null;
      const session = await api.startVoiceLiveSession(lockAfterSeconds);
      sessionId = session.session_id;
      liveLocked = false;
      liveTotalSeconds = 0;

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not support microphone capture.");
      }
      const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextCtor) {
        throw new Error("This browser does not support Web Audio (AudioContext).");
      }
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx = new AudioContextCtor();
      sourceNode = audioCtx.createMediaStreamSource(mediaStream);
      processorNode = audioCtx.createScriptProcessor(4096, 1, 1);

      processorNode.onaudioprocess = (event) => {
        const data = event.inputBuffer.getChannelData(0);
        pendingSamples.push(new Float32Array(data));
      };
      sourceNode.connect(processorNode);
      processorNode.connect(audioCtx.destination);

      ws = new WebSocket(api.voiceLiveStreamUrl(sessionId));
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        sendTimer = setInterval(flushPending, SEND_INTERVAL_MS);
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.error) {
          liveError = msg.error;
          render();
          return;
        }
        liveTotalSeconds = msg.total_seconds;
        liveLocked = msg.locked;
        if (liveLocked) {
          stopLiveCapture();
          referenceMode = "live";
          showToast("Voice clone locked in.", "success");
        }
        render();
      };
      ws.onerror = () => {
        liveError = "Live voice connection failed.";
        stopLiveCapture();
        sessionId = null;
        render();
      };
      referenceMode = "live";
      render();
    } catch (e) {
      liveError = e.message || String(e);
      stopLiveCapture();
      render();
    }
  }

  async function synthesizeAndPlay(text) {
    if (!autoSpeak || !hasReference() || !text?.trim()) return;
    try {
      const blob = await api.synthesizeVoice({
        text,
        backend: selectedBackend,
        voice_id: referenceMode === "upload" ? voiceId : null,
        session_id: referenceMode === "live" ? sessionId : null,
      });
      audioEl.src = URL.createObjectURL(blob);
      await audioEl.play().catch(() => {});
    } catch (e) {
      showToast(`Voice playback failed: ${e.message}`, "error");
    }
  }

  function render(container) {
    const focusedId = container.contains(document.activeElement) ? document.activeElement.id : null;
    const backendOptions = backends
      .map((b) => `<option value="${b.name}" ${b.name === selectedBackend ? "selected" : ""}>${b.name} (${b.weight})</option>`)
      .join("");

    container.innerHTML = `
      <div class="voice-panel">
        <div class="voice-panel-head">
          <strong>Voice clone</strong>
          <label class="checkbox-row">
            <input type="checkbox" id="voice-autospeak" ${autoSpeak ? "checked" : ""} />
            Speak replies
          </label>
        </div>

        <div class="voice-panel-row">
          <label for="voice-backend">Backend</label>
          <select id="voice-backend">${backendOptions}</select>
        </div>

        <div class="voice-panel-row voice-panel-tabs" role="tablist" aria-label="Voice reference source">
          <button type="button" class="btn btn-secondary btn-sm" id="voice-tab-upload" role="tab"
            aria-selected="${referenceMode !== "live"}" aria-controls="voice-upload-area">Upload sample</button>
          <button type="button" class="btn btn-secondary btn-sm" id="voice-tab-live" role="tab"
            aria-selected="${referenceMode === "live"}" aria-controls="voice-live-area">Live record</button>
        </div>

        <div id="voice-upload-area" role="tabpanel" aria-labelledby="voice-tab-upload" ${referenceMode === "live" ? "hidden" : ""}>
          <label for="voice-file-input" class="sr-only">Upload reference audio file</label>
          <input type="file" id="voice-file-input" accept="audio/*" />
          <div class="voice-status" aria-live="polite">
            ${voiceId ? "Reference uploaded." : "No reference uploaded yet."}
          </div>
        </div>

        <div id="voice-live-area" role="tabpanel" aria-labelledby="voice-tab-live" ${referenceMode === "upload" ? "hidden" : ""}>
          <div class="voice-panel-row">
            <label for="voice-lock-seconds">Lock after (s)</label>
            <input type="number" id="voice-lock-seconds" min="3" max="60" value="${lockAfterSeconds}" />
          </div>
          <button type="button" class="btn btn-primary btn-sm" id="voice-live-start" ${sessionId && !liveLocked ? "disabled" : ""}>
            ${liveLocked ? "Re-record" : "Start recording"}
          </button>
          <div class="voice-status" aria-live="polite">
            ${
              liveError
                ? `Error: ${escapeHtml(liveError)}`
                : sessionId
                  ? `${liveLocked ? "Locked" : "Listening"} — ${liveTotalSeconds.toFixed(1)}s / ${lockAfterSeconds}s`
                  : "Not recording."
            }
          </div>
        </div>
      </div>
    `;

    els = {
      backendSel: container.querySelector("#voice-backend"),
      tabUpload: container.querySelector("#voice-tab-upload"),
      tabLive: container.querySelector("#voice-tab-live"),
      uploadArea: container.querySelector("#voice-upload-area"),
      liveArea: container.querySelector("#voice-live-area"),
      fileInput: container.querySelector("#voice-file-input"),
      lockSecondsInput: container.querySelector("#voice-lock-seconds"),
      liveStartBtn: container.querySelector("#voice-live-start"),
      autospeakCheckbox: container.querySelector("#voice-autospeak"),
    };

    if (focusedId) {
      container.querySelector(`#${focusedId}`)?.focus();
    }

    els.backendSel.addEventListener("change", () => {
      selectedBackend = els.backendSel.value;
    });

    els.autospeakCheckbox.addEventListener("change", () => {
      autoSpeak = els.autospeakCheckbox.checked;
    });

    els.tabUpload.addEventListener("click", () => {
      referenceMode = "upload";
      render(container);
    });
    els.tabLive.addEventListener("click", () => {
      referenceMode = "live";
      render(container);
    });

    els.fileInput.addEventListener("change", async () => {
      const file = els.fileInput.files[0];
      if (!file) return;
      try {
        const res = await api.uploadVoiceReference(file);
        voiceId = res.voice_id;
        referenceMode = "upload";
        showToast(`Reference uploaded (${res.duration_seconds.toFixed(1)}s).`, "success");
      } catch (e) {
        showToast(`Upload failed: ${e.message}`, "error");
      }
      render(container);
    });

    els.lockSecondsInput.addEventListener("change", () => {
      lockAfterSeconds = Number(els.lockSecondsInput.value) || 12;
    });

    els.liveStartBtn.addEventListener("click", () => {
      stopLiveCapture();
      startLiveCapture(() => render(container));
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  return {
    init: loadBackends,
    render,
    synthesizeAndPlay,
    hasReference,
    stopLiveCapture,
  };
}
