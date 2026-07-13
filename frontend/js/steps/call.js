import { showToast } from "../api.js";
import { MicrophoneCapture } from "../audio.js";

export function renderCall(container, state) {
  const persona = state.personas.find((item) => item.id === state.personaId);
  const trained = Boolean(persona?.adapter_path || persona?.status === "trained");
  const voiced = Boolean(persona?.voice_profile);
  const canCall = Boolean(persona && trained && voiced);
  container.innerHTML = `
    <h2 class="step-heading">Live call</h2>
    <p class="step-desc">Talk naturally. Silence detection sends each turn through local STT → persona LLM → cloned TTS.</p>
    ${persona ? "" : '<div class="empty-state">Select a persona to start a call.</div>'}
    ${persona && !trained ? '<div class="empty-state">Train this persona before calling.</div>' : ""}
    ${persona && trained && !voiced ? '<div class="empty-state">Add a cloned voice in the Voice step before calling.</div>' : ""}

    <div class="call-stage ${canCall ? "" : "disabled"}">
      <div class="call-orb" id="call-orb" aria-hidden="true"><span></span><span></span><span></span></div>
      <div class="call-persona">${persona ? escapeHtml(persona.name) : "No persona selected"}</div>
      <div class="call-status" id="call-status" aria-live="polite">Ready to connect</div>
      <p class="call-hint">Speak after “Listening” appears. A 0.7 second pause ends your turn.</p>
      <div class="call-actions">
        <button type="button" class="btn btn-primary call-button" id="start-call" ${canCall ? "" : "disabled"}>Start call</button>
        <button type="button" class="btn btn-danger call-button" id="end-call" disabled>End call</button>
      </div>
    </div>

    <section class="call-transcript">
      <div class="call-transcript-header">
        <h3>Live transcript</h3>
        <span>Processed locally</span>
      </div>
      <div id="call-messages" class="call-messages">
        <p class="chat-empty">The conversation will appear here.</p>
      </div>
    </section>
  `;

  const startBtn = container.querySelector("#start-call");
  const endBtn = container.querySelector("#end-call");
  const statusEl = container.querySelector("#call-status");
  const orb = container.querySelector("#call-orb");
  const messagesEl = container.querySelector("#call-messages");
  let socket = null;
  let capture = null;
  let activeAudio = null;
  let activeAudioUrl = null;
  let callEnded = true;
  let serverReady = false;

  function setStatus(status, label) {
    statusEl.textContent = label;
    orb.className = `call-orb ${status}`;
  }

  function appendTranscript(role, text) {
    messagesEl.querySelector(".chat-empty")?.remove();
    const row = document.createElement("div");
    row.className = `call-line ${role}`;
    const label = document.createElement("span");
    label.className = "call-line-label";
    label.textContent = role === "user" ? "You" : persona.name;
    const body = document.createElement("p");
    body.textContent = text;
    row.append(label, body);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function enableListening() {
    if (callEnded || activeAudio) return;
    serverReady = true;
    capture?.setEnabled(true);
    setStatus("listening", "Listening");
  }

  async function playSpeech(buffer) {
    capture?.setEnabled(false);
    serverReady = false;
    setStatus("speaking", `${persona.name} is speaking`);
    if (activeAudioUrl) URL.revokeObjectURL(activeAudioUrl);
    activeAudioUrl = URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
    activeAudio = new Audio(activeAudioUrl);
    activeAudio.onended = () => {
      activeAudio = null;
      if (!callEnded) enableListening();
    };
    activeAudio.onerror = () => {
      activeAudio = null;
      showToast("Could not play the synthesized response.", "error");
      if (!callEnded) enableListening();
    };
    try {
      await activeAudio.play();
    } catch (error) {
      activeAudio = null;
      showToast(error.message, "error");
      enableListening();
    }
  }

  function handleEvent(event) {
    if (event.type === "ready") {
      serverReady = true;
      enableListening();
    } else if (event.type === "state") {
      capture?.setEnabled(false);
      serverReady = false;
      const labels = {
        transcribing: "Transcribing your voice",
        thinking: `${persona.name} is thinking`,
        speaking: "Generating cloned speech",
      };
      setStatus(event.state, labels[event.state] || event.state);
    } else if (event.type === "user_transcript") {
      appendTranscript("user", event.text);
    } else if (event.type === "assistant_transcript") {
      appendTranscript("assistant", event.text);
    } else if (event.type === "error") {
      showToast(event.message, "error");
      if (event.recoverable) enableListening();
    }
  }

  async function startCall() {
    startBtn.disabled = true;
    setStatus("connecting", "Requesting microphone");
    try {
      capture = await MicrophoneCapture.open();
      await capture.resume();
      callEnded = false;
      capture.startVoiceActivity(async (blob) => {
        if (!serverReady || socket?.readyState !== WebSocket.OPEN) return;
        serverReady = false;
        capture.setEnabled(false);
        setStatus("transcribing", "Sending your turn");
        socket.send(await blob.arrayBuffer());
      });

      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${location.host}/api/voice/call/${state.personaId}`);
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        endBtn.disabled = false;
        setStatus("connecting", "Loading local speech models");
      };
      socket.onmessage = (message) => {
        if (typeof message.data === "string") {
          handleEvent(JSON.parse(message.data));
        } else {
          playSpeech(message.data);
        }
      };
      socket.onerror = () => showToast("The voice call connection failed.", "error");
      socket.onclose = () => {
        if (!callEnded) endCall("Call disconnected");
      };
    } catch (error) {
      callEnded = true;
      startBtn.disabled = false;
      setStatus("", "Could not start call");
      capture?.close();
      capture = null;
      showToast(error.message, "error");
    }
  }

  function endCall(label = "Call ended") {
    callEnded = true;
    serverReady = false;
    if (socket && socket.readyState < WebSocket.CLOSING) socket.close(1000, "Call ended");
    socket = null;
    capture?.close();
    capture = null;
    activeAudio?.pause();
    activeAudio = null;
    if (activeAudioUrl) URL.revokeObjectURL(activeAudioUrl);
    activeAudioUrl = null;
    startBtn.disabled = !canCall;
    endBtn.disabled = true;
    setStatus("", label);
  }

  startBtn.addEventListener("click", startCall);
  endBtn.addEventListener("click", () => endCall());
  const onStepChange = (event) => {
    if (event.detail !== "call" && !callEnded) endCall("Call ended");
  };
  document.addEventListener("personafinetuner:stepchange", onStepChange);
  return () => {
    document.removeEventListener("personafinetuner:stepchange", onStepChange);
    endCall("Call ended");
  };
}

function escapeHtml(value) {
  return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
