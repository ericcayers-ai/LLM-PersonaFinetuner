import { api, showToast } from "../api.js";

export function renderTest(container, state) {
  container.innerHTML = `
    <h2 class="step-heading">Test</h2>
    <p class="step-desc">Chat with your fine-tuned persona. Toggle comparison to see base model vs clone side-by-side.</p>

    <div class="form-grid" style="grid-template-columns:1fr 1fr; margin-bottom:0.75rem">
      <div class="slider-row form-row">
        <label>Temperature <span id="temp-val">0.7</span></label>
        <input type="range" id="temperature" min="0" max="10" value="7" />
      </div>
      <div class="checkbox-row" style="align-self:end">
        <input type="checkbox" id="compare-base" />
        <label for="compare-base">Compare with base model</label>
      </div>
    </div>

    <div class="chat-container">
      <div class="chat-messages" id="chat-messages"></div>
      <div class="chat-input-row">
        <input type="text" id="chat-input" placeholder="Type a message…" ${state.personaId ? "" : "disabled"} />
        <button type="button" class="btn btn-primary" id="send-btn" ${state.personaId ? "" : "disabled"}>Send</button>
      </div>
    </div>
  `;

  const tempEl = container.querySelector("#temperature");
  tempEl.addEventListener("input", () => {
    container.querySelector("#temp-val").textContent = (tempEl.value / 10).toFixed(1);
  });

  const messagesEl = container.querySelector("#chat-messages");
  const inputEl = container.querySelector("#chat-input");
  const sendBtn = container.querySelector("#send-btn");
  const chatHistory = [];

  function appendBubble(role, content, label = null) {
    const div = document.createElement("div");
    div.className = `chat-bubble ${role}`;
    if (label) {
      div.innerHTML = `<div class="chat-bubble-label">${label}</div>${escapeHtml(content)}`;
    } else {
      div.textContent = content;
    }
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function appendCompare(finetuned, base) {
    const grid = document.createElement("div");
    grid.className = "compare-grid";
    grid.innerHTML = `
      <div class="compare-col clone">
        <h4>Fine-tuned</h4>
        <p>${escapeHtml(finetuned)}</p>
      </div>
      <div class="compare-col base">
        <h4>Base model</h4>
        <p>${escapeHtml(base)}</p>
      </div>
    `;
    messagesEl.appendChild(grid);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function send() {
    const text = inputEl.value.trim();
    if (!text || !state.personaId) return;

    inputEl.value = "";
    sendBtn.disabled = true;
    appendBubble("user", text);
    chatHistory.push({ role: "user", content: text });

    const compareBase = container.querySelector("#compare-base").checked;
    const temperature = tempEl.value / 10;

    try {
      const result = await api.chat({
        persona_id: state.personaId,
        messages: chatHistory,
        max_tokens: 512,
        temperature,
        compare_base: compareBase,
      });

      chatHistory.push({ role: "assistant", content: result.response });

      if (compareBase && result.base_response != null) {
        appendCompare(result.response, result.base_response);
      } else {
        appendBubble("assistant", result.response);
      }
    } catch (e) {
      showToast(e.message, "error");
      appendBubble("assistant", `Error: ${e.message}`);
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  sendBtn.addEventListener("click", send);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") send();
  });
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
