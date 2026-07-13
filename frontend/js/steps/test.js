import { api, showToast, setLoading } from "../api.js";

export function renderTest(container, state) {
  const hasPersona = Boolean(state.personaId);
  container.innerHTML = `
    <h2 class="step-heading">Test</h2>
    <p class="step-desc">Chat with your fine-tuned persona. Toggle comparison to see base model vs clone side-by-side.</p>

    ${hasPersona ? "" : '<div class="empty-state">Train a persona first, then select it from the sidebar to chat.</div>'}

    <div class="test-controls">
      <div class="slider-row form-row">
        <label for="temperature">Temperature <span id="temp-val">0.7</span></label>
        <input type="range" id="temperature" min="0" max="10" value="7" />
      </div>
      <div class="checkbox-row">
        <input type="checkbox" id="compare-base" />
        <label for="compare-base">Compare with base model</label>
      </div>
      <div class="btn-row chat-actions">
        <button type="button" class="btn btn-secondary btn-sm" id="clear-chat" ${hasPersona ? "" : "disabled"}>Clear chat</button>
      </div>
    </div>

    <div class="chat-container">
      <div class="chat-messages" id="chat-messages">
        ${hasPersona ? "" : '<p class="chat-empty">Messages will appear here.</p>'}
      </div>
      <div class="chat-input-row">
        <input type="text" id="chat-input" placeholder="Type a message…" ${hasPersona ? "" : "disabled"} />
        <button type="button" class="btn btn-primary" id="send-btn" ${hasPersona ? "" : "disabled"}>Send</button>
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
  const clearBtn = container.querySelector("#clear-chat");
  const chatHistory = [];

  function formatContent(text) {
    return escapeHtml(text).replace(/\n/g, "<br>");
  }

  function appendBubble(role, content, label = null) {
    const empty = messagesEl.querySelector(".chat-empty");
    if (empty) empty.remove();

    const div = document.createElement("div");
    div.className = `chat-bubble ${role}`;
    const body = document.createElement("div");
    body.className = "chat-bubble-body";
    body.innerHTML = formatContent(content);

    if (label) {
      const lbl = document.createElement("div");
      lbl.className = "chat-bubble-label";
      lbl.textContent = label;
      div.appendChild(lbl);
    }
    div.appendChild(body);

    const actions = document.createElement("div");
    actions.className = "chat-bubble-actions";
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "btn-icon";
    copyBtn.textContent = "Copy";
    copyBtn.setAttribute("aria-label", "Copy message");
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(content);
        showToast("Copied to clipboard", "success");
      } catch (_) {
        showToast("Could not copy", "error");
      }
    });
    actions.appendChild(copyBtn);
    div.appendChild(actions);

    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function appendCompare(finetuned, base) {
    const grid = document.createElement("div");
    grid.className = "compare-grid";
    grid.innerHTML = `
      <div class="compare-col clone">
        <h4>Fine-tuned</h4>
        <p>${formatContent(finetuned)}</p>
      </div>
      <div class="compare-col base">
        <h4>Base model</h4>
        <p>${formatContent(base)}</p>
      </div>
    `;
    messagesEl.appendChild(grid);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function send() {
    const text = inputEl.value.trim();
    if (!text || !state.personaId) return;

    inputEl.value = "";
    setLoading(sendBtn, true, "…");
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
      setLoading(sendBtn, false);
      sendBtn.textContent = "Send";
      inputEl.focus();
    }
  }

  sendBtn.addEventListener("click", send);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") send();
  });

  clearBtn.addEventListener("click", () => {
    chatHistory.length = 0;
    messagesEl.innerHTML = '<p class="chat-empty">Chat cleared.</p>';
  });
}

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
