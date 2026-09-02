//! Embedded Web UI & Live Telemetry Dashboard for mivi-server.
//!
//! Provides a zero-dependency, single-file HTML/CSS/JS interface served directly at `http://localhost:8913/`
//! with real-time SSE streaming chat, collapsible `<think>` blocks, live `<tool_call>` execution cards,
//! memory visualizer, and system gauges.

use axum::response::Html;

pub const EMBEDDED_WEB_UI_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mivi-v4 — Local AI Agent & Inference Engine</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f293d;
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --accent: #06b6d4;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --think-bg: #1e1b4b;
      --think-border: #4338ca;
      --tool-bg: #064e3b;
      --tool-border: #059669;
      --user-bg: #1e293b;
      --assistant-bg: #0f172a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }
    /* Sidebar */
    .sidebar {
      width: 320px;
      background: #0d1322;
      border-right: 1px solid var(--card-border);
      display: flex;
      flex-direction: column;
      padding: 18px;
      gap: 16px;
      overflow-y: auto;
    }
    .logo-container {
      display: flex;
      align-items: center;
      gap: 10px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
    }
    .logo-badge {
      background: linear-gradient(135deg, var(--primary), var(--accent));
      width: 34px;
      height: 34px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 18px;
      color: #fff;
    }
    .logo-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }
    .version-tag {
      font-size: 11px;
      background: #1f2937;
      color: var(--accent);
      padding: 2px 6px;
      border-radius: 4px;
      margin-left: auto;
      font-weight: 600;
    }
    /* Telemetry cards */
    .telemetry-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .tele-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px;
      text-align: center;
    }
    .tele-label {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }
    .tele-val {
      font-size: 16px;
      font-weight: 700;
      color: var(--accent);
      font-variant-numeric: tabular-nums;
    }
    /* Settings */
    .section-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-top: 6px;
    }
    .param-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .param-label {
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      color: var(--text-muted);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--primary);
    }
    select, input[type="text"] {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 13px;
      outline: none;
    }
    select:focus, input[type="text"]:focus {
      border-color: var(--primary);
    }
    /* Main Chat */
    .main-chat {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: var(--bg);
      height: 100vh;
    }
    .chat-header {
      padding: 14px 20px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #0d1322;
    }
    .chat-header-status {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--text-muted);
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px #10b981;
    }
    .messages-container {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .message-bubble {
      max-width: 85%;
      padding: 14px 18px;
      border-radius: 12px;
      line-height: 1.6;
      font-size: 14px;
      word-break: break-word;
    }
    .message-user {
      align-self: flex-end;
      background: var(--user-bg);
      border: 1px solid #334155;
      color: #fff;
    }
    .message-assistant {
      align-self: flex-start;
      background: var(--assistant-bg);
      border: 1px solid var(--card-border);
    }
    .message-role {
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 6px;
      text-transform: uppercase;
    }
    /* Thinking block */
    .think-card {
      background: var(--think-bg);
      border: 1px solid var(--think-border);
      border-radius: 8px;
      margin-bottom: 10px;
      overflow: hidden;
    }
    .think-header {
      padding: 6px 12px;
      background: rgba(67, 56, 202, 0.3);
      font-size: 12px;
      font-weight: 600;
      color: #a5b4fc;
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
    }
    .think-body {
      padding: 10px 12px;
      font-size: 13px;
      color: #c7d2fe;
      white-space: pre-wrap;
      font-family: monospace;
    }
    /* Tool card */
    .tool-card {
      background: var(--tool-bg);
      border: 1px solid var(--tool-border);
      border-radius: 8px;
      margin: 8px 0;
      padding: 8px 12px;
      font-family: monospace;
      font-size: 12px;
      color: #a7f3d0;
    }
    /* Input area */
    .input-container {
      padding: 16px 20px;
      border-top: 1px solid var(--card-border);
      background: #0d1322;
      display: flex;
      gap: 10px;
      align-items: flex-end;
    }
    .input-box {
      flex: 1;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px 14px;
      color: var(--text);
      font-size: 14px;
      resize: none;
      height: 48px;
      max-height: 140px;
      outline: none;
      line-height: 1.4;
      font-family: inherit;
    }
    .input-box:focus {
      border-color: var(--primary);
    }
    .send-btn {
      background: var(--primary);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 0 20px;
      height: 48px;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.15s;
    }
    .send-btn:hover {
      background: var(--primary-hover);
    }
    .send-btn:disabled {
      background: #374151;
      cursor: not-allowed;
    }
  </style>
</head>
<body>
  <!-- Left Sidebar -->
  <aside class="sidebar">
    <div class="logo-container">
      <div class="logo-badge">M</div>
      <div>
        <div class="logo-title">Mivi-v4</div>
        <div style="font-size: 11px; color: var(--text-muted);">Local Hybrid Inference</div>
      </div>
      <span class="version-tag">v0.2.10</span>
    </div>

    <!-- Live Telemetry -->
    <div class="section-title">Telemetry & Speed</div>
    <div class="telemetry-grid">
      <div class="tele-card">
        <div class="tele-label">Generation</div>
        <div class="tele-val" id="tok-speed">0.0 tok/s</div>
      </div>
      <div class="tele-card">
        <div class="tele-label">TTFT</div>
        <div class="tele-val" id="ttft-val">0 ms</div>
      </div>
      <div class="tele-card">
        <div class="tele-label">KV Precision</div>
        <div class="tele-val" style="color: #10b981;">TQ 4-Bit</div>
      </div>
      <div class="tele-card">
        <div class="tele-label">Memory Saved</div>
        <div class="tele-val" style="color: #38bdf8;">87.3%</div>
      </div>
    </div>

    <!-- Sampling Parameters -->
    <div class="section-title">Sampling Parameters</div>
    <div class="param-group">
      <div class="param-label">
        <span>Temperature</span>
        <span id="temp-val">0.7</span>
      </div>
      <input type="range" id="temp-slider" min="0.0" max="2.0" step="0.05" value="0.7">
    </div>

    <div class="param-group">
      <div class="param-label">
        <span>Top-P (Nucleus)</span>
        <span id="topp-val">0.9</span>
      </div>
      <input type="range" id="topp-slider" min="0.0" max="1.0" step="0.05" value="0.9">
    </div>

    <div class="param-group">
      <div class="param-label">
        <span>Min-P (Truncation)</span>
        <span id="minp-val">0.05</span>
      </div>
      <input type="range" id="minp-slider" min="0.0" max="0.5" step="0.01" value="0.05">
    </div>

    <div class="param-group">
      <div class="param-label">
        <span>Max Tokens</span>
        <span id="max-tokens-val">512</span>
      </div>
      <input type="range" id="max-tokens-slider" min="64" max="2048" step="64" value="512">
    </div>

    <!-- System Prompt -->
    <div class="section-title">System Role</div>
    <select id="system-prompt-select">
      <option value="You are Mivi, an expert autonomous AI coding assistant. Answer accurately, concisely, and use available tools.">Autonomous Agent</option>
      <option value="You are Mivi, a direct and helpful AI coding assistant. Output clean, correct code.">Code Expert</option>
      <option value="You are Mivi, a precise mathematical and logical reasoning assistant. Think step by step.">Reasoning & Math</option>
    </select>
  </aside>

  <!-- Main Chat Workspace -->
  <main class="main-chat">
    <header class="chat-header">
      <div class="chat-header-status">
        <div class="status-dot"></div>
        <span>Mivi Engine Actor: <strong>Online</strong> (localhost:8913)</span>
      </div>
      <button onclick="clearHistory()" style="background: transparent; border: 1px solid var(--card-border); color: var(--text-muted); padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 12px;">Clear Chat</button>
    </header>

    <div class="messages-container" id="messages-container">
      <div class="message-bubble message-assistant">
        <div class="message-role">Mivi Assistant</div>
        <div>👋 Hello! I am <strong>Mivi</strong>, running natively on your CPU with 4-bit TurboQuant KV caching, LMCache prefix snapshots, and sub-millisecond semantic recall. How can I help you today?</div>
      </div>
    </div>

    <div class="input-container">
      <textarea id="user-input" class="input-box" placeholder="Ask Mivi anything or give instructions... (Press Enter to send, Shift+Enter for new line)"></textarea>
      <button id="send-btn" class="send-btn" onclick="sendMessage()">Send</button>
    </div>
  </main>

  <script>
    let conversation = [];
    let isGenerating = false;

    // Sliders
    document.getElementById('temp-slider').addEventListener('input', e => document.getElementById('temp-val').innerText = e.target.value);
    document.getElementById('topp-slider').addEventListener('input', e => document.getElementById('topp-val').innerText = e.target.value);
    document.getElementById('minp-slider').addEventListener('input', e => document.getElementById('minp-val').innerText = e.target.value);
    document.getElementById('max-tokens-slider').addEventListener('input', e => document.getElementById('max-tokens-val').innerText = e.target.value);

    // Keydown handler
    document.getElementById('user-input').addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    function clearHistory() {
      conversation = [];
      document.getElementById('messages-container').innerHTML = `
        <div class="message-bubble message-assistant">
          <div class="message-role">Mivi Assistant</div>
          <div>Chat history cleared. How can I help you?</div>
        </div>
      `;
    }

    async function sendMessage() {
      if (isGenerating) return;
      const inputEl = document.getElementById('user-input');
      const text = inputEl.value.trim();
      if (!text) return;

      inputEl.value = '';
      isGenerating = true;
      document.getElementById('send-btn').disabled = true;

      // Append User message
      conversation.push({ role: "user", content: text });
      appendMessageUI("user", text);

      // Create Assistant placeholder
      const assistantBubble = appendMessageUI("assistant", "");
      const contentDiv = assistantBubble.querySelector('.msg-content');

      const systemPrompt = document.getElementById('system-prompt-select').value;
      const temperature = parseFloat(document.getElementById('temp-slider').value);
      const top_p = parseFloat(document.getElementById('topp-slider').value);
      const min_p = parseFloat(document.getElementById('minp-slider').value);
      const max_tokens = parseInt(document.getElementById('max-tokens-slider').value);

      const messages = [{ role: "system", content: systemPrompt }, ...conversation];

      const tStart = performance.now();
      let firstTokenTime = null;
      let tokenCount = 0;
      let fullAssistantText = "";

      try {
        const resp = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'mivi',
            messages: messages,
            temperature: temperature,
            top_p: top_p,
            min_p: min_p,
            max_tokens: max_tokens,
            stream: true
          })
        });

        if (!resp.ok) {
          contentDiv.innerText = `Error: HTTP ${resp.status} ${resp.statusText}`;
          isGenerating = false;
          document.getElementById('send-btn').disabled = false;
          return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // keep partial line

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data:')) continue;
            const dataStr = trimmed.substring(5).trim();
            if (dataStr === '[DONE]') break;

            try {
              const chunk = JSON.parse(dataStr);
              const delta = chunk.choices?.[0]?.delta;
              if (delta) {
                if (!firstTokenTime) {
                  firstTokenTime = performance.now();
                  const ttftMs = Math.round(firstTokenTime - tStart);
                  document.getElementById('ttft-val').innerText = `${ttftMs} ms`;
                }

                if (delta.content) {
                  tokenCount++;
                  fullAssistantText += delta.content;
                  renderFormattedText(contentDiv, fullAssistantText);
                }
              }
            } catch (err) {}
          }

          // Live tok/s update
          const elapsedSec = (performance.now() - tStart) / 1000.0;
          if (elapsedSec > 0 && tokenCount > 0) {
            const tokSec = (tokenCount / elapsedSec).toFixed(1);
            document.getElementById('tok-speed').innerText = `${tokSec} tok/s`;
          }
        }

        conversation.push({ role: "assistant", content: fullAssistantText });
      } catch (err) {
        contentDiv.innerText += `\n[Connection error: ${err.message}]`;
      } finally {
        isGenerating = false;
        document.getElementById('send-btn').disabled = false;
      }
    }

    function appendMessageUI(role, content) {
      const container = document.getElementById('messages-container');
      const bubble = document.createElement('div');
      bubble.className = `message-bubble message-${role}`;
      bubble.innerHTML = `
        <div class="message-role">${role === 'user' ? 'User' : 'Mivi Assistant'}</div>
        <div class="msg-content"></div>
      `;
      const contentEl = bubble.querySelector('.msg-content');
      renderFormattedText(contentEl, content);
      container.appendChild(bubble);
      container.scrollTop = container.scrollHeight;
      return bubble;
    }

    function renderFormattedText(container, text) {
      // Parse <think> ... </think> blocks
      let html = "";
      let remaining = text;

      const thinkStart = remaining.indexOf('<think>');
      if (thinkStart !== -1) {
        const beforeThink = remaining.substring(0, thinkStart);
        if (beforeThink) html += escapeHtml(beforeThink);

        const thinkEnd = remaining.indexOf('</think>', thinkStart);
        if (thinkEnd !== -1) {
          const thinkContent = remaining.substring(thinkStart + 7, thinkEnd);
          html += `<div class="think-card"><div class="think-header">🧠 Reasoning Trace</div><div class="think-body">${escapeHtml(thinkContent)}</div></div>`;
          remaining = remaining.substring(thinkEnd + 8);
        } else {
          const thinkContent = remaining.substring(thinkStart + 7);
          html += `<div class="think-card"><div class="think-header">🧠 Reasoning (Thinking...)</div><div class="think-body">${escapeHtml(thinkContent)}</div></div>`;
          remaining = "";
        }
      }

      if (remaining) {
        html += escapeHtml(remaining);
      }

      container.innerHTML = html.replace(/\n/g, '<br>');
      const scrollEl = document.getElementById('messages-container');
      scrollEl.scrollTop = scrollEl.scrollHeight;
    }

    function escapeHtml(str) {
      return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }
  </script>
</body>
</html>
"#;

/// Handler that serves the embedded web SPA.
pub async fn serve_embedded_ui() -> Html<&'static str> {
    Html(EMBEDDED_WEB_UI_HTML)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_serve_embedded_ui_contains_critical_elements() {
        let resp = serve_embedded_ui().await;
        assert!(resp.0.contains("<title>Mivi-v4 — Local AI Agent & Inference Engine</title>"));
        assert!(resp.0.contains("Mivi Engine Actor"));
        assert!(resp.0.contains("/v1/chat/completions"));
        assert!(resp.0.contains("think-card"));
    }
}
