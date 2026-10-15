"use strict";
const $ = (id) => document.getElementById(id);
const escapeHTML = EoleUI.escape;
const format = (value, digits = 1) =>
  Number.isFinite(value) ? value.toFixed(digits) : "—";
let chats = [],
  selected = null,
  state = null,
  busy = false,
  activeRequest = null,
  attached = null,
  benchmark = null,
  stopRequested = false;
try {
  chats = JSON.parse(localStorage.getItem("eole.chats") || "[]");
  if (!Array.isArray(chats)) chats = [];
} catch {
  chats = [];
}
try {
  if (localStorage.getItem("eole.theme") === "dark")
    document.body.classList.add("dark");
} catch {}
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => {
    $("toast").hidden = true;
  }, 4000);
}
function save() {
  try {
    localStorage.setItem("eole.chats", JSON.stringify(chats));
  } catch {
    toast(
      "Browser storage is full. Export your chat before clearing old conversations.",
    );
  }
}
function current() {
  return chats.find((c) => c.id === selected);
}
function freshChat() {
  if (busy)
    return toast("Stop the active generation before starting a new chat.");
  selected = null;
  attached = null;
  $("attachment").hidden = true;
  renderMessages();
  renderHistory();
  switchView("chat");
  $("prompt").focus();
}
function switchView(view) {
  document.body.classList.remove("history-open");
  $("mobile-history").setAttribute("aria-expanded", "false");
  document
    .querySelectorAll(".view")
    .forEach((el) => (el.hidden = el.id !== `view-${view}`));
  document
    .querySelectorAll("[data-view]")
    .forEach((el) => el.classList.toggle("selected", el.dataset.view === view));
  $("view-title").textContent = {
    chat: "Chat",
    translate: "Translate",
    ocr: "Documents",
    monitor: "Monitor",
    benchmark: "Benchmark",
    connect: "API & setup",
  }[view];
}
function renderHistory() {
  $("history").innerHTML = chats
    .map(
      (c) =>
        `<div class="history-entry ${c.id === selected ? "current" : ""}"><button class="open" data-chat="${escapeHTML(c.id)}">${escapeHTML(c.title)}</button><button class="delete" data-delete="${escapeHTML(c.id)}" aria-label="Delete conversation">×</button></div>`,
    )
    .join("");
}
function markdown(raw) {
  const blocks = [];
  let text = escapeHTML(raw).replace(
    /```([^\n]*)\n([\s\S]*?)(?:```|$)/g,
    (_, lang, code) => {
      const index = blocks.length;
      blocks.push(
        `<div class="code-head"><span>${lang || "code"}</span><button data-copy-code="${escapeHTML(
          encodeURIComponent(
            code
              .replace(/&lt;/g, "<")
              .replace(/&gt;/g, ">")
              .replace(/&quot;/g, '"')
              .replace(/&#39;/g, "'")
              .replace(/&amp;/g, "&"),
          ),
        )}">Copy code</button></div><pre><code>${code}</code></pre>`,
      );
      return `\n\n@@BLOCK${index}@@\n\n`;
    },
  );
  text = text
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return text
    .split(/\n\s*\n/)
    .filter(Boolean)
    .map((p) => {
      const match = p.trim().match(/^@@BLOCK(\d+)@@$/);
      if (match) return blocks[Number(match[1])] || "";
      if (/^#{1,3} /.test(p))
        return p.replace(/^#{1,3} (.+)$/gm, "<h3>$1</h3>");
      if (/^(?:[-*]|\d+\.) /.test(p))
        return (
          "<ul>" +
          p
            .split("\n")
            .map((l) => "<li>" + l.replace(/^(?:[-*]|\d+\.) /, "") + "</li>")
            .join("") +
          "</ul>"
        );
      return "<p>" + p.replace(/\n/g, "<br>") + "</p>";
    })
    .join("");
}
function statsHTML(s) {
  if (!s) return "";
  return `<div class="response-stats"><span><strong>${format(s.decode_tokens_per_second)}</strong> tok/s decode</span><span>${format(s.first_token_seconds, 2)} s first token</span><span>${s.output_tokens ?? "—"} output tokens</span><span>${s.mtp_active ? "MTP active · " + format((s.mtp_acceptance ?? 0) * 100, 0) + "% acceptance" : "MTP off"}</span>${s.warmup_seconds > 1 ? `<span>${format(s.warmup_seconds)} s warmup</span>` : ""}${s.cancelled ? "<span>Stopped</span>" : s.finish_reason === "length" ? "<span>Output limit reached</span>" : ""}</div>`;
}
function messageHTML(m, index) {
  const assistant = m.role === "assistant";
  return `<article class="message ${assistant ? "assistant" : "user"}"><div class="message-header"><span class="avatar">${assistant ? "≋" : "Y"}</span>${assistant ? "Eole" : "You"}</div>${m.reasoning ? `<details class="reasoning" ${busy && index === current()?.messages.length - 1 ? "open" : ""}><summary>Thinking</summary><pre>${escapeHTML(m.reasoning)}</pre></details>` : ""}<div class="message-body">${m.error ? `<div class="error">${escapeHTML(m.error)}</div>` : m.content ? markdown(m.content) : busy ? '<span class="pending muted">Preparing the model / reading your prompt…</span>' : ""}</div>${statsHTML(m.stats)}${!busy ? `<div class="message-actions"><button data-copy-message="${index}">Copy</button><button data-edit-message="${index}">${assistant ? "Regenerate" : "Edit & resend"}</button>${assistant && m.stats?.finish_reason === "length" && index === current()?.messages.length - 1 ? '<button data-continue="true">Continue →</button>' : ""}</div>` : ""}</article>`;
}
function renderMessages() {
  const chat = current();
  if (!chat || !chat.messages.length) {
    $("messages").innerHTML = welcomeHTML;
    return;
  }
  const nearBottom =
    $("messages").scrollHeight -
      $("messages").scrollTop -
      $("messages").clientHeight <
    120;
  $("messages").innerHTML = chat.messages.map(messageHTML).join("");
  if (nearBottom) $("messages").scrollTop = $("messages").scrollHeight;
}
const welcomeHTML = $("messages").innerHTML;
function setBusy(value) {
  busy = value;
  if (value) stopRequested = false;
  $("benchmark-stop").hidden = !value;
  $("send").hidden = value;
  $("stop").hidden = !value;
  $("benchmark-run").disabled = value;
  $("model").disabled = value;
  $("max-tokens").disabled = value;
  $("preset").disabled = value;
  $("thinking").disabled = value;
  $("prompt").disabled = value;
  $("run-state").textContent = value
    ? "Preparing model / reading prompt…"
    : "Ready when you are";
  if (!value) activeRequest = null;
}
function selectedModel() {
  return state?.models.find((m) => m.id === $("model").value);
}
function requestPayload(messages, maxTokens = Number($("max-tokens").value)) {
  return {
    model: $("model").value,
    messages: messages.map((m) => ({
      role: m.role,
      content: m.content,
      ...(m.role === "assistant" && m.reasoning
        ? { reasoning_content: m.reasoning }
        : {}),
    })),
    max_tokens: maxTokens,
    preset: $("preset").value,
    thinking: $("thinking").checked,
    temperature: Number($("temperature").value),
  };
}
async function consume(payload, onDelta) {
  const response = await fetch("/ui/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let result = null;
  try {
    for await (const event of EoleUI.events(response)) {
      if (event.type === "start") {
        activeRequest = event.request_id;
        if (stopRequested)
          await fetch(`/ui/cancel/${activeRequest}`, { method: "POST" });
      }
      if (event.type === "delta") {
        $("run-state").textContent = "Generating locally…";
        onDelta?.(event);
      }
      if (event.type === "error") throw new Error(event.message);
      if (event.type === "done") {
        result = event.stats;
        activeRequest = null;
      }
    }
    if (!result)
      throw new Error("The stream ended without completion metrics.");
    return result;
  } catch (error) {
    if (activeRequest)
      await fetch(`/ui/cancel/${activeRequest}`, { method: "POST" }).catch(
        () => {},
      );
    throw error;
  }
}
async function send(text = null) {
  if (busy) return;
  text = (text ?? $("prompt").value).trim();
  if (!text && !attached) return;
  if (!$("model").value) return toast("Wait for a model to become available.");
  if (attached)
    text += `\n\nAttached file: ${attached.name}\n\n${attached.content}`;
  if (!current()) {
    const chat = {
      id: crypto.randomUUID(),
      title: text.slice(0, 45),
      messages: [],
      created: Date.now(),
    };
    chats.unshift(chat);
    selected = chat.id;
  }
  const chat = current();
  chat.messages.push({ role: "user", content: text });
  const messages = chat.messages.slice();
  const reply = { role: "assistant", content: "", reasoning: "" };
  chat.messages.push(reply);
  $("prompt").value = "";
  attached = null;
  $("attachment").hidden = true;
  setBusy(true);
  renderHistory();
  renderMessages();
  $("messages").scrollTop = $("messages").scrollHeight;
  save();
  let lastRender = 0;
  try {
    reply.stats = await consume(requestPayload(messages), (e) => {
      if (e.kind === "reasoning") reply.reasoning += e.text;
      else reply.content += e.text;
      const now = performance.now();
      if (now - lastRender > 45) {
        renderMessages();
        lastRender = now;
      }
    });
  } catch (error) {
    reply.error = error.message;
  } finally {
    setBusy(false);
    renderMessages();
    save();
    refreshState();
    $("prompt").focus();
  }
}
function updateContext() {
  const model = selectedModel();
  if (!model) return;
  $("model-caption").textContent =
    `${model.weight_bits ? "INT" + model.weight_bits + " · " : ""}${model.dtype.replace("torch.", "")} compute · ${model.context_length ? Number(model.context_length).toLocaleString() + " context" : "context automatic"} · ${state.compile_enabled ? "compiled mode " + state.compile_mode : "eager"}`;
  const max = Number(model.max_tokens) || 4096;
  $("max-tokens").max = max;
  if (Number($("max-tokens").value) > max) $("max-tokens").value = max;
  const last = current()
    ?.messages.filter((m) => m.stats)
    .at(-1)?.stats;
  $("context-note").textContent = Number.isFinite(last?.prompt_tokens)
    ? `Last input: ${last.prompt_tokens.toLocaleString()} tokens · ${model.context_length ? Number(model.context_length).toLocaleString() + " context" : "automatic context"}`
    : "Conversations are saved in this browser.";
}
function metric(label, value, unit, caption) {
  return `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}<small>${unit}</small></div><div class="metric-caption">${caption}</div></div>`;
}
function renderMonitor() {
  if (!state) return;
  const h = state.hardware,
    last = state.recent.find((r) =>
      Number.isFinite(r.decode_tokens_per_second),
    ),
    gpu = h.gpus[0];
  $("metrics").innerHTML =
    metric(
      "Last decode speed",
      format(last?.decode_tokens_per_second),
      "tok/s",
      "Measured after the first token",
    ) +
    metric(
      "GPU memory",
      format(gpu?.used_gib),
      "GiB",
      gpu ? escapeHTML(gpu.name) : "Install eole[ui] for GPU telemetry",
    ) +
    metric(
      "GPU utilization",
      format(gpu?.utilization, 0),
      "%",
      gpu
        ? `GPU ${gpu.index} · ${format(gpu.power_w, 0)} W`
        : "Telemetry unavailable",
    ) +
    metric(
      "System memory",
      format(h.ram_used_gib),
      "GiB",
      h.ram_total_gib
        ? `${format(h.ram_total_gib)} GiB total · CPU ${format(h.cpu_percent, 0)}%`
        : "Install eole[ui] for host telemetry",
    );
  $("activity").innerHTML =
    '<span class="dot"></span><span>' +
    escapeHTML(
      state.active.length
        ? state.active
            .map(
              (r) =>
                ({
                  queued: "Queued",
                  preparing: "Preparing model / reading prompt",
                  generating: "Generating",
                  stopping: "Stopping after the current step",
                })[r.status] || r.status,
            )
            .join(" · ")
        : "Ready for your next idea",
    ) +
    "</span><small>" +
    escapeHTML(
      state.active.length
        ? `${state.active.length} active request${state.active.length > 1 ? "s" : ""}`
        : "Inference stays local",
    ) +
    "</small>";
  const speeds = state.recent
    .filter((r) => Number.isFinite(r.decode_tokens_per_second))
    .slice(0, 20)
    .reverse();
  const peak = Math.max(...speeds.map((r) => r.decode_tokens_per_second), 1);
  $("speed-chart").innerHTML = speeds.length
    ? speeds
        .map(
          (r) =>
            `<div class="speed-bar" style="height:${Math.max(5, (r.decode_tokens_per_second / peak) * 100)}px" title="${format(r.decode_tokens_per_second)} tok/s"><span>${format(r.decode_tokens_per_second, 0)}</span></div>`,
        )
        .join("")
    : '<div class="empty-chart">Your first response will start the story.</div>';
  $("gpu-details").innerHTML = h.gpus
    .map(
      (g) =>
        `<div class="gpu-card"><h3>GPU ${g.index} · ${escapeHTML(g.name)}</h3><div class="meter"><div style="width:${Math.min(100, (g.used_gib / g.total_gib) * 100)}%"></div></div><p>${format(g.used_gib)} / ${format(g.total_gib)} GiB VRAM · ${format(g.temperature, 0)} °C · ${format(g.power_w, 0)} W</p></div>`,
    )
    .join("");
  $("requests").innerHTML = state.recent.length
    ? state.recent
        .slice(0, 12)
        .map(
          (r) =>
            `<tr><td>${escapeHTML(r.model)}<small>${escapeHTML(r.task ? r.task + " · " + r.status : r.status)} · ${new Date(r.timestamp * 1000).toLocaleTimeString()}</small></td><td>${r.prompt_tokens ?? "—"}</td><td>${r.output_tokens ?? "—"}</td><td>${format(r.first_token_seconds, 2)} s</td><td>${format(r.decode_tokens_per_second)} t/s</td><td>${r.mtp_active ? format(r.mtp_acceptance * 100, 0) + "%" : "Off"}</td></tr>`,
        )
        .join("")
    : '<tr><td colspan="6" class="empty-cell">No requests yet. Start a conversation to see real measurements.</td></tr>';
}
function renderAPI() {
  const base = location.origin + "/v1";
  $("api-url").textContent = base;
  $("api-example").textContent =
    `curl ${base}/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify({ model: $("model").value, messages: [{ role: "user", content: "Explain binary search." }], temperature: 0, max_tokens: 128, stream: true })}'`;
  $("about").innerHTML = [
    ["Eole version", state.version],
    [
      "Compilation",
      state.compile_enabled ? "Mode " + state.compile_mode : "Disabled",
    ],
    ["Model", $("model").value],
    [
      "Configured MTP",
      selectedModel()?.mtp_configured
        ? "Enabled for eligible requests"
        : "Disabled",
    ],
    [
      "Hardware",
      state.hardware.gpus.map((g) => g.name).join(" + ") || "Not reported",
    ],
    ["Storage", "Browser-local conversations"],
  ]
    .map(
      ([label, value]) =>
        `<div><small>${label}</small>${escapeHTML(value)}</div>`,
    )
    .join("");
}
async function refreshState() {
  try {
    const response = await fetch("/ui/state");
    if (!response.ok) throw new Error();
    state = await response.json();
    const old = $("model").value;
    const signature = JSON.stringify(state.models.map((m) => m.id));
    if (refreshState.modelSignature !== signature) {
      refreshState.modelSignature = signature;
      $("model").innerHTML = state.models
        .map(
          (m) =>
            `<option value="${escapeHTML(m.id)}">${escapeHTML(m.id)}</option>`,
        )
        .join("");
    }
    if (state.models.some((m) => m.id === old)) $("model").value = old;
    $("connection-dot").classList.remove("offline");
    $("connection-text").textContent = "Connected to local server";
    $("version").textContent = "EOLE " + state.version;
    updateContext();
    renderMonitor();
    renderAPI();
    window.dispatchEvent(new CustomEvent("eole:state", { detail: state }));
  } catch {
    $("connection-dot").classList.add("offline");
    $("connection-text").textContent = "Server unavailable";
  }
}
function download(name, content, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([content], { type })),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function copy(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied to clipboard");
  } catch {
    toast("Clipboard unavailable in this browser. Select and copy the text.");
  }
}
function median(values) {
  const a = values.slice().sort((x, y) => x - y),
    mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}
async function runBenchmark() {
  if (busy) return;
  const payload = requestPayload(
    [
      {
        role: "user",
        content:
          "Write a detailed essay of at least 1500 words about George Orwell, his life, and his influence on literature.",
      },
    ],
    128,
  );
  if (Number(selectedModel()?.max_tokens) < 128)
    return toast(
      "This model requires an output limit of at least 128 for the benchmark.",
    );
  setBusy(true);
  benchmark = {
    model: payload.model,
    preset: payload.preset,
    thinking: payload.thinking,
    context: selectedModel()?.context_length,
    compile: state.compile_enabled,
    compile_mode: state.compile_mode,
    hardware: state.hardware,
    output_limit: 128,
    request: payload,
    eole_version: state.version,
    runs: [],
    timestamp: new Date().toISOString(),
  };
  $("benchmark-runs").innerHTML = "";
  $("benchmark-speed").innerHTML = "—<small>tokens / second</small>";
  $("benchmark-download").disabled = true;
  try {
    for (let i = 0; i < 4; i++) {
      if (stopRequested) throw new Error("Benchmark stopped.");
      $("benchmark-progress").textContent =
        i === 0
          ? "Warming all generation shapes…"
          : `Measuring request ${i} of 3…`;
      const stats = await consume(payload, () => {});
      if (stats.cancelled) throw new Error("Benchmark stopped.");
      if (i) {
        benchmark.runs.push(stats);
        const line = document.createElement("div");
        line.className = "benchmark-run";
        line.textContent = `Run ${i}: ${format(stats.decode_tokens_per_second)} tok/s · ${stats.output_tokens} tokens · ${format(stats.first_token_seconds, 2)} s first token`;
        $("benchmark-runs").append(line);
      }
    }
    const valid = benchmark.runs.filter(
      (r) =>
        r.output_tokens === 128 && Number.isFinite(r.decode_tokens_per_second),
    );
    if (valid.length !== 3)
      throw new Error(
        "Some requests stopped early; results are available but no fixed-length median is reported.",
      );
    benchmark.median_decode_tokens_per_second = median(
      valid.map((r) => r.decode_tokens_per_second),
    );
    $("benchmark-speed").innerHTML =
      format(benchmark.median_decode_tokens_per_second) +
      "<small>tokens / second</small>";
    $("benchmark-caption").textContent =
      `Median of 3 warmed requests · ${payload.model} · ${payload.preset} mode`;
    $("benchmark-progress").textContent =
      "Benchmark complete. Your results are ready.";
  } catch (error) {
    $("benchmark-progress").textContent = error.message;
    toast(error.message);
  } finally {
    setBusy(false);
    $("benchmark-download").disabled = !benchmark.runs.length;
    refreshState();
  }
}
document.addEventListener("click", (event) => {
  const target = event.target.closest("button");
  if (!target) return;
  if (target.dataset.continue) send("Continue from where you stopped.");
  if (target.dataset.view) switchView(target.dataset.view);
  if (target.dataset.prompt) {
    $("prompt").value = target.dataset.prompt;
    $("prompt").focus();
  }
  if (target.dataset.chat) {
    if (busy) return toast("Stop generation before switching chats.");
    selected = target.dataset.chat;
    renderMessages();
    renderHistory();
    updateContext();
    switchView("chat");
  }
  if (target.dataset.delete) {
    if (busy) return;
    chats = chats.filter((c) => c.id !== target.dataset.delete);
    if (selected === target.dataset.delete) selected = null;
    save();
    renderHistory();
    renderMessages();
  }
  if (target.dataset.copyCode)
    copy(decodeURIComponent(target.dataset.copyCode));
  if (target.dataset.copyMessage !== undefined)
    copy(
      current()?.messages[Number(target.dataset.copyMessage)]?.content || "",
    );
  if (target.dataset.editMessage !== undefined) {
    const index = Number(target.dataset.editMessage),
      chat = current(),
      message = chat?.messages[index];
    if (!message || busy) return;
    if (message.role === "user") {
      $("prompt").value = message.content;
      chat.messages = chat.messages.slice(0, index);
      renderMessages();
      save();
      $("prompt").focus();
    } else {
      const user = chat.messages[index - 1];
      chat.messages = chat.messages.slice(0, index - 1);
      send(user?.content || "");
    }
  }
});
$("send").onclick = () => send();
$("new-chat").onclick = freshChat;
$("prompt").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
};
async function stopGeneration() {
  stopRequested = true;
  if (activeRequest) {
    $("run-state").textContent = "Stopping after the current GPU step…";
    await fetch(`/ui/cancel/${activeRequest}`, { method: "POST" }).catch(() =>
      toast("Could not reach server"),
    );
  } else $("run-state").textContent = "Stopping as soon as the request starts…";
}
$("stop").onclick = stopGeneration;
$("benchmark-stop").onclick = stopGeneration;
$("settings-toggle").onclick = () => {
  $("settings").hidden = !$("settings").hidden;
};
$("theme").onclick = () => {
  document.body.classList.toggle("dark");
  try {
    localStorage.setItem(
      "eole.theme",
      document.body.classList.contains("dark") ? "dark" : "light",
    );
  } catch {}
};
$("model").onchange = updateContext;
$("preset").onchange = () => {
  $("temperature").disabled = $("preset").value === "fast";
};
$("temperature").disabled = true;
$("attach").onclick = () => $("file").click();
$("file").onchange = async () => {
  const file = $("file").files[0];
  if (!file) return;
  if (file.size > 200000) {
    toast("Please attach a text file smaller than 200 KB.");
    return;
  }
  attached = { name: file.name, content: await file.text() };
  $("attachment").hidden = false;
  $("attachment").textContent = "Attached: " + file.name;
  const button = document.createElement("button");
  button.textContent = "×";
  button.onclick = () => {
    attached = null;
    $("attachment").hidden = true;
  };
  $("attachment").append(button);
  $("file").value = "";
};
$("export-chat").onclick = () => {
  const chat = current();
  if (!chat) return toast("Open a conversation first.");
  download(
    "eole-conversation.md",
    chat.messages.map((m) => `## ${m.role}\n\n${m.content}\n`).join("\n"),
  );
};
$("benchmark-run").onclick = runBenchmark;
$("benchmark-download").onclick = () =>
  download(
    "eole-benchmark.json",
    JSON.stringify(benchmark, null, 2),
    "application/json",
  );
$("copy-url").onclick = () => copy($("api-url").textContent);
$("copy-example").onclick = () => copy($("api-example").textContent);
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "k") {
    e.preventDefault();
    freshChat();
  }
});
renderHistory();
refreshState();
setInterval(refreshState, 2500);

$("mobile-history").onclick = () => {
  document.body.classList.toggle("history-open");
  $("mobile-history").setAttribute(
    "aria-expanded",
    document.body.classList.contains("history-open"),
  );
};
