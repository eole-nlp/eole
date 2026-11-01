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
  chatImages = [],
  imageUploads = 0,
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
  if (busy || imageUploads)
    return toast(
      "Finish the active generation or image upload before starting a new chat.",
    );
  selected = null;
  chatImages = [];
  renderImageAttachments();
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
    ocr: "Documents (OCR)",
    knowledge: "Knowledge (RAG)",
    monitor: "Monitor",
    benchmark: "Benchmark",
    connect: "API & setup",
  }[view];
  window.dispatchEvent(new CustomEvent("eole:view", { detail: view }));
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
  return `<div class="response-stats"><span><strong>${format(s.decode_tokens_per_second)}</strong> tok/s decode</span><span>${format(s.first_token_seconds, 2)} s first token</span><span>${s.output_tokens ?? "—"} output tokens</span><span>${speculationLabel(s) !== "Off" ? speculationLabel(s) + " active · " + format((s.mtp_acceptance ?? 0) * 100, 0) + "% acceptance" : "Speculation off"}</span>${s.warmup_seconds > 1 ? `<span>${format(s.warmup_seconds)} s warmup</span>` : ""}${s.cancelled ? "<span>Stopped</span>" : s.finish_reason === "length" ? "<span>Output limit reached</span>" : ""}</div>`;
}
function messageHTML(m, index) {
  const assistant = m.role === "assistant";
  return `<article class="message ${assistant ? "assistant" : "user"}"><div class="message-header"><span class="avatar">${assistant ? "≋" : "Y"}</span>${assistant ? "Eole" : "You"}</div>${m.reasoning ? `<details class="reasoning" ${busy && index === current()?.messages.length - 1 ? "open" : ""}><summary>Thinking</summary><pre>${escapeHTML(m.reasoning)}</pre></details>` : ""}<div class="message-body">${m.error ? `<div class="error">${escapeHTML(m.error)}</div>` : m.content ? markdown(m.content) : busy ? '<span class="pending muted">Preparing the model / reading your prompt…</span>' : ""}</div>${imageThumbnails(m.images)}${statsHTML(m.stats)}${!busy ? `<div class="message-actions"><button data-copy-message="${index}">Copy</button><button data-edit-message="${index}">${assistant ? "Regenerate" : "Edit & resend"}</button>${assistant && m.stats?.finish_reason === "length" && index === current()?.messages.length - 1 ? '<button data-continue="true">Continue →</button>' : ""}</div>` : ""}</article>`;
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
  $("head-quantization").disabled = value;
  $("max-tokens").disabled = value;
  $("preset").disabled = value;
  $("thinking").disabled = value;
  $("prompt").disabled = value;
  $("run-state").textContent = value
    ? "Preparing model / reading prompt…"
    : "Ready when you are";
  if (!value) activeRequest = null;
  updateSpeculation();
}
const headPreferences = new Map();
const speculationPreferences = new Map();
const draftPreferences = new Map();
function speculationLabel(stats) {
  const mode = stats.speculation_mode ?? (stats.mtp_active ? "mtp" : "off");
  return mode === "dflash2" ? "DFlash2" : mode === "mtp" ? "MTP" : "Off";
}
function updateSpeculation() {
  const model = selectedModel();
  if (!model) return;
  const options = model.speculation ?? {
    default: model.mtp_configured ? "mtp" : "off",
    mtp_available: model.mtp_configured,
  };
  const select = $("speculation");
  select.querySelector('[value="mtp"]').disabled = !options.mtp_available;
  select.querySelector('[value="dflash2"]').disabled =
    !options.dflash2_available;
  const preferred =
    speculationPreferences.get(model.id) ?? options.default ?? "off";
  select.value =
    preferred === "dflash2" && !options.dflash2_available
      ? options.mtp_available
        ? "mtp"
        : "off"
      : preferred === "mtp" && !options.mtp_available
        ? "off"
        : preferred;
  $("draft-precision").value =
    draftPreferences.get(model.id) ?? options.dflash2_quantization ?? "none";
  $("draft-precision-label").hidden = select.value !== "dflash2";
  const eligible = $("preset").value === "fast";
  select.disabled = busy || !eligible;
  $("draft-precision").disabled = busy || !eligible;
  $("speculation-note").textContent = !eligible
    ? "Creative mode uses ordinary decoding; leaving DFlash2 reloads the model. Your speculation choice is kept for Fast mode."
    : select.value === "dflash2"
      ? `DFlash2 · ${options.dflash2_drafts ?? "—"} draft tokens · text-only. Image conversations use ordinary decoding and reload a DFlash2 engine. Drafter precision is separate from LM head precision.`
      : options.dflash2_available
        ? "DFlash2 is available. Switching to or from it reloads this model."
        : (options.dflash2_reason ??
          "DFlash2 is not configured for this model.");
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
      ...(m.images?.length ? { images: m.images } : {}),
      ...(m.role === "assistant" && m.reasoning
        ? { reasoning_content: m.reasoning }
        : {}),
    })),
    max_tokens: maxTokens,
    preset: $("preset").value,
    thinking: $("thinking").checked,
    vocabulary_quantization: $("head-quantization").value,
    speculation: $("speculation").value,
    dflash2_quantization: $("draft-precision").value,
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
  if (imageUploads) return toast("Wait for the image preview before sending.");
  text = (text ?? $("prompt").value).trim();
  if (!text && !attached && !chatImages.length) return;
  const imageCount =
    (current()?.messages || []).reduce(
      (n, m) => n + (m.images?.length || 0),
      0,
    ) + chatImages.length;
  if (imageCount > 4)
    return toast(
      "Use at most four images per conversation. Start a new chat for more.",
    );
  if (imageCount && !selectedModel()?.images_supported)
    return toast(
      "Select an image-capable model before sending this conversation.",
    );
  if (!$("model").value) return toast("Wait for a model to become available.");
  if (attached)
    text += `\n\nAttached file: ${attached.name}\n\n${attached.content}`;
  if (!current()) {
    const chat = {
      id: crypto.randomUUID(),
      title: text.slice(0, 45) || chatImages[0]?.name || "Image conversation",
      messages: [],
      created: Date.now(),
    };
    chats.unshift(chat);
    selected = chat.id;
  }
  const chat = current();
  chat.messages.push({
    role: "user",
    content: text,
    ...(chatImages.length ? { images: chatImages.slice() } : {}),
  });
  chatImages = [];
  renderImageAttachments();
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
    `${model.weight_bits ? "INT" + model.weight_bits + " · " : ""}${model.dtype.replace("torch.", "")} compute · ${model.vocabulary_quantization === "none" ? "native" : (model.vocabulary_quantization?.toUpperCase() ?? "native")} head · ${model.context_length ? Number(model.context_length).toLocaleString() + " context" : "context automatic"} · ${state.compile_enabled ? "compiled mode " + state.compile_mode : "eager"}`;
  $("head-quantization").value =
    headPreferences.get(model.id) ?? model.vocabulary_quantization ?? "int4";
  updateSpeculation();
  benchmarkChoices();
  const max = Number(model.max_tokens) || 16384;
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
            `<tr><td>${escapeHTML(r.model)}<small>${escapeHTML(r.task ? r.task + " · " + r.status : r.status)} · ${new Date(r.timestamp * 1000).toLocaleTimeString()}</small></td><td>${r.prompt_tokens ?? "—"}</td><td>${r.output_tokens ?? "—"}</td><td>${format(r.first_token_seconds, 2)} s</td><td>${format(r.decode_tokens_per_second)} t/s</td><td>${speculationLabel(r) !== "Off" ? speculationLabel(r) + " · " + format(r.mtp_acceptance * 100, 0) + "%" : "Off"}</td></tr>`,
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
      "Speculation",
      $("speculation").value === "dflash2"
        ? "DFlash2 · text-only greedy"
        : $("speculation").value === "mtp"
          ? "MTP · eligible requests"
          : "Off",
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
    const chatModels = state.models.filter((m) => m.chat_available !== false);
    const signature = JSON.stringify(
      chatModels.map((m) => [m.id, m.images_supported]),
    );
    if (refreshState.modelSignature !== signature) {
      refreshState.modelSignature = signature;
      $("model").innerHTML = chatModels
        .map(
          (m) =>
            `<option value="${escapeHTML(m.id)}">${escapeHTML(m.id)}${m.images_supported ? " · Vision" : ""}</option>`,
        )
        .join("");
    }
    $("model").value = EoleUI.chooseModel(state, "chat", chatModels, old);
    if (!refreshState.initialView && state.workspace?.default_view) {
      refreshState.initialView = true;
      switchView(state.workspace.default_view);
    }
    $("connection-dot").classList.remove("offline");
    const preparing = state.active.find((row) =>
      ["preparing", "loading"].includes(row.status),
    );
    $("connection-text").textContent =
      state.model_activity?.message ||
      (preparing
        ? `Preparing ${preparing.model}…`
        : "Connected to local server");
    $("version").textContent = "EOLE " + state.version;
    updateContext();
    benchmarkChoices();
    updateImageInput();
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
function benchmarkChoices() {
  if (!$("benchmark-profiles").querySelector("input")) {
    for (const [kind, values] of [
      ["workloads", EoleBenchmark.workloads],
      ["profiles", EoleBenchmark.profiles],
    ]) {
      $("benchmark-" + kind).insertAdjacentHTML(
        "beforeend",
        values
          .map(
            (item) =>
              `<label class="check"><input type="checkbox" value="${item.id}" checked /> ${escapeHTML(item.label)}</label>`,
          )
          .join(""),
      );
    }
  }
  const model = selectedModel();
  for (const input of $("benchmark-profiles").querySelectorAll("input")) {
    const profile = EoleBenchmark.profiles.find((p) => p.id === input.value);
    const supported = EoleBenchmark.available(profile, model);
    input.disabled = busy || !supported;
    input.parentElement.title = supported
      ? ""
      : "Not configured for the selected model";
  }
  $("benchmark-tokens").max = Math.min(4096, Number(model?.max_tokens) || 4096);
}
function renderBenchmark() {
  const rows = benchmark?.results ?? [];
  $("benchmark-runs").innerHTML =
    rows
      .map(
        (row) =>
          `<tr><td>${escapeHTML(row.model)}<br><small>${escapeHTML(row.workload.label)}</small></td><td>${escapeHTML(row.profile.label)}</td><td>${row.output_limit}</td><td><strong>${format(row.decode_tokens_per_second)}</strong></td><td>${format(row.first_token_seconds, 2)} s</td><td>${row.acceptance == null ? "—" : format(row.acceptance * 100, 1) + "%"}</td><td title="${escapeHTML(row.error ?? "")}">${row.valid_runs ?? 0}/${row.repeats} · ${escapeHTML(row.status)}</td></tr>`,
      )
      .join("") ||
    '<tr><td colspan="7" class="empty-cell">No measurements yet.</td></tr>';
  $("benchmark-download").disabled = !rows.length;
}
async function runBenchmark() {
  if (busy) return;
  const model = selectedModel();
  if (!model)
    return toast("Choose an available Chat model before benchmarking.");
  const selected = (kind) =>
    Array.from($("benchmark-" + kind).querySelectorAll("input:checked")).map(
      (input) => input.value,
    );
  const workloads = EoleBenchmark.workloads.filter((p) =>
    selected("workloads").includes(p.id),
  );
  const profiles = EoleBenchmark.profiles.filter(
    (p) =>
      selected("profiles").includes(p.id) && EoleBenchmark.available(p, model),
  );
  const outputLimit = Number($("benchmark-tokens").value),
    repeats = Number($("benchmark-repeats").value);
  if (!workloads.length || !profiles.length)
    return toast("Select at least one workload and available profile.");
  if (
    !Number.isInteger(outputLimit) ||
    outputLimit < 128 ||
    outputLimit > Math.min(4096, Number(model?.max_tokens) || 4096) ||
    !Number.isInteger(repeats) ||
    repeats < 1 ||
    repeats > 5
  )
    return toast("Choose a supported output cap (128–4096) and 1–5 repeats.");
  const session = {
    id: new Date().toISOString(),
    model: model.id,
    context: model.context_length,
    compile: state.compile_enabled,
    compile_mode: state.compile_mode,
    hardware: state.hardware,
    eole_version: state.version,
    output_limit: outputLimit,
    repeats,
    draft_tokens: model.speculation?.dflash2_drafts,
    native_mtp_draft_tokens: model.mtp_draft_tokens,
    compute_dtype: model.dtype,
    fuse_kvq: model.fuse_kvq,
    fuse_gate: model.fuse_gate,
  };
  benchmark ??= { schema_version: 2, sessions: [], results: [] };
  benchmark.sessions.push(session);
  setBusy(true);
  benchmarkChoices();
  $("benchmark-tokens").disabled = $("benchmark-repeats").disabled = true;
  $("benchmark-workloads").disabled = true;
  $("benchmark-speed").innerHTML = "—<small>tokens / second</small>";
  $("benchmark-caption").textContent =
    "Comparing fixed workloads. Preparation can take several minutes.";
  try {
    for (const profile of profiles) {
      for (const workload of workloads) {
        if (stopRequested)
          throw new Error(
            "Benchmark stopped. Completed and partial results are retained.",
          );
        const payload = {
          model: model.id,
          messages: workload.messages,
          max_tokens: outputLimit,
          preset: "fast",
          thinking: false,
          temperature: 1,
          speculation: profile.speculation,
          vocabulary_quantization: profile.vocabulary_quantization,
          dflash2_quantization: profile.dflash2_quantization,
        };
        const row = {
          session: session.id,
          model: model.id,
          workload,
          profile,
          output_limit: outputLimit,
          repeats,
          request: payload,
          runs: [],
          status: "warming",
        };
        benchmark.results.push(row);
        renderBenchmark();
        try {
          for (let i = 0; i <= repeats; i++) {
            if (stopRequested) throw new Error("Benchmark stopped.");
            $("benchmark-progress").textContent =
              `${profile.label} · ${workload.label} · ${i ? "measuring " + i + "/" + repeats : "loading / warming"}`;
            const stats = await consume(payload, () => {});
            if (i) row.runs.push(stats);
            else row.warmup = stats;
            Object.assign(
              row,
              EoleBenchmark.summarize(
                row.runs,
                outputLimit,
                profile.speculation,
                repeats,
              ),
            );
            renderBenchmark();
            if (stats.cancelled) stopRequested = true;
            if (stopRequested) throw new Error("Benchmark stopped.");
          }
        } catch (error) {
          row.status = stopRequested ? "cancelled" : "error";
          row.error = error.message;
          renderBenchmark();
          if (stopRequested) throw error;
          if (/CUDA.*out of memory/i.test(error.message))
            throw new Error(
              "Benchmark stopped: GPU memory is full. Free memory before retrying. Details are retained in the report.",
            );
        }
      }
    }
    const complete = benchmark.results.filter(
      (r) => r.session === session.id && r.status === "complete",
    );
    const best = complete.reduce(
      (a, b) =>
        !a || b.decode_tokens_per_second > a.decode_tokens_per_second ? b : a,
      null,
    );
    $("benchmark-speed").innerHTML =
      format(best?.decode_tokens_per_second) +
      "<small>tokens / second · best completed row</small>";
    $("benchmark-caption").textContent = best
      ? `${best.workload.label} · ${best.profile.label}`
      : "No complete fixed-length measurements. See individual rows and report.";
    $("benchmark-progress").textContent =
      `Comparison finished: ${complete.length}/${profiles.length * workloads.length} complete rows. Other rows retain early stops or errors.`;
  } catch (error) {
    $("benchmark-progress").textContent = error.message;
  } finally {
    activeRequest = null;
    setBusy(false);
    $("benchmark-tokens").disabled = $("benchmark-repeats").disabled = false;
    $("benchmark-workloads").disabled = false;
    benchmarkChoices();
    renderBenchmark();
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
    if (busy || imageUploads)
      return toast("Finish generation or image upload before switching chats.");
    selected = target.dataset.chat;
    chatImages = [];
    renderImageAttachments();
    attached = null;
    $("attachment").hidden = true;
    renderMessages();
    renderHistory();
    updateContext();
    benchmarkChoices();
    updateImageInput();
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
      chatImages = (message.images || []).slice();
      renderImageAttachments();
      chat.messages = chat.messages.slice(0, index);
      renderMessages();
      save();
      $("prompt").focus();
    } else {
      const user = chat.messages[index - 1];
      chat.messages = chat.messages.slice(0, index - 1);
      chatImages = (user?.images || []).slice();
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
  $("settings-toggle").setAttribute(
    "aria-expanded",
    String(!$("settings").hidden),
  );
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
$("model").onchange = () => {
  EoleUI.rememberModel("chat", $("model").value);
  updateContext();
  updateImageInput();
};
$("head-quantization").onchange = () => {
  headPreferences.set($("model").value, $("head-quantization").value);
};
$("speculation").onchange = () => {
  speculationPreferences.set($("model").value, $("speculation").value);
  updateSpeculation();
};
$("draft-precision").onchange = () => {
  draftPreferences.set($("model").value, $("draft-precision").value);
};
$("preset").onchange = () => {
  $("temperature").disabled = $("preset").value === "fast";
  updateSpeculation();
};
$("temperature").disabled = true;
$("attach").onclick = () => $("file").click();
$("file").onchange = async () => {
  if (busy)
    return toast("Wait for generation to finish before attaching files.");
  const file = $("file").files[0];
  if (!file) return;
  if (file.type.startsWith("image/")) {
    await addChatImages(Array.from($("file").files));
    $("file").value = "";
    return;
  }
  if (!/\.(txt|md|py|js|ts|json|csv|yaml|yml|html|css|log)$/i.test(file.name)) {
    toast("Attach a supported text file or a JPEG, PNG or WebP image.");
    $("file").value = "";
    return;
  }
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

function imageThumbnails(images = []) {
  return images.length
    ? `<div class="chat-images">${images
        .filter((image) => /^[A-Za-z0-9+/=]+$/.test(image.data))
        .map(
          (image) =>
            `<img src="data:image/jpeg;base64,${image.data}" alt="${escapeHTML(image.name)}" title="${escapeHTML(image.name)}" />`,
        )
        .join("")}</div>`
    : "";
}
function renderImageAttachments() {
  $("chat-image-attachments").innerHTML = chatImages
    .map(
      (image, index) =>
        `<div class="chat-image-preview"><img src="data:image/jpeg;base64,${image.data}" alt="${escapeHTML(image.name)}" /><button type="button" data-remove-image="${index}" aria-label="Remove ${escapeHTML(image.name)}">×</button></div>`,
    )
    .join("");
}
function updateImageInput() {
  const enabled = Boolean(selectedModel()?.images_supported);
  $("image-hint").textContent = enabled
    ? "Drop images here · up to 4 per chat"
    : "Text attachments · select a vision model for images";
  $("file").accept =
    ".txt,.md,.py,.js,.ts,.json,.csv,.yaml,.yml,.html,.css,.log" +
    (enabled ? ",image/png,image/jpeg,image/webp" : "");
}
async function addChatImages(files) {
  if (busy)
    return toast("Wait for generation to finish before attaching images.");
  if (!selectedModel()?.images_supported)
    return toast("Select an image-capable model to attach images.");
  for (const file of files) {
    const count =
      chatImages.length +
      (current()?.messages || []).reduce(
        (n, m) => n + (m.images?.length || 0),
        0,
      );
    if (count >= 4) {
      toast("Use at most four images per conversation.");
      break;
    }
    if (
      !["image/jpeg", "image/png", "image/webp"].includes(file.type) ||
      file.size > 10000000
    ) {
      toast("Use JPEG, PNG or WebP images smaller than 10 MB.");
      continue;
    }
    try {
      imageUploads++;
      const bitmap = await createImageBitmap(file);
      if (bitmap.width * bitmap.height > 25000000) {
        bitmap.close();
        throw new Error("Images must be smaller than 25 megapixels.");
      }
      const scale = Math.min(1, 1024 / Math.max(bitmap.width, bitmap.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(bitmap.width * scale));
      canvas.height = Math.max(1, Math.round(bitmap.height * scale));
      const context = canvas.getContext("2d");
      context.fillStyle = "white";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      bitmap.close();
      const latestCount =
        chatImages.length +
        (current()?.messages || []).reduce(
          (n, m) => n + (m.images?.length || 0),
          0,
        );
      if (latestCount >= 4)
        throw new Error("Use at most four images per conversation.");
      chatImages.push({
        name: file.name.slice(0, 200),
        data: canvas.toDataURL("image/jpeg", 0.9).split(",")[1],
      });
      renderImageAttachments();
    } catch (error) {
      toast(error.message || "Could not read this image.");
    } finally {
      imageUploads--;
    }
  }
}
$("chat-image-attachments").onclick = (event) => {
  const button = event.target.closest("[data-remove-image]");
  if (button) {
    chatImages.splice(Number(button.dataset.removeImage), 1);
    renderImageAttachments();
  }
};
const chatComposer = document.querySelector("#view-chat .composer");
chatComposer.addEventListener("dragover", (event) => {
  event.preventDefault();
  chatComposer.classList.add("image-dragover");
});
chatComposer.addEventListener("dragleave", () =>
  chatComposer.classList.remove("image-dragover"),
);
chatComposer.addEventListener("drop", async (event) => {
  event.preventDefault();
  chatComposer.classList.remove("image-dragover");
  await addChatImages(Array.from(event.dataTransfer.files));
});
$("prompt").addEventListener("paste", async (event) => {
  const files = Array.from(event.clipboardData.files).filter((file) =>
    file.type.startsWith("image/"),
  );
  if (files.length) {
    event.preventDefault();
    await addChatImages(files);
  }
});
