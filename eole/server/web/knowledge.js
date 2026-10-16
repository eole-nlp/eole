"use strict";
(() => {
  const $ = (id) => document.getElementById(id),
    escape = EoleUI.escape;
  let token = null,
    collections = [],
    collection = "",
    busy = false,
    requestId = null,
    stop = false;
  let turns = [],
    sources = [],
    models = [],
    info = null;
  const root = "/ui/knowledge";
  const status = (text, error = false) => {
    $("kg-status").textContent = text;
    $("kg-status").classList.toggle("knowledge-error", error);
  };
  const headers = () => ({
    "Content-Type": "application/json",
    "X-Eole-Knowledge-Session": token || "",
  });
  const request = async (path, options = {}) => {
    const response = await fetch(root + path, {
      ...options,
      headers: headers(),
    });
    const value = await response.json();
    if (!response.ok)
      throw new Error(
        typeof value.detail === "string" ? value.detail : "Request rejected",
      );
    return value;
  };
  function setBusy(value) {
    busy = value;
    for (const id of [
      "kg-collection",
      "kg-upload",
      "kg-new",
      "kg-delete",
      "kg-create",
      "kg-model",
      "kg-clear",
      "kg-reset",
      "kg-ask",
      "kg-question",
    ])
      $(id).disabled = value;
    $("kg-stop").hidden = !value;
    $("kg-ask").hidden = value;
    $("kg-documents")
      .querySelectorAll("button")
      .forEach((button) => (button.disabled = value));
  }
  function renderDocuments() {
    const current = collections.find((item) => item.id === collection);
    $("kg-documents").innerHTML = current?.documents.length
      ? current.documents
          .map(
            (doc) =>
              `<div class="knowledge-document"><span>▤</span><div><strong>${escape(doc.name)}</strong><small>${doc.chunks} indexed passages</small></div><button data-kg-remove="${escape(doc.id)}" aria-label="Remove ${escape(doc.name)}">×</button></div>`,
          )
          .join("")
      : '<p class="small muted">Your collection is empty.</p>';
  }
  async function refreshCollections() {
    collections = (await request("/collections")).collections;
    if (!collections.some((item) => item.id === collection))
      collection = collections[0]?.id || "";
    $("kg-collection").innerHTML = collections
      .map(
        (item) =>
          `<option value="${escape(item.id)}">${escape(item.name)}</option>`,
      )
      .join("");
    $("kg-collection").value = collection;
    renderDocuments();
  }
  function renderSources() {
    $("kg-sources").innerHTML = sources.length
      ? sources
          .map(
            (source) =>
              `<details class="knowledge-source" id="kg-source-${source.id}" ${source.id === 1 ? "open" : ""}><summary><span class="knowledge-reference">[${source.id}]</span><strong>${escape(source.name)}</strong><small>${source.page ? `Page ${source.page} · ` : ""}Passage ${source.chunk}</small></summary><p>${escape(source.text)}</p></details>`,
          )
          .join("")
      : '<p class="muted">Retrieved passages appear here.</p>';
  }
  function citationHTML(text, evidence) {
    return markdown(text).replace(/\[(\d+)\]/g, (match, id) =>
      evidence.some((source) => source.id === Number(id))
        ? `<button class="knowledge-citation" data-kg-cite="${id}">[${id}]</button>`
        : match,
    );
  }
  function metricsHTML(stats) {
    if (!stats) return "";
    const seconds = (value) =>
      Number.isFinite(value) ? value.toFixed(2) + " s" : "—";
    return `<div class="knowledge-metrics"><span><strong>${seconds(stats.retrieval_seconds)}</strong> retrieval</span><span><strong>${Number.isFinite(stats.decode_tokens_per_second) ? stats.decode_tokens_per_second.toFixed(1) : "—"}</strong> tok/s decode</span><span>${seconds(stats.first_token_seconds)} first token</span><span>${stats.output_tokens ?? "—"} output tokens</span><span>MTP ${stats.mtp_active ? "on" : "off"}</span>${stats.warmup_seconds > 0.1 ? `<span>${seconds(stats.warmup_seconds)} warmup</span>` : ""}<span>${stats.finish_reason === "length" ? "Output limit reached" : stats.cancelled ? "Stopped" : "Complete"}</span></div>`;
  }
  function renderMessages() {
    $("kg-messages").innerHTML = turns.length
      ? turns
          .map(
            (turn, index) =>
              `<article class="knowledge-turn ${turn.role}"><div class="knowledge-turn-label">${turn.role === "user" ? "YOU" : "EOLE · WITH SOURCES"}</div><div class="message-body">${turn.role === "assistant" ? citationHTML(turn.content, turn.sources || []) : escape(turn.content)}</div>${turn.error ? `<p class="knowledge-error">${escape(turn.error)}</p>` : ""}${turn.role === "assistant" && turn.sources?.length ? `<button class="quiet-button" data-kg-evidence="${index}">Inspect ${turn.sources.length} passages</button>` : ""}${metricsHTML(turn.stats)}</article>`,
          )
          .join("")
      : '<div class="knowledge-empty"><span>▧</span><h2>Start with a source.</h2><p>Import a document, then ask a question.<br/>Answers link back to the passages they use.</p></div>';
    $("kg-messages").scrollTop = $("kg-messages").scrollHeight;
  }
  function clearConversation() {
    turns = [];
    sources = [];
    renderMessages();
    renderSources();
  }
  async function session(reset = false) {
    if (reset && token) {
      await fetch(root + "/session", { method: "DELETE", headers: headers() });
      token = null;
    }
    if (!reset) {
      try {
        token = sessionStorage.getItem("eole.knowledge.session");
      } catch {}
    }
    if (token && !reset) {
      try {
        await refreshCollections();
        return;
      } catch {
        token = null;
      }
    }
    token = (await request("/session", { method: "POST" })).token;
    try {
      sessionStorage.setItem("eole.knowledge.session", token);
    } catch {}
    collection = (
      await request("/collections", {
        method: "POST",
        body: JSON.stringify({ name: "My documents" }),
      })
    ).id;
    await refreshCollections();
  }
  async function events(response, onEvent) {
    let completed = false;
    try {
      for await (const event of EoleUI.events(response)) {
        if (event.type === "start") {
          requestId = event.request_id;
          if (stop) await cancel();
        }
        if (event.type === "error") throw new Error(event.message);
        if (event.type === "progress") status(event.message);
        if (event.type === "done") completed = true;
        onEvent(event);
      }
      if (!completed) throw new Error("The stream ended before completion.");
    } finally {
      if (!completed && requestId) await cancel();
      requestId = null;
    }
  }
  async function cancel() {
    stop = true;
    if (requestId) await fetch(`/ui/cancel/${requestId}`, { method: "POST" });
    status("Stopping after the current operation…");
  }
  async function upload(files) {
    if (busy || !token) return;
    if (!info?.available)
      return status(
        "Install the optional RAG dependencies: pip install 'eole[rag]'",
        true,
      );
    if (!collection) return status("Create a collection first.", true);
    setBusy(true);
    stop = false;
    try {
      for (const file of files) {
        if (stop) break;
        if (file.size > 4000000 || !/\.(pdf|docx|md|txt)$/i.test(file.name))
          throw new Error("Use PDF, DOCX, Markdown or text files up to 4 MB.");
        status(`Reading ${file.name}…`);
        const data = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result.split(",")[1]);
          reader.onerror = () => reject(new Error("Could not read this file"));
          reader.readAsDataURL(file);
        });
        const response = await fetch(
          `${root}/collections/${collection}/import`,
          {
            method: "POST",
            headers: headers(),
            body: JSON.stringify({ name: file.name, data }),
          },
        );
        await events(response, (event) => {
          if (event.type === "done" && event.cancelled) {
            status("Indexing stopped.");
            return;
          }
          if (event.type === "done")
            status(
              `Indexed ${event.document.name} · ${event.document.chunks} passages · ${event.indexing_seconds.toFixed(1)} s`,
            );
        });
        await refreshCollections();
      }
      clearConversation();
    } catch (error) {
      status(error.message, true);
      await refreshCollections().catch(() => {});
    } finally {
      setBusy(false);
      $("kg-file").value = "";
    }
  }
  async function ask() {
    if (busy) return;
    const question = $("kg-question").value.trim();
    if (!question) return;
    if (!collections.find((item) => item.id === collection)?.documents.length)
      return status("Import a document first.", true);
    if (!$("kg-model").value) return status("Select an answer model.", true);
    const history = turns
      .filter((turn) => !turn.error && turn.content)
      .slice(-6)
      .map(({ role, content }) => ({ role, content: content.slice(-12000) }));
    const reply = { role: "assistant", content: "", sources: [] };
    turns.push({ role: "user", content: question }, reply);
    $("kg-question").value = "";
    setBusy(true);
    stop = false;
    renderMessages();
    status("Retrieving relevant passages…");
    try {
      const response = await fetch(`${root}/collections/${collection}/ask`, {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({
          model: $("kg-model").value,
          question,
          history,
          max_tokens: Number($("kg-max-tokens").value),
        }),
      });
      let lastRender = 0;
      await events(response, (event) => {
        if (event.type === "sources") {
          sources = event.sources;
          reply.sources = sources.slice();
          renderSources();
          status(`Retrieved ${sources.length} passages · generating answer…`);
        }
        if (event.type === "delta" && event.kind === "text") {
          reply.content += event.text;
          if (performance.now() - lastRender > 60) {
            renderMessages();
            lastRender = performance.now();
          }
        }
        if (event.type === "done") {
          reply.stats = event.stats;
          status(
            event.stats.cancelled
              ? "Stopped."
              : "Answer ready. Select a reference to inspect its source.",
          );
        }
      });
    } catch (error) {
      reply.error = error.message;
      status(error.message, true);
    } finally {
      setBusy(false);
      renderMessages();
      $("kg-question").focus();
    }
  }
  $("kg-upload").onclick = () => $("kg-file").click();
  $("kg-file").onchange = () => upload(Array.from($("kg-file").files));
  $("kg-upload").ondragover = (event) => {
    event.preventDefault();
    $("kg-upload").classList.add("dragover");
  };
  $("kg-upload").ondragleave = () =>
    $("kg-upload").classList.remove("dragover");
  $("kg-upload").ondrop = (event) => {
    event.preventDefault();
    $("kg-upload").classList.remove("dragover");
    upload(Array.from(event.dataTransfer.files));
  };
  $("kg-new").onclick = () => {
    $("kg-new-form").hidden = !$("kg-new-form").hidden;
    $("kg-name").focus();
  };
  $("kg-create").onclick = async () => {
    try {
      collection = (
        await request("/collections", {
          method: "POST",
          body: JSON.stringify({
            name: $("kg-name").value.trim() || "Untitled collection",
          }),
        })
      ).id;
      await refreshCollections();
      clearConversation();
      $("kg-new-form").hidden = true;
    } catch (error) {
      status(error.message, true);
    }
  };
  $("kg-delete").onclick = async () => {
    if (
      !collection ||
      !confirm("Delete this temporary collection and its indexed documents?")
    )
      return;
    try {
      await request(`/collections/${collection}`, { method: "DELETE" });
      await refreshCollections();
      clearConversation();
    } catch (error) {
      status(error.message, true);
    }
  };
  $("kg-documents").onclick = async (event) => {
    const button = event.target.closest("[data-kg-remove]");
    if (!button || busy) return;
    try {
      await request(
        `/collections/${collection}/documents/${button.dataset.kgRemove}`,
        { method: "DELETE" },
      );
      await refreshCollections();
      clearConversation();
    } catch (error) {
      status(error.message, true);
    }
  };
  $("kg-collection").onchange = () => {
    collection = $("kg-collection").value;
    clearConversation();
    renderDocuments();
  };
  $("kg-clear").onclick = clearConversation;
  $("kg-stop").onclick = cancel;
  $("kg-ask").onclick = ask;
  $("kg-question").onkeydown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ask();
    }
  };
  $("kg-model").onchange = () =>
    EoleUI.rememberModel("knowledge", $("kg-model").value);
  $("kg-reset").onclick = async () => {
    try {
      await session(true);
      clearConversation();
      status("New temporary session ready.");
    } catch (error) {
      status(error.message, true);
    }
  };
  $("kg-messages").onclick = (event) => {
    const evidence = event.target.closest("[data-kg-evidence]");
    if (evidence) {
      sources = turns[Number(evidence.dataset.kgEvidence)].sources;
      renderSources();
    }
    const button = event.target.closest("[data-kg-cite]");
    if (!button) return;
    const article = button.closest("article");
    const index = Array.from($("kg-messages").children).indexOf(article);
    sources = turns[index].sources;
    renderSources();
    const source = $(`kg-source-${button.dataset.kgCite}`);
    if (source) {
      source.open = true;
      source.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  };
  window.addEventListener("eole:state", (event) => {
    models = event.detail.models.filter(
      (model) => model.chat_available !== false,
    );
    const current = $("kg-model").value;
    $("kg-model").innerHTML = models
      .map(
        (model) =>
          `<option value="${escape(model.id)}">${escape(model.id)}</option>`,
      )
      .join("");
    $("kg-model").value = EoleUI.chooseModel(
      event.detail,
      "knowledge",
      models,
      current,
    );
    const model = models.find((model) => model.id === $("kg-model").value);
    if (model) {
      $("kg-max-tokens").max = model.max_tokens;
      if (Number($("kg-max-tokens").value) > model.max_tokens)
        $("kg-max-tokens").value = model.max_tokens;
    }
  });
  async function initialize() {
    try {
      info = await request("/info");
      $("kg-embedding").textContent =
        `${info.embedding_model} · text only · CPU`;
      if (!info.available) {
        status("To enable Knowledge, install: pip install 'eole[rag]'", true);
        return;
      }
      status("Open Knowledge to prepare your temporary library.");
    } catch (error) {
      status(error.message, true);
    }
  }
  let initialization = null;
  async function ensureSession() {
    if (!info?.available || token) return;
    if (!initialization)
      initialization = session()
        .then(() =>
          status("Your library is ready. Import documents to get started."),
        )
        .catch((error) => status(error.message, true))
        .finally(() => {
          initialization = null;
        });
    await initialization;
  }
  window.addEventListener("eole:view", (event) => {
    if (event.detail === "knowledge") ensureSession();
  });
  initialize().then(() => {
    if (!$("view-knowledge").hidden) ensureSession();
  });
})();
