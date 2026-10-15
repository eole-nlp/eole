/* Translation is a separate workflow; only transport and workspace chrome are shared. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id),
    esc = EoleUI.escape;
  let models = [],
    running = false,
    requestId = null,
    stop = false,
    last = null,
    files = [],
    selectedFile = -1,
    entries = [],
    preferences = {},
    initialized = false;
  try {
    entries = JSON.parse(localStorage.getItem("eole.translations") || "[]");
    if (!Array.isArray(entries)) entries = [];
    preferences = JSON.parse(
      localStorage.getItem("eole.translation.settings") || "{}",
    );
  } catch {}
  for (const [id, key] of [
    ["tr-context", "context"],
    ["tr-glossary", "glossary"],
    ["tr-tone", "tone"],
  ])
    if (preferences[key]) $(id).value = preferences[key];
  const status = (text, error = false) => {
    $("tr-status").textContent = text;
    $("tr-status").classList.toggle("translation-error", error);
  };
  const notify = (text) => {
    status(text, true);
  };
  const selectedModel = () => models.find((m) => m.id === $("tr-model").value);
  const saveHistory = () => {
    try {
      localStorage.setItem("eole.translations", JSON.stringify(entries));
    } catch {
      notify(
        "Browser storage is full. Export your history before clearing old entries.",
      );
    }
  };
  const savePreferences = () => {
    try {
      localStorage.setItem(
        "eole.translation.settings",
        JSON.stringify({
          context: $("tr-context").value,
          glossary: $("tr-glossary").value,
          tone: $("tr-tone").value,
        }),
      );
    } catch {}
  };
  function setMode(mode) {
    $("tr-saved").hidden = mode !== "saved";
    $("tr-workbench").hidden = mode === "saved";
    $("tr-files").hidden = mode !== "files";
    document
      .querySelectorAll("[data-tr-mode]")
      .forEach((b) => b.classList.toggle("active", b.dataset.trMode === mode));
    if (mode === "saved") renderHistory();
  }
  document
    .querySelectorAll("[data-tr-mode]")
    .forEach((b) => (b.onclick = () => setMode(b.dataset.trMode)));
  function updateLanguages() {
    const model = selectedModel();
    if (!model) return;
    const caps = model.translation,
      raw = caps.prompt_style === "raw";
    const oldSource = $("tr-source").value,
      oldTarget = $("tr-target").value;
    const options = caps.languages
      .map((l) => `<option value="${esc(l)}">${esc(l)}</option>`)
      .join("");
    $("tr-source").innerHTML =
      '<option value="auto">Auto · infer from text</option>' + options;
    $("tr-target").innerHTML = options;
    $("tr-source").value = raw
      ? caps.source
      : caps.languages.includes(oldSource)
        ? oldSource
        : "auto";
    $("tr-target").value =
      caps.languages.includes(oldTarget) && !raw ? oldTarget : caps.target;
    if (!$("tr-target").value) $("tr-target").selectedIndex = 0;
    $("tr-limit").max = model.max_tokens || 4096;
    $("tr-limit").value = Math.min(
      Number($("tr-limit").value) || 4096,
      model.max_tokens || 4096,
    );
    for (const id of [
      "tr-source",
      "tr-target",
      "tr-tone",
      "tr-context",
      "tr-glossary",
      "tr-swap",
    ])
      $(id).disabled = running || raw;
    $("tr-model-note").textContent = raw
      ? "Native encoder-decoder model · fixed language direction · complete segments appear as they finish."
      : "Paragraphs stay in order. Auto infers the source language. Your settings guide tone and terminology.";
  }
  window.addEventListener("eole:state", (e) => {
    const next = e.detail.models.filter(
      (m) => m.translation?.available !== false && m.translation,
    );
    if (
      JSON.stringify(models.map((m) => [m.id, m.translation])) !==
      JSON.stringify(next.map((m) => [m.id, m.translation]))
    ) {
      const old = $("tr-model").value;
      models = next;
      $("tr-model").innerHTML = models
        .map((m) => `<option value="${esc(m.id)}">${esc(m.id)}</option>`)
        .join("");
      $("tr-model").value = models.some((m) => m.id === old)
        ? old
        : models.find((m) => m.translation.preferred)?.id ||
          models[0]?.id ||
          "";
      updateLanguages();
    } else models = next;
    if (!initialized && models.length) {
      initialized = true;
      if (models.some((m) => m.translation.preferred)) switchView("translate");
    }
  });
  $("tr-model").onchange = updateLanguages;
  function count() {
    const text = $("tr-input").value;
    $("tr-count").textContent =
      `${text.length.toLocaleString()} characters · ${text.trim() ? text.trim().split(/\s+/).length : 0} words`;
    $("tr-output-count").textContent =
      `${$("tr-output").value.length.toLocaleString()} characters · editable`;
  }
  $("tr-input").oninput = () => {
    count();
    if (last) status("Source changed. Translate again to update the result.");
  };
  $("tr-output").oninput = () => {
    count();
    $("tr-result-label").textContent = "Edited locally";
    if (last) {
      last.text = $("tr-output").value;
      last.edited = true;
    }
  };
  $("tr-clear").onclick = () => {
    if (running) return;
    $("tr-input").value = "";
    $("tr-output").value = "";
    last = null;
    resultButtons(false);
    count();
    status("Ready for your next translation.");
  };
  $("tr-example").onclick = () => {
    if (running) return;
    $("tr-input").value =
      "Our new workspace brings translation and conversation together.\n\nKeep your data on your machine, and share your ideas with the world.";
    $("tr-format").value = "text";
    count();
  };
  $("tr-swap").onclick = () => {
    if ($("tr-source").value === "auto")
      return notify("Choose the source language before swapping.");
    const source = $("tr-source").value;
    $("tr-source").value = $("tr-target").value;
    $("tr-target").value = source;
    if ($("tr-output").value) {
      $("tr-input").value = $("tr-output").value;
      $("tr-output").value = "";
      last = null;
      resultButtons(false);
    }
    count();
    status("Languages swapped. Translate to check the reverse direction.");
  };
  function glossary() {
    const terms = $("tr-glossary")
      .value.split("\n")
      .filter((l) => l.trim())
      .map((line, i) => {
        const sep = line.includes("\t") ? "\t" : "=",
          at = line.indexOf(sep);
        if (at < 1 || !line.slice(at + 1).trim())
          throw new Error(
            `Glossary line ${i + 1}: use source = translation, or a tab separator.`,
          );
        return {
          source: line.slice(0, at).trim(),
          target: line.slice(at + 1).trim(),
        };
      });
    if (terms.length > 100) throw new Error("Use at most 100 glossary terms.");
    $("tr-terms").textContent = `${terms.length} terms`;
    return terms;
  }
  for (const id of ["tr-context", "tr-glossary", "tr-tone"])
    $(id).addEventListener("change", () => {
      savePreferences();
      try {
        glossary();
      } catch (e) {
        notify(e.message);
      }
    });
  function resultButtons(available) {
    $("tr-save").textContent = last?.saved ? "★ Saved" : "☆ Save";
    for (const id of [
      "tr-copy",
      "tr-download",
      "tr-save",
      "tr-bilingual",
      "tr-project",
      "tr-alternative",
    ])
      $(id).disabled = !available || running;
  }
  function setRunning(value) {
    running = value;
    for (const id of [
      "tr-run",
      "tr-upload",
      "tr-clear",
      "tr-example",
      "tr-model",
      "tr-input",
      "tr-output",
      "tr-format",
      "tr-limit",
      "tr-glossary-import",
      "tr-glossary-file",
      "tr-file-input",
    ])
      $(id).disabled = value;
    $("tr-stop").hidden = !value;
    resultButtons(!!last);
    updateLanguages();
  }
  function request(text, format, alternative) {
    return {
      model: $("tr-model").value,
      text,
      format,
      source: $("tr-source").value,
      target: $("tr-target").value,
      tone: $("tr-tone").value,
      context: $("tr-context").value,
      glossary: glossary(),
      max_tokens: Number($("tr-limit").value),
      alternative,
    };
  }
  const fmt = (n, d = 1) => (Number.isFinite(n) ? n.toFixed(d) : "—");
  function showMetrics(stats) {
    $("tr-metrics").hidden = false;
    $("tr-metrics").innerHTML = [
      ["Decode", `${fmt(stats.decode_tokens_per_second)} tok/s`],
      ["First token", `${fmt(stats.first_token_seconds, 2)} s`],
      ["Elapsed", `${fmt(stats.request_seconds, 2)} s`],
      ["Segments", `${stats.completed_segments} / ${stats.total_segments}`],
    ]
      .map(
        ([k, v]) => `<div><small>${k}</small><strong>${esc(v)}</strong></div>`,
      )
      .join("");
  }
  async function translate(payload, name = "translation.txt") {
    let result = null,
      provisional = "",
      completedText = [];
    requestId = null;
    const response = await fetch("/ui/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    try {
      for await (const e of EoleUI.events(response)) {
        if (e.type === "start") {
          requestId = e.request_id;
          if (stop) await stopJob();
        }
        if (e.type === "progress") {
          status(`Translating segment ${e.segment + 1} of ${e.total}…`);
          provisional = completedText.length
            ? completedText.join("\n\n") + "\n\n"
            : "";
        }
        if (e.type === "delta") {
          $("tr-output").value = provisional + e.text;
          count();
        }
        if (e.type === "segment") {
          $("tr-output").value = e.document;
          completedText.push(e.text);
          provisional = completedText.join("\n\n") + "\n\n";
          count();
        }
        if (e.type === "warning") status(e.message, true);
        if (e.type === "error") {
          $("tr-output").value = e.text || $("tr-output").value;
          throw new Error(e.message);
        }
        if (e.type === "done") {
          result = e;
          requestId = null;
        }
      }
      if (!result)
        throw new Error("Connection closed before translation completed.");
      $("tr-output").value = result.text;
      showMetrics(result.stats);
      count();
      last = {
        id: crypto.randomUUID(),
        date: new Date().toISOString(),
        ...payload,
        text: result.text,
        sourceText: payload.text,
        name,
        stats: result.stats,
        segments: result.segments,
        complete: result.complete,
        saved: false,
      };
      // History holds only complete translations; partial output remains visible.
      if (result.complete) {
        entries.unshift({ ...last });
        entries = entries.slice(0, 50);
        saveHistory();
      }
      $("tr-result-label").textContent = result.complete
        ? "Ready to refine"
        : result.stats.cancelled
          ? "Stopped · partial result"
          : "Incomplete · review before using";
      status(
        result.complete
          ? "Translation ready. Make it your own."
          : result.stats.cancelled
            ? "Stopped. Untranslated segments remain in the document."
            : "Output limit reached. Increase the limit and translate again.",
        !result.complete,
      );
      return { ...last };
    } finally {
      if (requestId)
        await fetch(`/ui/cancel/${requestId}`, { method: "POST" }).catch(
          () => {},
        );
      requestId = null;
    }
  }
  async function run(alternative = false) {
    if (running) return;
    try {
      if (!$("tr-input").value.trim())
        throw new Error("Add text or open a file first.");
      const originalFile = files[selectedFile];
      const source =
        originalFile &&
        originalFile.text.replace(/\r\n/g, "\n") === $("tr-input").value
          ? originalFile.text
          : $("tr-input").value;
      const payload = request(source, $("tr-format").value, alternative);
      stop = false;
      last = null;
      setRunning(true);
      $("tr-output").value = "";
      $("tr-result-label").textContent = "Translating…";
      $("tr-metrics").hidden = true;
      status("Preparing the model…");
      const result = await translate(
        payload,
        files[selectedFile]?.name ||
          `translation.${{ text: "txt", markdown: "md", srt: "srt", vtt: "vtt", json: "json" }[payload.format]}`,
      );
      if (files[selectedFile]) {
        files[selectedFile].result = result;
        renderFiles();
      }
    } catch (e) {
      notify(e.message);
      $("tr-result-label").textContent = "Needs attention";
    } finally {
      setRunning(false);
    }
  }
  $("tr-run").onclick = () => run();
  $("tr-alternative").onclick = () => run(true);
  async function stopJob() {
    stop = true;
    status("Stopping after the current generation step…");
    if (requestId)
      await fetch(`/ui/cancel/${requestId}`, { method: "POST" }).catch(
        () => {},
      );
  }
  $("tr-stop").onclick = stopJob;
  document.addEventListener("keydown", (e) => {
    if (
      !document.getElementById("view-translate").hidden &&
      (e.ctrlKey || e.metaKey) &&
      e.key === "Enter"
    ) {
      e.preventDefault();
      run();
    }
  });
  $("tr-copy").onclick = async () => {
    try {
      await navigator.clipboard.writeText($("tr-output").value);
      status("Translation copied.");
    } catch {
      notify("Select the translation and copy it manually.");
    }
  };
  const csvCell = (value) =>
    '"' + String(value ?? "").replace(/"/g, '""') + '"';
  $("tr-download").onclick = () => {
    if (last)
      EoleUI.download(
        "translated-" + last.name,
        $("tr-output").value,
        last.format === "json" ? "application/json" : "text/plain",
      );
  };
  $("tr-bilingual").onclick = () => {
    if (last)
      EoleUI.download(
        "bilingual.csv",
        "\ufeff" +
          [
            ["Source", "Translation"],
            [last.sourceText, $("tr-output").value],
          ]
            .map((r) => r.map(csvCell).join(","))
            .join("\r\n"),
        "text/csv",
      );
  };
  $("tr-project").onclick = () => {
    if (last)
      EoleUI.download(
        "translation-project.json",
        JSON.stringify({ ...last, text: $("tr-output").value }, null, 2),
        "application/json",
      );
  };
  $("tr-save").onclick = () => {
    if (!last) return;
    last.saved = true;
    last.text = $("tr-output").value;
    const found = entries.findIndex((e) => e.id === last.id);
    if (found >= 0) entries[found] = { ...last };
    else entries.unshift({ ...last });
    entries = entries.slice(0, 50);
    saveHistory();
    status("Saved to your local phrasebook.");
    $("tr-save").textContent = "★ Saved";
  };
  function renderHistory() {
    const q = $("tr-search").value.toLocaleLowerCase(),
      only = $("tr-only-saved").checked;
    const list = entries.filter(
      (e) =>
        (!only || e.saved) &&
        [e.sourceText, e.text, e.source, e.target]
          .join(" ")
          .toLocaleLowerCase()
          .includes(q),
    );
    $("tr-history").innerHTML = list.length
      ? list
          .map(
            (e) =>
              `<article class="translation-history-entry"><div><small>${esc(e.source === "auto" ? "Auto" : e.source)} → ${esc(e.target)} · ${esc(new Date(e.date).toLocaleDateString())}</small><div class="translation-history-pair"><p>${esc(e.sourceText.slice(0, 220))}</p><p>${esc(e.text.slice(0, 220))}</p></div></div><div><button class="quiet-button" data-tr-open="${esc(e.id)}">Open</button><button class="quiet-button" data-tr-star="${esc(e.id)}" aria-label="${e.saved ? "Unsave" : "Save"} translation">${e.saved ? "★" : "☆"}</button><button class="quiet-button" data-tr-remove="${esc(e.id)}" aria-label="Remove translation from history">×</button></div></article>`,
          )
          .join("")
      : '<div class="translation-empty">Your next translation could be the one worth keeping.</div>';
  }
  $("tr-search").oninput = renderHistory;
  $("tr-only-saved").onchange = renderHistory;
  $("tr-history").onclick = (e) => {
    const open = e.target.closest("[data-tr-open]"),
      star = e.target.closest("[data-tr-star]"),
      remove = e.target.closest("[data-tr-remove]");
    if (open) {
      if (running) return;
      const item = entries.find((x) => x.id === open.dataset.trOpen);
      if (!item) return;
      last = { ...item };
      setMode("text");
      $("tr-model").value = models.some((m) => m.id === item.model)
        ? item.model
        : $("tr-model").value;
      updateLanguages();
      $("tr-source").value = item.source;
      $("tr-target").value = item.target;
      $("tr-input").value = item.sourceText;
      $("tr-output").value = item.text;
      $("tr-format").value = item.format;
      $("tr-tone").value = item.tone || "default";
      $("tr-context").value = item.context || "";
      $("tr-glossary").value = (item.glossary || [])
        .map((t) => t.source + " = " + t.target)
        .join("\n");
      $("tr-limit").value = item.max_tokens || 4096;
      savePreferences();
      $("tr-result-label").textContent = "From local history";
      resultButtons(true);
      count();
      if (item.stats) showMetrics(item.stats);
      status("Restored from local history.");
    }
    if (star) {
      const item = entries.find((x) => x.id === star.dataset.trStar);
      item.saved = !item.saved;
      saveHistory();
      renderHistory();
    }
    if (remove) {
      entries = entries.filter((x) => x.id !== remove.dataset.trRemove);
      saveHistory();
      renderHistory();
    }
  };
  $("tr-export-history").onclick = () =>
    EoleUI.download(
      "translation-history.json",
      JSON.stringify(entries, null, 2),
      "application/json",
    );
  $("tr-upload").onclick = () => $("tr-file-input").click();
  $("tr-file-input").onchange = async (e) => {
    try {
      if (e.target.files.length > 8)
        throw new Error("Choose at most 8 files at a time.");
      const imported = [];
      for (const file of e.target.files) {
        const ext = file.name.split(".").at(-1).toLowerCase();
        if (["docx", "pdf"].includes(ext)) {
          if (file.size > 4000000)
            throw new Error(`${file.name}: maximum 4 MB.`);
          const bytes = new Uint8Array(await file.arrayBuffer());
          let binary = "";
          for (let i = 0; i < bytes.length; i += 16384)
            binary += String.fromCharCode(...bytes.subarray(i, i + 16384));
          const response = await fetch("/ui/translation/import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: file.name, data: btoa(binary) }),
          });
          const body = await response.json();
          if (!response.ok)
            throw new Error(body.detail || "Document import failed.");
          imported.push(body);
          continue;
        }
        if (file.size > 320000)
          throw new Error(`${file.name}: file is too large.`);
        const format = {
          txt: "text",
          md: "markdown",
          markdown: "markdown",
          srt: "srt",
          vtt: "vtt",
          json: "json",
        }[ext];
        if (!format)
          throw new Error("Choose TXT, Markdown, SRT, VTT or JSON files.");
        const text = new TextDecoder("utf-8", { fatal: true }).decode(
          await file.arrayBuffer(),
        );
        if (text.length > 80000)
          throw new Error(`${file.name}: maximum 80,000 characters.`);
        imported.push({ name: file.name, text, format });
      }
      files = imported;
      selectedFile = -1;
      renderFiles();
      if (files.length) openFile(0);
    } catch (error) {
      notify(error.message);
    } finally {
      e.target.value = "";
    }
  };
  function openFile(i) {
    if (running) return;
    selectedFile = i;
    const f = files[i];
    $("tr-input").value = f.text;
    $("tr-format").value = f.format;
    $("tr-output").value = f.result?.text || "";
    last = f.result || null;
    resultButtons(!!last);
    count();
    renderFiles();
    status(`Opened ${f.name}. Translate this file, or translate the queue.`);
  }
  function renderFiles() {
    $("tr-file-list").innerHTML =
      files
        .map(
          (f, i) =>
            `<button class="translation-file ${i === selectedFile ? "active" : ""}" data-tr-file="${i}"><span>▤ ${esc(f.name)}</span><small>${f.result?.complete ? "✓ Translated" : f.result ? "Partial" : `${f.text.length.toLocaleString()} chars`}</small></button>`,
        )
        .join("") +
      (files.length > 1
        ? '<button class="quiet-button" id="tr-run-all">Translate all files ↗</button>'
        : "");
  }
  $("tr-file-list").onclick = async (e) => {
    const file = e.target.closest("[data-tr-file]");
    if (file) return openFile(Number(file.dataset.trFile));
    if (e.target.id !== "tr-run-all" || running) return;
    stop = false;
    try {
      for (let i = 0; i < files.length; i++) {
        if (stop) break;
        openFile(i);
        await run();
        if (!last?.complete) break;
      }
    } catch (error) {
      notify(error.message);
    }
  };
  $("tr-glossary-import").onclick = () => $("tr-glossary-file").click();
  $("tr-glossary-file").onchange = async (e) => {
    try {
      const f = e.target.files[0];
      if (!f) return;
      if (f.size > 40000) throw new Error("Glossary file is too large.");
      const value = new TextDecoder("utf-8", { fatal: true }).decode(
        await f.arrayBuffer(),
      );
      const old = $("tr-glossary").value;
      $("tr-glossary").value = value;
      try {
        glossary();
      } catch (error) {
        $("tr-glossary").value = old;
        throw error;
      }
      savePreferences();
      status("Glossary imported.");
    } catch (error) {
      notify(error.message);
    } finally {
      e.target.value = "";
    }
  };
  $("tr-glossary-export").onclick = () => {
    try {
      EoleUI.download(
        "glossary.tsv",
        glossary()
          .map((t) => t.source + "\t" + t.target)
          .join("\n"),
        "text/tab-separated-values",
      );
    } catch (e) {
      notify(e.message);
    }
  };
  count();
  try {
    glossary();
  } catch {}
})();
