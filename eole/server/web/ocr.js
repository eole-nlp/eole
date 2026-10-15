/* Page images and OCR results live only in this tab, never localStorage. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(`ocr-${id}`),
    esc = EoleUI.escape;
  let pages = [],
    selected = 0,
    mode = "rendered",
    running = false,
    stopping = false,
    requestId = null,
    importing = false,
    initialized = false,
    models = [],
    layoutPreference = true;
  const status = (text) => {
    $("status").textContent = text;
  };
  const content = (p) => (mode === "raw" ? p.raw || "" : p.markdown || "");
  const filename = (p) =>
    p.name.replace(/\.[^.]+$/, "") + `-page-${p.number}.md`;
  function controls() {
    for (const id of ["current", "all"])
      $(id).disabled =
        running || importing || !pages.length || !$("model").value;
    $("file").disabled = running || importing;
    $("dropzone").setAttribute("aria-disabled", String(running || importing));
    $("clear").disabled = running || importing || !pages.length;
    $("model").disabled = running;
    $("limit").disabled = running;
    $("stop").hidden = !running;
    $("stop").disabled = stopping;
    $("prev").disabled = selected === 0;
    $("next").disabled = selected >= pages.length - 1;
    $("copy").disabled = !pages[selected]?.raw;
    $("download").disabled = !pages[selected]?.raw;
    $("download-all").disabled = !pages.some((p) => p.raw);
  }

  function output() {
    const p = pages[selected];
    if (!p) return;
    const text = content(p);
    $("output").innerHTML = text
      ? mode === "rendered"
        ? EoleOCR.render(text, p.tables || [])
        : `<pre>${esc(text)}</pre>`
      : `<div class="ocr-empty"><h2>${p.status === "processing" ? "Reading this page…" : "Ready when you are"}</h2><p>${esc(p.error || "Process this page or the complete document to extract its contents.")}</p></div>`;
    if (!p.stats)
      $("metrics").textContent =
        `Page ${selected + 1} · ${p.status} · results stay in this browser tab`;
    if (p.stats) {
      const s = p.stats;
      $("metrics").textContent =
        `Page ${selected + 1} · ${s.output_tokens ?? "—"} output tokens · ${Number(s.request_seconds || 0).toFixed(1)} s total · ${s.decode_tokens_per_second ? Number(s.decode_tokens_per_second).toFixed(1) + " tok/s decode" : "decode speed unavailable"} · ${p.status}`;
    }
  }
  function layoutControl() {
    const page = pages[selected],
      model = models.find((m) => m.id === $("model").value);
    const supported = page?.layout_supported ?? model?.ocr_layout ?? false;
    $("overlay").disabled = !supported;
    $("overlay").checked = supported && layoutPreference;
    if ($("layout-label"))
      $("layout-label").textContent = supported ? "Layout" : "No layout data";
    $("overlay").parentElement.title = supported
      ? "Show model grounding coordinates"
      : "This model's document parsing does not provide layout boxes";
  }
  $("model").onchange = layoutControl;
  function show() {
    controls();
    layoutControl();
    const p = pages[selected];
    if (!p) return;
    $("name").textContent = p.name;
    $("page-count").textContent =
      `Page ${selected + 1} / ${pages.length} · ${p.status}`;
    const boxes = $("overlay").checked
      ? (p.boxes || [])
          .map(
            (b) =>
              `<div class="ocr-box" title="${esc(b.label)}" style="left:${b.x}%;top:${b.y}%;width:${b.width}%;height:${b.height}%"></div>`,
          )
          .join("")
      : "";
    $("preview").innerHTML =
      `<div class="ocr-sheet" style="width:${Number($("zoom").value)}%"><img src="data:image/jpeg;base64,${p.data}" alt="${esc(p.name)} page ${p.number}">${boxes}</div>`;
    $("thumbnails").innerHTML = pages
      .map(
        (p, i) =>
          `<button class="ocr-thumb ${i === selected ? "active" : ""}" data-page="${i}" aria-label="Page ${i + 1}: ${esc(p.status)}" aria-current="${i === selected ? "page" : "false"}"><img src="data:image/jpeg;base64,${p.data}" alt="" loading="lazy"><span>${i + 1} · ${esc(p.status)}</span></button>`,
      )
      .join("");
    output();
  }
  $("thumbnails").onclick = (e) => {
    const b = e.target.closest("[data-page]");
    if (b) {
      selected = Number(b.dataset.page);
      show();
    }
  };
  $("prev").onclick = () => {
    if (selected > 0) {
      selected--;
      show();
    }
  };
  $("next").onclick = () => {
    if (selected < pages.length - 1) {
      selected++;
      show();
    }
  };
  $("overlay").onchange = () => {
    layoutPreference = $("overlay").checked;
    show();
  };
  $("zoom").onchange = show;
  for (const id of ["rendered", "source", "raw"])
    $(id).onclick = () => {
      mode = id;
      for (const tab of ["rendered", "source", "raw"])
        $(tab).setAttribute("aria-pressed", String(tab === id));
      output();
    };
  window.addEventListener("eole:state", ({ detail }) => {
    if (!initialized && detail.models.length) {
      initialized = true;
      if (detail.models.every((m) => m.ocr)) switchView("ocr");
    }
    models = detail.models.filter((m) => m.ocr);
    const old = $("model").value;
    const options =
      models
        .map((m) => `<option value="${esc(m.id)}">${esc(m.id)}</option>`)
        .join("") || '<option value="">No OCR model configured</option>';
    if ($("model").innerHTML !== options) {
      $("model").innerHTML = options;
      if (models.some((m) => m.id === old)) $("model").value = old;
    }
    controls();
    layoutControl();
  });
  const area = $("document-area");
  let dragDepth = 0;
  const hasFiles = (event) =>
    Array.from(event.dataTransfer?.types || []).includes("Files");
  $("dropzone").onclick = (event) => {
    if (event.target !== $("file") && !running && !importing) $("file").click();
  };
  $("dropzone").onkeydown = (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("dropzone").click();
    }
  };
  area.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth++;
    if (!running && !importing) area.classList.add("ocr-drag-active");
  });
  area.addEventListener("dragover", (event) => {
    if (hasFiles(event)) {
      event.preventDefault();
      event.dataTransfer.dropEffect = running || importing ? "none" : "copy";
    }
  });
  area.addEventListener("dragleave", (event) => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) area.classList.remove("ocr-drag-active");
  });
  area.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.stopPropagation();
    dragDepth = 0;
    area.classList.remove("ocr-drag-active");
    importFiles(Array.from(event.dataTransfer.files));
  });
  // A missed drop must not navigate away and discard the current document.
  for (const name of ["dragover", "drop"])
    document.addEventListener(name, (event) => {
      if (hasFiles(event)) event.preventDefault();
    });
  $("file").onchange = () => importFiles(Array.from($("file").files));
  async function importFiles(files) {
    if (running || importing) {
      status("Wait for processing to finish before adding files.");
      return;
    }
    if (!files.length) return;
    if (
      files.length > 50 ||
      files.some((f) => !/\.(pdf|png|jpe?g|webp)$/i.test(f.name))
    ) {
      status("Choose up to 50 PDF, PNG, JPEG or WebP files.");
      $("file").value = "";
      return;
    }
    importing = true;
    controls();
    try {
      for (const f of files) {
        if (f.size > 20000000) throw new Error("File limit: 20 MB");
        status(`Opening ${f.name}…`);
        const data = await new Promise((resolve, reject) => {
          const r = new FileReader();
          r.onload = () => resolve(r.result.split(",")[1]);
          r.onerror = reject;
          r.readAsDataURL(f);
        });
        const response = await fetch("/ui/ocr/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: f.name, data }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Import failed");
        if (pages.length + result.pages.length > 50)
          throw new Error(
            "Workspace limit: 50 pages. Clear the workspace before importing more.",
          );
        const first = pages.length;
        pages.push(
          ...result.pages.map((p, i) => ({
            ...p,
            name: f.name,
            number: i + 1,
            status: "pending",
            raw: "",
            markdown: "",
            boxes: [],
          })),
        );
        selected = first;
        show();
      }
      status(
        `${pages.length} ${pages.length === 1 ? "page" : "pages"} ready. Select a page or process the complete document.`,
      );
    } catch (e) {
      status(e.message);
    } finally {
      importing = false;
      $("file").value = "";
      controls();
    }
  }
  async function process(indices) {
    const max_tokens = Number($("limit").value);
    if (!Number.isInteger(max_tokens) || max_tokens < 1 || max_tokens > 16384) {
      status("Choose an output limit between 1 and 16384.");
      return;
    }
    const model = $("model").value;
    running = true;
    stopping = false;
    controls();
    let failed = 0;
    try {
      for (const i of indices) {
        if (stopping) break;
        const p = pages[i];
        selected = i;
        p.status = "processing";
        p.raw = "";
        p.markdown = "";
        p.boxes = [];
        p.error = null;
        p.stats = null;
        p.layout_supported =
          models.find((m) => m.id === model)?.ocr_layout ?? false;
        p.tables = [];
        show();
        status(`Reading page ${i + 1} of ${pages.length}…`);
        let terminal = false;
        try {
          const response = await fetch("/ui/ocr/page", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model, data: p.data, max_tokens }),
          });
          for await (const item of EoleUI.events(response)) {
            if (item.type === "start") {
              requestId = item.request_id;
              if (stopping) await cancel();
            } else if (item.type === "delta") {
              p.raw += item.text;
              if (selected === i && mode === "raw") output();
            } else if (item.type === "error") throw new Error(item.message);
            else if (item.type === "done") {
              terminal = true;
              p.raw = item.raw || p.raw;
              p.markdown = item.markdown || "";
              p.boxes = item.boxes || [];
              p.tables = item.tables || [];
              p.layout_supported = item.layout_supported ?? false;
              p.stats = item.stats;
              p.status = item.cancelled
                ? "cancelled"
                : item.incomplete
                  ? "incomplete"
                  : "complete";
            }
          }
          if (!terminal)
            throw new Error(
              "Connection ended before the page finished; retry this page.",
            );
        } catch (e) {
          p.status = "failed";
          p.error = e.message;
          failed++;
        } finally {
          requestId = null;
          show();
        }
      }
      status(
        stopping
          ? "Stopped. Completed pages are preserved; process any page to retry."
          : `${pages.filter((p) => p.status === "complete").length} / ${pages.length} pages complete${failed ? ` · ${failed} failed — select a page to retry` : ""}${pages.some((p) => p.status === "incomplete") ? " · Output limit reached: increase tokens and retry incomplete pages" : ""}.`,
      );
    } finally {
      running = false;
      controls();
    }
  }
  async function cancel() {
    if (requestId) {
      const response = await fetch(`/ui/cancel/${requestId}`, {
        method: "POST",
      });
      if (!response.ok && response.status !== 404)
        throw new Error("Could not stop the request");
    }
  }
  $("current").onclick = () => process([selected]);
  $("all").onclick = () =>
    process(
      pages.map((p, i) => i).filter((i) => pages[i].status !== "complete"),
    );
  $("stop").onclick = async () => {
    stopping = true;
    controls();
    status("Stopping generation… completed pages will be kept.");
    try {
      await cancel();
    } catch (e) {
      status(e.message);
    }
  };
  $("copy").onclick = async () => {
    try {
      await navigator.clipboard.writeText(content(pages[selected]));
      status("Copied this page.");
    } catch {
      status("Clipboard unavailable. Download the Markdown instead.");
    }
  };
  $("download").onclick = () =>
    EoleUI.download(
      filename(pages[selected]),
      pages[selected].markdown || pages[selected].raw,
      "text/markdown",
    );
  $("download-all").onclick = () =>
    EoleUI.download(
      "document.md",
      pages
        .map(
          (p, i) =>
            `<!-- Page ${i + 1}: ${p.name.replaceAll("--", "—")} · ${p.status} -->\n\n${p.markdown || (p.raw ? p.raw : "[Page not processed]")}`,
        )
        .join("\n\n---\n\n"),
      "text/markdown",
    );
  $("clear").onclick = () => {
    pages = [];
    selected = 0;
    $("preview").innerHTML =
      '<div class="ocr-empty"><h2>Ready for your next document</h2><p>Drop a PDF or images above to begin.</p></div>';
    $("output").textContent = "";
    $("thumbnails").textContent = "";
    $("name").textContent = "Original document";
    $("page-count").textContent = "No pages";
    $("metrics").textContent = "Results stay in this browser tab";
    status("Workspace cleared.");
    controls();
  };
  controls();
})();
