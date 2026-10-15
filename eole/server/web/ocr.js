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
    initialized = false;
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
  function renderMarkdown(text) {
    // Escape all model output, including table cells; never load model-supplied URLs.
    const lines = text.replace(/<\/?center>/gi, "").split("\n"),
      blocks = [];
    let paragraph = [],
      fenced = false;
    const flush = () => {
      if (paragraph.length) blocks.push(markdown(paragraph.join("\n")));
      paragraph = [];
    };
    const cells = (line) =>
      line
        .trim()
        .replace(/^\||\|$/g, "")
        .split("|")
        .map((cell) => cell.trim());
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].trim().startsWith("```")) fenced = !fenced;
      if (
        !fenced &&
        lines[i].includes("|") &&
        i + 1 < lines.length &&
        /^\s*\|?\s*:?-{3,}/.test(lines[i + 1]) &&
        cells(lines[i + 1]).every((c) => /^:?-{3,}:?$/.test(c))
      ) {
        flush();
        const header = cells(lines[i]);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes("|")) {
          rows.push(cells(lines[i]));
          i++;
        }
        i--;
        blocks.push(
          `<div class="ocr-table"><table><thead><tr>${header.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`,
        );
      } else paragraph.push(lines[i]);
    }
    flush();
    return blocks.join("");
  }
  function output() {
    const p = pages[selected];
    if (!p) return;
    const text = content(p);
    $("output").innerHTML = text
      ? mode === "rendered"
        ? renderMarkdown(text)
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
  function show() {
    controls();
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
  $("overlay").onchange = show;
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
    const models = detail.models.filter((m) => m.ocr),
      old = $("model").value;
    const options =
      models
        .map((m) => `<option value="${esc(m.id)}">${esc(m.id)}</option>`)
        .join("") || '<option value="">No OCR model configured</option>';
    if ($("model").innerHTML !== options) {
      $("model").innerHTML = options;
      if (models.some((m) => m.id === old)) $("model").value = old;
    }
    controls();
  });
  document.querySelector(".ocr-upload").onkeydown = (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("file").click();
    }
  };
  $("file").onchange = async () => {
    const files = Array.from($("file").files);
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
        `${pages.length} pages ready. Select a page or process the complete document.`,
      );
    } catch (e) {
      status(e.message);
    } finally {
      importing = false;
      $("file").value = "";
      controls();
    }
  };
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
      '<div class="ocr-empty"><h2>Ready for your next document</h2><p>Import a PDF or images to begin.</p></div>';
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
