/* Local OCR preview: model HTML becomes data cells; math is rendered with trust disabled. */
"use strict";
window.EoleOCR = (() => {
  const esc = EoleUI.escape;
  const mathPattern =
    /\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)|\$\$([\s\S]*?)\$\$|(?<!\\)\$(?!\s)([^$\n]*?\S)\$(?!\d)/g;
  function typeset(source, display) {
    if (!window.katex || source.length > 12000) return esc(source);
    try {
      return window.katex.renderToString(source, {
        output: "mathml",
        displayMode: display,
        trust: false,
        throwOnError: true,
        maxExpand: 1000,
        maxSize: 10,
        strict: "ignore",
      });
    } catch {
      return esc(source);
    }
  }
  function cellText(text) {
    let end = 0,
      html = "";
    for (const match of String(text).matchAll(mathPattern)) {
      html += esc(text.slice(end, match.index));
      html += typeset(
        match[1] ?? match[2] ?? match[3] ?? match[4],
        match[1] !== undefined || match[3] !== undefined,
      );
      end = match.index + match[0].length;
    }
    return html + esc(text.slice(end));
  }
  function tableHTML(rows) {
    const span = (n) => (Number.isInteger(n) && n > 0 && n <= 100 ? n : 1);
    return `<div class="ocr-table"><table><tbody>${rows
      .map(
        (row) =>
          `<tr>${row
            .map((cell) => {
              const tag = cell.header ? "th" : "td";
              return `<${tag} colspan="${span(cell.colspan)}" rowspan="${span(cell.rowspan)}">${cellText(cell.text)}</${tag}>`;
            })
            .join("")}</tr>`,
      )
      .join("")}</tbody></table></div>`;
  }
  function basicMarkdown(text) {
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
  function render(text, tables = []) {
    let prefix = "EOLEOCRFRAGMENT";
    while (text.includes(prefix)) prefix += "X";
    const fragments = [];
    const stash = (html) => {
      const key = `${prefix}${fragments.length}END`;
      fragments.push({ key, html });
      return key;
    };
    // Protect code before interpreting table or formula syntax.
    text = text.replace(/```[\s\S]*?(?:```|$)|`[^`\n]+`/g, (code) =>
      stash(markdown(code)),
    );
    for (const table of tables)
      text = text.replace(
        table.source,
        () => `\n\n${stash(tableHTML(table.rows))}\n\n`,
      );
    text = text.replace(
      mathPattern,
      (all, displayA, inlineA, displayB, inlineB) =>
        stash(
          typeset(
            displayA ?? inlineA ?? displayB ?? inlineB,
            displayA !== undefined || displayB !== undefined,
          ),
        ),
    );
    let html = basicMarkdown(text);
    for (const { key, html: fragment } of fragments) {
      html = html
        .split(`<p>${key}</p>`)
        .join(fragment)
        .split(key)
        .join(fragment);
    }
    return html;
  }
  return { render };
})();
