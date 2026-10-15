/* Pure renderer regression checks: no browser, GPU, build system or npm install. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.resolve(__dirname, "../../server/web");
const escape = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const context = { EoleUI: { escape }, escapeHTML: escape };
context.window = context;
vm.createContext(context);
vm.runInContext(
  fs.readFileSync(path.join(root, "katex.min.js"), "utf8"),
  context,
);
const app = fs.readFileSync(path.join(root, "app.js"), "utf8");
vm.runInContext(
  app.slice(
    app.indexOf("function markdown(raw)"),
    app.indexOf("function statsHTML"),
  ),
  context,
);
vm.runInContext(
  fs.readFileSync(path.join(root, "ocr_render.js"), "utf8"),
  context,
);
const render = context.EoleOCR.render;
assert.match(render("Inline \\(x^2\\) and $$\\frac{1}{2}$$"), /<math/);
assert.match(render("$x+y$"), /<math/);
assert.doesNotMatch(render("`$x+y$`"), /<math/);
assert.doesNotMatch(
  render("```html\n<table><tr><td>unsafe</td></tr></table>\n```"),
  /<table>/,
);
assert.doesNotMatch(
  render('<img src=x onerror="bad()"><script>bad()</script>'),
  /<img|<script/,
);
const raw = '<table onclick="bad()"><tr><td>raw</td></tr></table>';
const html = render(raw, [
  {
    source: raw,
    rows: [
      [
        {
          text: "<script>bad()</script> \\(x^2\\)",
          colspan: 2,
          rowspan: 1,
          header: true,
        },
      ],
    ],
  },
]);
assert.match(html, /<table>/);
assert.match(html, /colspan="2"/);
assert.match(html, /<math/);
assert.doesNotMatch(html, /onclick|<script/);
assert.match(
  render("\\(\\includegraphics{https://example.invalid/x}\\)"),
  /includegraphics/,
);
assert.doesNotMatch(
  render("\\(\\includegraphics{https://example.invalid/x}\\)"),
  /<img/,
);
assert.match(
  render("| Name | Value |\n| --- | --- |\n| A | $x^2$ |"),
  /<table>/,
);
assert.match(render("EOLEOCRFRAGMENT0END \\(x\\)"), /^<p>EOLEOCRFRAGMENT0END/);
console.log(
  "OCR renderer: tables, math, code isolation and unsafe markup checks passed",
);
