/* Exercise the real file-drop handlers with browser primitives replaced by small fakes. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const elements = new Map(),
  documentEvents = {},
  uploads = [];
function element(id) {
  if (!elements.has(id)) {
    const classes = new Set();
    elements.set(id, {
      value: id === "ocr-model" ? "ocr" : "",
      textContent: "",
      files: [],
      innerHTML: "",
      handlers: {},
      setAttribute() {},
      classList: {
        add: (c) => classes.add(c),
        remove: (c) => classes.delete(c),
        has: (c) => classes.has(c),
      },
      addEventListener(name, fn) {
        this.handlers[name] = fn;
      },
      parentElement: {},
      click() {
        this.onclick?.({ target: this });
      },
    });
  }
  return elements.get(id);
}
const context = {
  document: {
    getElementById: element,
    addEventListener: (name, fn) => (documentEvents[name] = fn),
  },
  window: { addEventListener() {} },
  EoleUI: { escape: String },
  EoleOCR: { render: String },
  FileReader: class {
    readAsDataURL() {
      this.result = "data:image/png;base64,QQ==";
      this.onload();
    }
  },
  fetch: async (url, options) => {
    uploads.push(JSON.parse(options.body));
    return {
      ok: true,
      json: async () => ({ pages: [{ data: "QQ==", width: 1, height: 1 }] }),
    };
  },
};
vm.runInNewContext(
  fs.readFileSync(path.resolve(__dirname, "../../server/web/ocr.js"), "utf8"),
  context,
);
const area = element("ocr-document-area"),
  drop = (files) => ({
    dataTransfer: { types: ["Files"], files },
    preventDefault() {
      this.prevented = true;
    },
    stopPropagation() {},
  });
(async () => {
  const first = drop([
    { name: "one.pdf", size: 20 },
    { name: "two.png", size: 20 },
  ]);
  area.handlers.dragenter(first);
  assert(area.classList.has("ocr-drag-active"));
  area.handlers.dragover(first);
  assert.equal(first.dataTransfer.dropEffect, "copy");
  area.handlers.drop(first);
  assert(first.prevented);
  assert(!area.classList.has("ocr-drag-active"));
  // A second drop during import must not start another upload batch.
  area.handlers.drop(drop([{ name: "ignored.png", size: 20 }]));
  await new Promise(setImmediate);
  assert.deepEqual(
    uploads.map((f) => f.name),
    ["one.pdf", "two.png"],
  );
  assert.match(element("ocr-status").textContent, /2 pages ready/);
  assert.match(element("ocr-thumbnails").innerHTML, /Page 2/);
  area.handlers.drop(drop([{ name: "bad.exe", size: 20 }]));
  await new Promise(setImmediate);
  assert.equal(uploads.length, 2);
  assert.match(element("ocr-status").textContent, /PDF, PNG, JPEG or WebP/);
  const outside = drop([]);
  documentEvents.drop(outside);
  assert(outside.prevented);
  console.log(
    "OCR drop zone: multiple files, highlight, import gating and navigation protection passed",
  );
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
