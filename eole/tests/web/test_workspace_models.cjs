const assert = require("node:assert/strict"),
  vm = require("node:vm"),
  fs = require("node:fs"),
  path = require("node:path");
const saved = new Map(),
  context = {
    window: {},
    localStorage: {
      getItem: (k) => saved.get(k),
      setItem: (k, v) => saved.set(k, v),
    },
  };
vm.runInNewContext(
  fs.readFileSync(
    path.resolve(__dirname, "../../server/web/workspace.js"),
    "utf8",
  ),
  context,
);
const ui = context.window.EoleUI,
  models = [
    { id: "qwen" },
    { id: "hy", translation: { preferred: true } },
    { id: "ocr" },
  ];
const state = {
  workspace: { models: { chat: "qwen", translate: "hy", ocr: "ocr" } },
};
assert.equal(ui.chooseModel(state, "chat", models, ""), "qwen");
assert.equal(ui.chooseModel(state, "translate", models, ""), "hy");
assert.equal(ui.chooseModel(state, "ocr", models, ""), "ocr");
ui.rememberModel("translate", "qwen");
assert.equal(ui.chooseModel(state, "translate", models, ""), "qwen");
assert.equal(ui.chooseModel(state, "ocr", models, ""), "ocr");
assert.equal(ui.chooseModel(state, "chat", models, "hy"), "hy");
saved.clear();
assert.equal(ui.chooseModel({}, "translate", models, ""), "hy");
assert.equal(
  ui.chooseModel(state, "ocr", [{ id: "replacement" }], ""),
  "replacement",
);
console.log(
  "Independent workspace model defaults, remembered choices and stale selection checks passed",
);
