const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const script = fs.readFileSync("eole/server/web/knowledge.js", "utf8");
const esc = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
const elements = new Map();
const context = vm.createContext({
  escape: esc,
  markdown: esc,
  $: (id) => {
    if (!elements.has(id)) elements.set(id, { innerHTML: "" });
    return elements.get(id);
  },
});
vm.runInContext("let sources=[];", context);
vm.runInContext(
  script.slice(
    script.indexOf("  function renderSources"),
    script.indexOf("  function renderMessages"),
  ),
  context,
);
assert.match(
  vm.runInContext(
    'citationHTML("Supported [1], unknown [9]",[{id:1}])',
    context,
  ),
  /data-kg-cite="1"/,
);
assert.doesNotMatch(
  vm.runInContext('citationHTML("[9]",[{id:1}])', context),
  /data-kg-cite/,
);
assert.doesNotMatch(
  vm.runInContext('citationHTML("<script>bad</script>",[])', context),
  /<script>/,
);
vm.runInContext(
  'sources=[{id:1,name:"<script>alert(1)</script>",page:3,chunk:2,text:"<img src=x onerror=bad()>"}]; renderSources();',
  context,
);
assert.match(elements.get("kg-sources").innerHTML, /Page 3/);
assert.doesNotMatch(elements.get("kg-sources").innerHTML, /<script>|<img/);
assert.match(elements.get("kg-sources").innerHTML, /&lt;img/);
console.log(
  "Knowledge renderer: supported citations, PDF pages and escaped untrusted sources passed",
);
