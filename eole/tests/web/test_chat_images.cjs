const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync("eole/server/web/app.js", "utf8");
const elements = new Map();
const element = (id) => {
  if (!elements.has(id))
    elements.set(id, {
      value: {
        model: "vision",
        "max-tokens": "128",
        preset: "fast",
        temperature: "0.7",
      }[id],
      checked: false,
      innerHTML: "",
    });
  return elements.get(id);
};
const context = vm.createContext({
  console,
  document: {
    createElement: () => ({
      getContext: () => ({ fillRect() {}, drawImage() {} }),
      toDataURL: () => "data:image/jpeg;base64,YQ==",
    }),
  },
  createImageBitmap: async () => ({ width: 640, height: 360, close() {} }),
  $: element,
  escapeHTML: (text) =>
    String(text).replaceAll("<", "&lt;").replaceAll('"', "&quot;"),
  notices: [],
});
vm.runInContext(
  `let busy=false, chatImages=[], imageUploads=0, compatible=true; let history=[]; const current=()=>({messages:history}); const selectedModel=()=>({images_supported:compatible}); const toast=text=>notices.push(text);`,
  context,
);
vm.runInContext(
  source.slice(
    source.indexOf("function imageThumbnails"),
    source.indexOf("const chatComposer"),
  ),
  context,
);
vm.runInContext(
  source.slice(
    source.indexOf("function requestPayload"),
    source.indexOf("async function consume"),
  ),
  context,
);
(async () => {
  await vm.runInContext(
    'addChatImages([{name:"shape.png",type:"image/png",size:1024}])',
    context,
  );
  assert.equal(vm.runInContext("chatImages.length", context), 1);
  assert.match(element("chat-image-attachments").innerHTML, /Remove shape.png/);
  assert.match(
    vm.runInContext("imageThumbnails(chatImages)", context),
    /data:image\/jpeg;base64,YQ==/,
  );
  assert.equal(
    vm.runInContext(
      'requestPayload([{role:"user",content:"Describe",images:chatImages}]).messages[0].images[0].name',
      context,
    ),
    "shape.png",
  );
  vm.runInContext("history=[{images:[{},{},{}]}]", context);
  await vm.runInContext(
    'addChatImages([{name:"fifth.png",type:"image/png",size:1024}])',
    context,
  );
  assert.equal(vm.runInContext("chatImages.length", context), 1);
  assert.match(context.notices.at(-1), /four images/);
  vm.runInContext("history=[]; compatible=false;", context);
  await vm.runInContext(
    'addChatImages([{name:"x.png",type:"image/png",size:1024}])',
    context,
  );
  assert.match(context.notices.at(-1), /image-capable/);
  vm.runInContext("compatible=true;", context);
  await vm.runInContext(
    'addChatImages([{name:"movie.mp4",type:"video/mp4",size:1024}])',
    context,
  );
  assert.match(context.notices.at(-1), /JPEG, PNG or WebP/);
  assert.equal(
    vm.runInContext(
      'imageThumbnails([{name:"bad",data:"javascript:alert(1)"}])',
      context,
    ),
    '<div class="chat-images"></div>',
  );
  console.log(
    "Chat images: previews, native payload, model gating, count and format limits passed",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
