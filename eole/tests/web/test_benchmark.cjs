const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const context = { window: {} };
vm.runInNewContext(
  fs.readFileSync(
    path.resolve(__dirname, "../../server/web/benchmark.js"),
    "utf8",
  ),
  context,
);
const bench = context.window.EoleBenchmark;
assert.equal(bench.workloads.length, 3);
assert.equal(new Set(bench.workloads.map((w) => w.id)).size, 3);
assert.equal(bench.profiles.length, 6);
assert.equal(bench.profiles.filter((p) => bench.available(p, {})).length, 1);
const full = { speculation: { mtp_available: true, dflash2_available: true } };
assert.ok(bench.profiles.every((p) => bench.available(p, full)));
const stats = (speed, extra = {}) => ({
  output_tokens: 512,
  decode_tokens_per_second: speed,
  first_token_seconds: 0.2,
  speculation_mode: "mtp",
  mtp_acceptance: 0.5,
  ...extra,
});
let result = bench.summarize(
  [stats(100), stats(140), stats(120)],
  512,
  "mtp",
  3,
);
assert.equal(result.decode_tokens_per_second, 120);
assert.equal(result.first_token_seconds, 0.2);
assert.equal(result.acceptance, 0.5);
for (const extra of [
  { output_tokens: 40 },
  { speculation_mode: "off" },
  { cancelled: true },
  { decode_tokens_per_second: NaN },
]) {
  result = bench.summarize(
    [stats(100), stats(140), stats(120, extra)],
    512,
    "mtp",
    3,
  );
  assert.equal(result.status, "incomplete");
  assert.equal(result.valid_runs, 2);
  assert.equal(result.decode_tokens_per_second, null);
}
assert.equal(bench.summarize([], 512, "off", 3).decode_tokens_per_second, null);
assert.equal(
  bench.summarize([stats(80, { speculation_mode: "off" })], 512, "off", 1)
    .acceptance,
  null,
);
console.log(
  "Benchmark profiles, fixed workloads, medians, early EOS and actual-mode checks passed",
);

// Exercise the real async UI orchestration without a GPU: warmups must never
// enter measured summaries, and retries must append rather than erase results.
const inputs = {
  "benchmark-workloads": bench.workloads.map((w) => ({ value: w.id })),
  "benchmark-profiles": bench.profiles
    .filter((p) => ["mtp-int4", "dflash-int8-int4"].includes(p.id))
    .map((p) => ({ value: p.id, parentElement: {} })),
};
const elements = new Map();
function element(id) {
  if (!elements.has(id))
    elements.set(id, {
      value:
        id === "benchmark-tokens"
          ? "512"
          : id === "benchmark-repeats"
            ? "2"
            : "",
      querySelector: () => true,
      querySelectorAll: () => inputs[id] ?? [],
    });
  return elements.get(id);
}
let requests = [];
Object.assign(context, {
  EoleBenchmark: bench,
  benchmark: null,
  busy: false,
  stopRequested: false,
  $: element,
  selectedModel: () => ({
    ...full,
    id: "qwen-test",
    max_tokens: 4096,
    context_length: 4096,
  }),
  state: {
    compile_enabled: true,
    compile_mode: 0,
    hardware: {},
    version: "test",
  },
  setBusy: (value) => {
    context.busy = value;
    if (value) context.stopRequested = false;
  },
  escapeHTML: String,
  format: String,
  toast: (message) => {
    throw new Error(message);
  },
  refreshState: () => {},
  consume: async (payload) => {
    requests.push(payload);
    const warmup = requests.length % 3 === 1;
    return stats(warmup ? 9999 : 100, {
      speculation_mode: payload.speculation,
    });
  },
});
const app = fs.readFileSync(
  path.resolve(__dirname, "../../server/web/app.js"),
  "utf8",
);
vm.runInNewContext(
  app.slice(
    app.indexOf("function benchmarkChoices()"),
    app.indexOf('document.addEventListener("click"'),
  ),
  context,
);
(async () => {
  await context.runBenchmark();
  assert.equal(requests.length, 18);
  assert.equal(context.benchmark.results.length, 6);
  assert.ok(
    context.benchmark.results.every(
      (r) =>
        r.decode_tokens_per_second === 100 &&
        r.runs.length === 2 &&
        r.warmup.decode_tokens_per_second === 9999,
    ),
  );
  assert.ok(
    requests.every(
      (p) =>
        p.preset === "fast" &&
        !p.thinking &&
        p.max_tokens === 512 &&
        p.vocabulary_quantization === "int4",
    ),
  );
  requests = [];
  await context.runBenchmark();
  assert.equal(context.benchmark.results.length, 12);
  context.consume = async () => {
    throw new Error("CUDA out of memory. test");
  };
  await context.runBenchmark();
  assert.equal(context.benchmark.results.length, 13);
  assert.equal(context.benchmark.results.at(-1).status, "error");
  assert.match(element("benchmark-progress").textContent, /GPU memory is full/);
  assert.equal(context.busy, false);
  context.consume = async (payload) =>
    stats(50, { speculation_mode: payload.speculation, cancelled: true });
  await context.runBenchmark();
  assert.equal(context.benchmark.results.at(-1).runs.length, 0);
  assert.equal(context.benchmark.results.at(-1).status, "cancelled");
  console.log(
    "Benchmark runner: discarded warmup, exact payloads, retained results, OOM and cancellation passed",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
