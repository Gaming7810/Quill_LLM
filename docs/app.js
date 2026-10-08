import { createQuill, generate } from "./quill.js";

const $ = (id) => document.getElementById(id);
const runBtn = $("run"), stopBtn = $("stop"), output = $("output"), stats = $("stats");

let quill = null;
let loadedPrecision = null;
let stopRequested = false;

$("temp").addEventListener("input", (e) => ($("tempValue").textContent = e.target.value));

async function init() {
  try {
    const [wasm, model] = await Promise.all([
      fetch("quill.wasm").then((r) => r.arrayBuffer()),
      fetch("shakespeare.bin").then((r) => r.arrayBuffer()),
    ]);
    quill = await createQuill(wasm, new Uint8Array(model));
    stats.textContent = `Model loaded (${(model.byteLength / 1e6).toFixed(1)} MB fp32 file, vocabulary of ${quill.vocab.length} characters).`;
    runBtn.textContent = "Generate";
    runBtn.disabled = false;
  } catch (err) {
    runBtn.textContent = "Failed to load";
    stats.textContent = String(err);
  }
}

function ensurePrecision() {
  const p = $("precision").value;
  if (p !== loadedPrecision) {
    const t0 = performance.now();
    quill.load(p === "1");
    loadedPrecision = p;
    return performance.now() - t0;
  }
  return 0;
}

async function run() {
  runBtn.disabled = true;
  stopBtn.disabled = false;
  stopRequested = false;

  const loadMs = ensurePrecision();
  quill.seed(Number($("seed").value) || 1);
  const prompt = $("prompt").value;
  const tokens = quill.encode(prompt);
  const steps = Math.max(1, Math.min(5000, Number($("steps").value) || 1));
  const temperature = Number($("temp").value);

  output.innerHTML = "";
  const promptSpan = document.createElement("span");
  promptSpan.className = "prompt";
  promptSpan.textContent = tokens.map(quill.decode).join("");
  const genNode = document.createTextNode("");
  output.append(promptSpan, genNode);

  const gen = generate(quill, tokens, steps, temperature);
  const start = performance.now();
  let count = 0;
  const precision = loadedPrecision === "1" ? "int8" : "fp32";

  // Generate in small batches between frames so the page stays responsive.
  await new Promise((resolve) => {
    function frame() {
      const frameStart = performance.now();
      let text = "";
      while (performance.now() - frameStart < 12) {
        const { value, done } = gen.next();
        if (done || stopRequested) return finish(text);
        text += quill.decode(value);
        count++;
      }
      genNode.appendData(text);
      const secs = (performance.now() - start) / 1000;
      stats.textContent = `${precision}: ${count} chars, ${(count / secs).toFixed(0)} tok/s`;
      requestAnimationFrame(frame);
    }
    function finish(text) {
      genNode.appendData(text);
      const secs = (performance.now() - start) / 1000;
      stats.textContent = `${precision}: ${count} chars in ${secs.toFixed(2)} s — ${(count / secs).toFixed(0)} tok/s` +
        (loadMs ? ` (weights prepared in ${loadMs.toFixed(0)} ms)` : "");
      resolve();
    }
    requestAnimationFrame(frame);
  });

  runBtn.disabled = false;
  stopBtn.disabled = true;
}

runBtn.addEventListener("click", run);
stopBtn.addEventListener("click", () => (stopRequested = true));
init();
