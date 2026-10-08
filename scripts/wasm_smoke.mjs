// Smoke test for the WebAssembly build: load the model in Node and generate
// greedily with fp32 and int8 weights.
//   scripts/build_wasm.sh && node scripts/wasm_smoke.mjs
import { readFileSync } from "node:fs";
import { createQuill, generate } from "../docs/quill.js";

const root = new URL("..", import.meta.url);
const quill = await createQuill(
  readFileSync(new URL("docs/quill.wasm", root)),
  new Uint8Array(readFileSync(new URL("models/shakespeare.bin", root))),
);

for (const quantize of [false, true]) {
  quill.load(quantize);
  const start = performance.now();
  const steps = 300;
  let text = "";
  for (const t of generate(quill, quill.encode("ROMEO:\n"), steps, 0)) text += quill.decode(t);
  const secs = (performance.now() - start) / 1000;
  console.log(`--- ${quantize ? "int8" : "fp32"}: ${(steps / secs).toFixed(0)} tok/s ---\n${text.slice(0, 200)}\n`);
  if (text.length !== steps) throw new Error(`expected ${steps} characters, got ${text.length}`);
}
