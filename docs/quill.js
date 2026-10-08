// Thin JavaScript wrapper around the engine's WebAssembly exports
// (engine/src/wasm.rs). Used by the demo page and scripts/wasm_smoke.mjs.

// Reads the vocabulary out of the model file header (see train/export.py).
function parseVocab(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = String.fromCharCode(...bytes.subarray(0, 4));
  if (magic !== "QUIL") throw new Error("not a quill model file");
  const vocabSize = view.getUint32(8 + 4 * 4, true);
  const decoder = new TextDecoder();
  const vocab = [];
  let off = 8 + 6 * 4 + 2 * 4;
  for (let i = 0; i < vocabSize; i++) {
    const len = bytes[off];
    vocab.push(decoder.decode(bytes.subarray(off + 1, off + 1 + len)));
    off += 1 + len;
  }
  return vocab;
}

export async function createQuill(wasmBytes, modelBytes) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const w = instance.exports;
  const vocab = parseVocab(modelBytes);
  const index = new Map(vocab.map((s, i) => [s, i]));

  return {
    vocab,
    // Unknown characters are dropped, matching the Rust tokenizer.
    encode: (text) => [...text].map((c) => index.get(c)).filter((t) => t !== undefined),
    decode: (token) => vocab[token],

    load(quantize) {
      const ptr = w.quill_alloc(modelBytes.length);
      new Uint8Array(w.memory.buffer, ptr, modelBytes.length).set(modelBytes);
      if (w.quill_load(ptr, modelBytes.length, quantize ? 1 : 0) !== 0) {
        throw new Error("engine rejected the model file");
      }
    },
    seed: (s) => w.quill_seed(s >>> 0),
    maxSeqLen: () => w.quill_max_seq_len(),
    step: (token, pos, temperature) => w.quill_step(token, pos, temperature),
  };
}

// Generator yielding one new token at a time. When the context window is
// full it keeps the last half of the tokens and re-reads them from position 0,
// exactly like the CLI's `generate`.
export function* generate(quill, promptTokens, steps, temperature) {
  const max = quill.maxSeqLen();
  let tokens = promptTokens.length ? [...promptTokens] : [0];
  let pos = 0, i = 0, generated = 0;
  while (generated < steps) {
    if (pos === max) {
      tokens = tokens.slice(tokens.length - max / 2);
      pos = 0;
      i = 0;
    }
    const next = quill.step(tokens[i], pos, temperature);
    pos++;
    i++;
    if (i === tokens.length) {
      tokens.push(next);
      generated++;
      yield next;
    }
  }
}
