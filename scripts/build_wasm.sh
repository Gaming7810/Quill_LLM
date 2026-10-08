#!/usr/bin/env bash
# Build the engine for the browser and copy it, with the model, into docs/.
#   rustup target add wasm32-unknown-unknown
set -euo pipefail
cd "$(dirname "$0")/.."

# simd128 lets LLVM vectorize the dot products (supported by all current browsers).
RUSTFLAGS="-C target-feature=+simd128" cargo rustc \
  --manifest-path engine/Cargo.toml --release --lib \
  --target wasm32-unknown-unknown --no-default-features --crate-type cdylib

cp engine/target/wasm32-unknown-unknown/release/quill.wasm docs/quill.wasm
cp models/shakespeare.bin docs/shakespeare.bin
ls -lh docs/quill.wasm docs/shakespeare.bin
