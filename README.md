# Quill — a tiny LLM and a from-scratch inference engine in Rust

I trained a 5M-parameter Llama-style transformer on Shakespeare in PyTorch, then
wrote the engine that runs it in Rust without an ML framework. The engine handles
attention, a KV cache, int8 quantization and multithreading, and it compiles to
WebAssembly so the model runs entirely in the browser.

**[Live demo](https://gaming7810.github.io/neww/)** · [Benchmarks](#results) · [What didn't work](#what-didnt-work)

```
$ quill generate --prompt "ROMEO:" --temp 0.8
ROMEO:
We have some courtesy of the chider. What saws
What, what could be buck, for his eyes there be
both the wind there, I am not to be too come.

LADY ANNE:
Why, no more: thou wilt not dead, and say 'twas I can,
Nor nor a mourner leanness of the sun.

GREMIO:
No, I know not I; if that importable say thy hand,
of all my dear enemy to the heart; where I thyself?
```

## How it fits together

```
train/ (Python, PyTorch)                    engine/ (Rust, no ML dependencies)
  prepare.py   text -> char tokens            model.rs   file loader + forward pass + KV cache
  model.py     RMSNorm, RoPE, SwiGLU   ───►   ops.rs     dot / matmul / RMSNorm / softmax
  train.py     AdamW, cosine schedule  .bin   quant.rs   int8 group-wise quantization
  export.py    weights + test vectors         main.rs    CLI: generate, bench, ppl, report
                                              wasm.rs    C-style API for the browser
                                                    │
                                                    ▼
                                          docs/  demo page (WebAssembly, no server)
```

`train/model.py` and `engine/src/model.rs` are meant to be read side by side: each
PyTorch operation has a hand-written counterpart in Rust.

**Correctness is tested, not assumed.** `export.py` saves PyTorch's logits for a fixed
input, and `engine/tests/pytorch_parity.rs` checks that the Rust engine reproduces them.
The fp32 path matches to within 7×10⁻⁶ (max absolute logit difference). The int8
path is checked to pick the same next token. CI runs these tests, plus clippy, rustfmt
and a WebAssembly smoke test in Node, on every push.

## Model

| | |
|---|---|
| Architecture | Llama-style decoder: RMSNorm, rotary position embeddings, SwiGLU MLP, tied input/output embeddings, no biases |
| Size | 6 layers, 8 heads, d_model 256, d_ff 768 — **5.13M parameters** |
| Tokenizer | Character-level, 65 symbols |
| Data | [Tiny Shakespeare](https://github.com/karpathy/char-rnn) (1.1M characters, 90/10 split) |
| Training | 2,000 steps × 32 sequences × 128 chars, AdamW, cosine LR, dropout 0.1 — **28 minutes on a 4-core CPU** |
| Result | validation loss **1.473** nats/char (perplexity 4.36) |

![training curve](results/training_curve.png)

## Int8 quantization

Weights are split into groups of 32 values. Each group stores 32 `i8`s plus one `f32`
scale (`scale = max|w| / 127`), which comes to 1.125 bytes per weight instead of 4.
Activations are quantized the same way just before each matmul, so the inner loop is an
`i8 × i8 → i32` dot product, with one float multiply per group
(`engine/src/quant.rs`). The unit tests check two properties:

- the integer kernel gives exactly the same result as an f32 matmul over the dequantized values, so all of the error comes from rounding, none from the kernel;
- the error of each output stays within the analytical bound Σ(|w|·δx + |x|·δw + δw·δx).

## Results

Native build (not WebAssembly), batch size 1, generating 128 tokens from an empty context (median of 5 runs).
Perplexity is measured on all 871 non-overlapping 128-character windows of the
validation set. "Agreement" is the fraction of positions where int8 predicts the same
most likely next character as fp32.

| | fp32 | int8 |
|---|---|---|
| Linear-layer weights | 20.5 MB | **5.8 MB** (3.6× smaller) |
| Validation loss (nats/char) | 1.4726 | 1.4728 (+0.0002) |
| Perplexity | 4.361 | 4.361 |
| Same top-1 prediction as fp32 | — | 99.4% |
| tok/s, 1 thread | 844 | **1,131** (1.34×) |
| tok/s, 2 threads | 1,247 | **1,427** |
| tok/s, 4 threads | 1,130 | 1,148 |
| tok/s, 1 thread, `-C target-cpu=native` | ~720 | **~1,440** |
| tok/s, WebAssembly (Node 22, 1 thread) | **~450** | ~215 |

Raw numbers: [`results/benchmarks.json`](results/benchmarks.json). Regenerate them with
`quill report`. This is a shared 4-vCPU cloud VM, so expect ±10% between runs. The
`target-cpu=native` and WebAssembly rows are medians of repeated runs.

PyTorch reference loss on the same windows: **1.4726**. The Rust fp32 engine gives
the same number, as it should.

![throughput](results/throughput.png)

**Takeaways**

- **int8 costs almost nothing in quality.** Validation loss moves by 0.0002 nats, and the
  quantized model picks the same next character as fp32 at 99.4% of the 111k validation
  positions.
- **It is smaller and faster on CPU.** The linear weights shrink 3.6×. Decoding is 1.34×
  faster with a portable build. When the compiler may use the CPU's integer dot-product
  instructions (AVX-512 VNNI, `-C target-cpu=native`), int8 reaches ~1,440 tok/s on one
  thread, 1.65× the fastest fp32 build.
- **More threads stopped helping after 2.** See below.

## What didn't work

1. **Four threads are slower than two.** One token needs about 1 ms of work, spread over
   43 matmuls (6 layers × 7, plus the output projection), so each parallel region lasts
   roughly 25 µs. At that size, waking and synchronizing rayon workers costs about as much
   as the work they share. The `report` run spent 4m43s of its 14m37s of CPU time in the
   kernel. Ideas: parallelize across attention heads within a larger fused region, or
   decode several sequences at once so each matmul has more work.
2. **int8 is 2× *slower* than fp32 in WebAssembly** (~215 vs ~450 tok/s), the opposite of
   native. My hypothesis, not yet verified, is that baseline WebAssembly SIMD has no 8-bit
   dot-product instruction, so the widening `i8 × i8 → i32` loop compiles to many more
   instructions than the f32 one. The demo therefore defaults to fp32. Next step: try
   the relaxed-SIMD `i32x4.relaxed_dot_i8x16_i7x16_add` instruction.
3. **`-C target-cpu=native` made fp32 ~18% slower** (~875 → ~720 tok/s, median of three
   runs each) while making int8 ~34% faster (~1,080 → ~1,440). I don't understand the fp32 regression yet. The
   next step is to compare the generated assembly for `ops::dot` (`cargo asm`) and test
   whether the AVX-512 code path is responsible.
4. **A small evaluation set misled me.** The first 64 validation windows gave a loss of
   1.30, but the full validation set gives 1.47. The start of the held-out text is simply
   easier than average. All numbers above use the whole validation set (871 windows).
5. **The model is starting to overfit.** At the end of training, training loss is 1.17
   and validation loss is 1.45 (see the curve). More dropout or more data would likely
   help more than more steps.

## Run it yourself

```bash
# 1. Train (about 30 minutes on 4 CPU cores; see below for GPU)
pip install -r train/requirements.txt
python train/prepare.py
python train/train.py
python train/export.py          # -> models/shakespeare.bin, models/test_vectors.bin

# 2. Run the engine
cd engine && cargo test --release && cd ..
cargo run --release --manifest-path engine/Cargo.toml -- generate --prompt "JULIET:" --temp 0.8
cargo run --release --manifest-path engine/Cargo.toml -- generate --q8          # int8 weights
cargo run --release --manifest-path engine/Cargo.toml -- report                 # benchmark table
python train/plot.py

# 3. Browser demo
rustup target add wasm32-unknown-unknown
scripts/build_wasm.sh
python -m http.server -d docs   # open http://localhost:8000
```

A trained model is checked in (`models/shakespeare.bin`), so step 1 is optional.

### Training on a GPU

`train.py` uses a CUDA GPU automatically when PyTorch can see one. The forward pass
runs in bf16 mixed precision, while weights and optimizer state stay in fp32. The
checkpoint format is the same, so `export.py` and the engine need no changes.

```bash
# RTX 50-series cards need a PyTorch build with CUDA 12.8 or newer
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.get_device_name())"   # check the GPU is visible

python train/train.py                       # same model as above, now on the GPU
python train/train.py --compile             # optional: torch.compile (needs Triton; on Windows use WSL)

# A bigger model with a longer context (dim and hidden-dim must be multiples of 32)
python train/train.py --dim 384 --n-layers 8 --n-heads 8 --hidden-dim 1152 \
    --seq-len 256 --batch-size 64 --max-iters 5000
```

For experiments that shouldn't overwrite the published results, point the outputs elsewhere:

```bash
python train/train.py --dropout 0.2 --out runs/drop02 --log runs/drop02/log.csv
python train/export.py --ckpt runs/drop02/ckpt.pt --out-dir runs/drop02 --reference runs/drop02/ref.json
QUILL_MODEL_DIR=runs/drop02 cargo test --release --manifest-path engine/Cargo.toml   # parity check
```

On Windows, Rust and Python both work natively. Run the `.sh` scripts from Git Bash or WSL.

## Next steps

- **Int4 quantization** (two weights per byte), and a comparison of quality against int8.
- **An int8 model file**, to cut the browser download from 20 MB to about 6 MB.
- **Explicit SIMD kernels** (AVX2 `vpmaddubsw`, WebAssembly relaxed-SIMD dot products) to make int8 faster than fp32 instead of just smaller.
- **Run the int8 matmul on a custom RISC-V core.** I built a pipelined RV32I processor on FPGA in a previous project. The next step is a custom dot-product instruction.
