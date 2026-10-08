"""Export a trained checkpoint to the flat binary format read by the Rust engine.

Also writes test vectors (PyTorch logits for a fixed prompt) that the engine's
tests compare against, and the PyTorch validation loss used as the reference
for the perplexity benchmark.

File layout (little-endian), see engine/src/model.rs:
    magic "QUIL", version u32
    dim, hidden_dim, n_layers, n_heads, vocab_size, max_seq_len : u32
    norm_eps, rope_theta : f32
    vocab: vocab_size x (u8 length, utf-8 bytes)
    tok_embeddings (vocab_size, dim)
    per layer: attention_norm (dim), wq, wk, wv, wo (dim, dim),
               ffn_norm (dim), w1 (hidden, dim), w2 (dim, hidden), w3 (hidden, dim)
    norm (dim)
    (output projection is tied to tok_embeddings and not stored)
"""

import argparse
import json
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

from model import Config, Transformer

ROOT = os.path.join(os.path.dirname(__file__), "..")
MAGIC = b"QUIL"
VERSION = 1


def write_tensor(f, t):
    f.write(t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())


def export_model(model, chars, path):
    cfg = model.cfg
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<6I", cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.vocab_size, cfg.max_seq_len))
        f.write(struct.pack("<2f", cfg.norm_eps, cfg.rope_theta))
        for c in chars:
            b = c.encode("utf-8")
            f.write(struct.pack("<B", len(b)) + b)
        write_tensor(f, model.tok_embeddings.weight)
        for layer in model.layers:
            write_tensor(f, layer.attention_norm.weight)
            for w in (layer.attention.wq, layer.attention.wk, layer.attention.wv, layer.attention.wo):
                write_tensor(f, w.weight)
            write_tensor(f, layer.ffn_norm.weight)
            for w in (layer.feed_forward.w1, layer.feed_forward.w2, layer.feed_forward.w3):
                write_tensor(f, w.weight)
        write_tensor(f, model.norm.weight)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


@torch.no_grad()
def export_test_vectors(model, val, path, n=64):
    tokens = torch.from_numpy(val[:n].astype(np.int64))[None]
    logits, _ = model(tokens)
    with open(path, "wb") as f:
        f.write(b"QTV1")
        f.write(struct.pack("<2I", n, model.cfg.vocab_size))
        f.write(tokens[0].numpy().astype("<u4").tobytes())
        f.write(logits[0].numpy().astype("<f4").tobytes())
    print(f"wrote {path}")


@torch.no_grad()
def reference_val_loss(model, val, windows):
    # Non-overlapping windows of max_seq_len + 1 tokens, the same split the
    # engine's `ppl` command uses, so the numbers are directly comparable.
    t = model.cfg.max_seq_len
    n = min(windows, (len(val) - 1) // t)
    x = torch.from_numpy(np.stack([val[i * t : i * t + t] for i in range(n)]).astype(np.int64))
    y = torch.from_numpy(np.stack([val[i * t + 1 : i * t + t + 1] for i in range(n)]).astype(np.int64))
    total = 0.0
    for i in range(0, n, 32):
        logits, _ = model(x[i : i + 32])
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[i : i + 32].reshape(-1), reduction="sum").item()
    return total / (n * t), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(ROOT, "train", "out", "ckpt.pt"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "models"))
    p.add_argument("--windows", type=int, default=1_000_000, help="max validation windows (default: all)")
    p.add_argument("--reference", default=os.path.join(ROOT, "results", "pytorch_reference.json"),
                   help="where to write the PyTorch reference loss")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = Config(**ckpt["config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    chars = json.load(open(os.path.join(ROOT, "data", "meta.json")))["chars"]
    val = np.fromfile(os.path.join(ROOT, "data", "val.bin"), dtype=np.uint8)

    os.makedirs(args.out_dir, exist_ok=True)
    export_model(model, chars, os.path.join(args.out_dir, "shakespeare.bin"))
    export_test_vectors(model, val, os.path.join(args.out_dir, "test_vectors.bin"))

    loss, n = reference_val_loss(model, val, args.windows)
    ref = {"val_loss": round(loss, 4), "perplexity": round(math.exp(loss), 4), "windows": n,
           "seq_len": cfg.max_seq_len, "train_iter": ckpt["iter"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.reference)), exist_ok=True)
    with open(args.reference, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"PyTorch reference: {ref}")


if __name__ == "__main__":
    main()
