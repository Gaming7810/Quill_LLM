"""Train the character-level transformer on Tiny Shakespeare.

Runs on a CUDA GPU when one is available (bf16 mixed precision), otherwise on
the CPU, where the default config takes about 30 minutes on 4 cores.

    python train/prepare.py
    python train/train.py                      # auto-selects the device
    python train/train.py --device cpu         # force CPU
    python train/train.py --compile            # torch.compile (needs Triton on GPU)
"""

import argparse
import contextlib
import csv
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from model import Config, Transformer

ROOT = os.path.join(os.path.dirname(__file__), "..")
# The Rust engine quantizes in groups of 32 values (engine/src/quant.rs).
QUANT_GROUP = 32


def get_batch(data, batch_size, seq_len, device):
    ix = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    x = torch.from_numpy(np.stack([data[i : i + seq_len] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + seq_len] for i in ix]).astype(np.int64))
    if device.type == "cuda":
        # Pinned memory lets the copy overlap with GPU work.
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, batch_size, seq_len, iters, device, autocast):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = []
        for _ in range(iters):
            with autocast:
                losses.append(model(*get_batch(data, batch_size, seq_len, device))[1].item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def lr_at(it, args):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    progress = (it - args.warmup) / max(1, args.max_iters - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def pick_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"],
                   help="autocast dtype for the forward pass (auto: bfloat16 on CUDA, float32 on CPU)")
    p.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    p.add_argument("--threads", type=int, default=os.cpu_count(), help="CPU threads (CPU training only)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default=os.path.join(ROOT, "train", "out"))
    p.add_argument("--log", default=os.path.join(ROOT, "results", "train_log.csv"))
    args = p.parse_args()

    for name in ("dim", "hidden_dim"):
        if getattr(args, name) % QUANT_GROUP:
            p.error(f"--{name.replace('_', '-')} must be a multiple of {QUANT_GROUP} for the engine's int8 path")
    if args.dim % args.n_heads or (args.dim // args.n_heads) % 2:
        p.error("--dim must split into --n-heads heads of even size (needed by RoPE)")

    device = pick_device(args.device)
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if device.type == "cuda" else "float32"
    # Weights and optimizer state stay in fp32; only the forward pass runs in
    # bf16, so no gradient scaling is needed.
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if dtype == "bfloat16"
        else contextlib.nullcontext()
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    else:
        # TF32 matmuls for any op that autocast leaves in fp32.
        torch.set_float32_matmul_precision("high")

    train_data = np.memmap(os.path.join(ROOT, "data", "train.bin"), dtype=np.uint8, mode="r")
    val_data = np.memmap(os.path.join(ROOT, "data", "val.bin"), dtype=np.uint8, mode="r")
    vocab_size = int(max(train_data.max(), val_data.max())) + 1

    cfg = Config(
        vocab_size=vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    )
    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"config {asdict(cfg)}")
    print(f"{n_params / 1e6:.2f}M parameters on {device_name}, {dtype}{', compiled' if args.compile else ''}")

    # No weight decay on norms (1-D params).
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    # Train through the compiled wrapper, but always save the plain module's
    # weights so export.py sees the usual parameter names.
    train_model = torch.compile(model) if args.compile else model

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    log = open(args.log, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["iter", "train_loss", "val_loss", "lr", "elapsed_s"])

    splits = {"train": train_data, "val": val_data}
    best_val = float("inf")
    tokens_per_step = args.batch_size * cfg.max_seq_len
    t0 = time.time()
    t_last, it_last = t0, 0
    for it in range(args.max_iters + 1):
        lr = lr_at(it, args)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(train_model, splits, args.batch_size, cfg.max_seq_len, args.eval_iters,
                                   device, autocast)
            now = time.time()
            tok_s = (it - it_last) * tokens_per_step / max(now - t_last, 1e-9)
            t_last, it_last = now, it
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} | "
                  f"{now - t0:.0f}s | {tok_s / 1e3:.0f}k tok/s")
            writer.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.6f}", f"{now - t0:.1f}"])
            log.flush()
            if losses["val"] < best_val:
                best_val = losses["val"]
                state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save({"config": asdict(cfg), "model": state, "iter": it, "val_loss": best_val},
                           os.path.join(args.out, "ckpt.pt"))
        if it == args.max_iters:
            break

        x, y = get_batch(train_data, args.batch_size, cfg.max_seq_len, device)
        with autocast:
            _, loss = train_model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    log.close()
    print(f"best val loss {best_val:.4f}, checkpoint in {args.out}")


if __name__ == "__main__":
    main()
