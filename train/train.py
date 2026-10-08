"""Train the character-level transformer on Tiny Shakespeare (CPU is enough).

    python train/prepare.py
    python train/train.py --max-iters 2000
"""

import argparse
import csv
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from model import Config, Transformer

ROOT = os.path.join(os.path.dirname(__file__), "..")


def get_batch(data, batch_size, seq_len):
    ix = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    x = torch.from_numpy(np.stack([data[i : i + seq_len] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + seq_len] for i in ix]).astype(np.int64))
    return x, y


@torch.no_grad()
def estimate_loss(model, splits, batch_size, seq_len, iters):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, batch_size, seq_len))[1].item() for _ in range(iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def lr_at(it, args):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    progress = (it - args.warmup) / max(1, args.max_iters - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


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
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default=os.path.join(ROOT, "train", "out"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

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
    model = Transformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"config {asdict(cfg)}")
    print(f"{n_params / 1e6:.2f}M parameters")

    # No weight decay on norms (1-D params).
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    log_path = os.path.join(ROOT, "results", "train_log.csv")
    log = open(log_path, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["iter", "train_loss", "val_loss", "lr", "elapsed_s"])

    splits = {"train": train_data, "val": val_data}
    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        lr = lr_at(it, args)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, splits, args.batch_size, cfg.max_seq_len, args.eval_iters)
            elapsed = time.time() - t0
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.0f}s")
            writer.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.6f}", f"{elapsed:.1f}"])
            log.flush()
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"config": asdict(cfg), "model": model.state_dict(), "iter": it, "val_loss": best_val},
                           os.path.join(args.out, "ckpt.pt"))
        if it == args.max_iters:
            break

        x, y = get_batch(train_data, args.batch_size, cfg.max_seq_len)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    log.close()
    print(f"best val loss {best_val:.4f}, checkpoint in {args.out}")


if __name__ == "__main__":
    main()
