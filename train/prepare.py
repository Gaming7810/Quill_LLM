"""Download Tiny Shakespeare and encode it as a character-level dataset.

Writes data/train.bin and data/val.bin (one uint8 token id per character)
and data/meta.json (the vocabulary, ordered by token id).
"""

import json
import os
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(path):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)

    with open(path, encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint8)

    # First 90% for training, last 10% held out for validation / perplexity.
    split = int(0.9 * len(ids))
    ids[:split].tofile(os.path.join(DATA_DIR, "train.bin"))
    ids[split:].tofile(os.path.join(DATA_DIR, "val.bin"))
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump({"chars": chars}, f)

    print(f"{len(text):,} chars, vocab {len(chars)}, train {split:,}, val {len(ids) - split:,}")


if __name__ == "__main__":
    main()
