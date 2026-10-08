//! The engine must reproduce PyTorch's logits for the exported model.
//! Test vectors are written by train/export.py next to the model file.

use std::path::PathBuf;

use quill::sampler::argmax;
use quill::Transformer;

fn model_dir() -> PathBuf {
    std::env::var_os("QUILL_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models"))
}

struct Vectors {
    tokens: Vec<usize>,
    vocab: usize,
    logits: Vec<f32>,
}

fn load_vectors() -> Vectors {
    let b = std::fs::read(model_dir().join("test_vectors.bin")).expect("run train/export.py first");
    assert_eq!(&b[..4], b"QTV1");
    let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
    let (n, vocab) = (u32_at(4), u32_at(8));
    let tokens = (0..n).map(|i| u32_at(12 + 4 * i)).collect();
    let off = 12 + 4 * n;
    let logits = b[off..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(logits.len(), n * vocab);
    Vectors {
        tokens,
        vocab,
        logits,
    }
}

fn load_model() -> Transformer {
    let bytes =
        std::fs::read(model_dir().join("shakespeare.bin")).expect("run train/export.py first");
    Transformer::from_bytes(&bytes).unwrap()
}

/// Returns (max |logit difference|, fraction of positions with the same argmax).
fn compare(model: &mut Transformer, v: &Vectors) -> (f32, f32) {
    let (mut max_diff, mut same) = (0.0f32, 0);
    for (pos, &tok) in v.tokens.iter().enumerate() {
        let got = model.forward(tok, pos);
        let want = &v.logits[pos * v.vocab..(pos + 1) * v.vocab];
        for (g, w) in got.iter().zip(want) {
            max_diff = max_diff.max((g - w).abs());
        }
        same += (argmax(got) == argmax(want)) as usize;
    }
    (max_diff, same as f32 / v.tokens.len() as f32)
}

#[test]
fn fp32_matches_pytorch() {
    let v = load_vectors();
    let mut model = load_model();
    let (max_diff, agreement) = compare(&mut model, &v);
    println!("fp32: max |Δlogit| = {max_diff:.2e}, argmax agreement {agreement:.3}");
    assert!(max_diff < 1e-3, "max |Δlogit| = {max_diff}");
    assert_eq!(agreement, 1.0);
}

#[test]
fn int8_stays_close_to_pytorch() {
    let v = load_vectors();
    let mut model = load_model().quantize();
    let (max_diff, agreement) = compare(&mut model, &v);
    println!("int8: max |Δlogit| = {max_diff:.2e}, argmax agreement {agreement:.3}");
    assert!(agreement >= 0.9, "argmax agreement {agreement}");
}

#[test]
fn kv_cache_restart_is_deterministic() {
    let v = load_vectors();
    let mut model = load_model();
    let first: Vec<f32> = v
        .tokens
        .iter()
        .take(8)
        .enumerate()
        .map(|(p, &t)| model.forward(t, p)[0])
        .collect();
    let again: Vec<f32> = v
        .tokens
        .iter()
        .take(8)
        .enumerate()
        .map(|(p, &t)| model.forward(t, p)[0])
        .collect();
    assert_eq!(first, again);
}

#[test]
fn rejects_truncated_file() {
    let bytes = std::fs::read(model_dir().join("shakespeare.bin")).unwrap();
    assert!(Transformer::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    assert!(Transformer::from_bytes(b"NOPE").is_err());
}
