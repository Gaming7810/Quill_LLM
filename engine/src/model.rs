//! Model loading and the transformer forward pass (one token at a time, with
//! a KV cache). Mirrors train/model.py operation by operation.

use crate::ops::{matmul, rmsnorm, silu, softmax};
use crate::quant::{matmul_q8, QTensor};

const MAGIC: &[u8; 4] = b"QUIL";
const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
}

/// Weight storage for a linear layer: full precision or int8.
pub enum Linear {
    F32 {
        w: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    Q8(QTensor),
}

impl Linear {
    fn quantize(self) -> Linear {
        match self {
            Linear::F32 { w, rows, cols } => Linear::Q8(QTensor::from_f32(&w, rows, cols)),
            q => q,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Linear::F32 { cols, .. } => *cols,
            Linear::Q8(t) => t.cols,
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            Linear::F32 { w, .. } => w.len() * 4,
            Linear::Q8(t) => t.size_bytes(),
        }
    }

    /// `out = W x`. For Q8 weights, `x` is first quantized into `xq`.
    fn forward(&self, out: &mut [f32], x: &[f32], xq: &mut QTensor) {
        match self {
            Linear::F32 { w, .. } => matmul(out, w, x),
            Linear::Q8(w) => {
                xq.quantize_from(x);
                matmul_q8(out, w, xq);
            }
        }
    }
}

pub struct Layer {
    pub attention_norm: Vec<f32>,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ffn_norm: Vec<f32>,
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
}

impl Layer {
    fn linears(&self) -> [&Linear; 7] {
        [
            &self.wq, &self.wk, &self.wv, &self.wo, &self.w1, &self.w2, &self.w3,
        ]
    }
}

pub struct Weights {
    /// Used for the embedding lookup; always kept in f32.
    pub tok_embeddings: Vec<f32>,
    pub layers: Vec<Layer>,
    pub norm: Vec<f32>,
    /// Output projection, tied to `tok_embeddings` in the trained model.
    pub output: Linear,
}

#[derive(Debug)]
pub struct LoadError(pub String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid model file: {}", self.0)
    }
}

impl std::error::Error for LoadError {}

/// Sequential little-endian reader over the model file.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], LoadError> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end =
            end.ok_or_else(|| LoadError(format!("unexpected end of file at byte {}", self.pos)))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, LoadError> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, LoadError> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, LoadError> {
        let b = self.bytes(n * 4)?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    fn linear(&mut self, rows: usize, cols: usize) -> Result<Linear, LoadError> {
        Ok(Linear::F32 {
            w: self.f32s(rows * cols)?,
            rows,
            cols,
        })
    }
}

pub struct Transformer {
    pub config: Config,
    pub vocab: Vec<String>,
    pub weights: Weights,
    state: RunState,
}

/// Activation buffers and the KV cache, allocated once.
struct RunState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    /// (layer, position, dim)
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    /// Quantized copies of the activation vectors fed to Q8 matmuls.
    xq_dim: QTensor,
    xq_hidden: QTensor,
    /// Precomputed RoPE tables, (position, head_dim / 2).
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl RunState {
    fn new(c: &Config) -> Self {
        let kv = c.n_layers * c.max_seq_len * c.dim;
        let half = c.head_dim() / 2;
        let mut rope_cos = Vec::with_capacity(c.max_seq_len * half);
        let mut rope_sin = Vec::with_capacity(c.max_seq_len * half);
        for pos in 0..c.max_seq_len {
            for i in 0..half {
                let freq = 1.0 / c.rope_theta.powf((2 * i) as f32 / c.head_dim() as f32);
                let angle = pos as f32 * freq;
                rope_cos.push(angle.cos());
                rope_sin.push(angle.sin());
            }
        }
        RunState {
            x: vec![0.0; c.dim],
            xb: vec![0.0; c.dim],
            xb2: vec![0.0; c.dim],
            hb: vec![0.0; c.hidden_dim],
            hb2: vec![0.0; c.hidden_dim],
            q: vec![0.0; c.dim],
            att: vec![0.0; c.n_heads * c.max_seq_len],
            logits: vec![0.0; c.vocab_size],
            key_cache: vec![0.0; kv],
            value_cache: vec![0.0; kv],
            xq_dim: QTensor::zeros(1, c.dim),
            xq_hidden: QTensor::zeros(1, c.hidden_dim),
            rope_cos,
            rope_sin,
        }
    }
}

impl Transformer {
    /// Parse a model file produced by train/export.py.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, LoadError> {
        let mut r = Reader { buf, pos: 0 };
        if r.bytes(4)? != MAGIC {
            return Err(LoadError("bad magic, expected \"QUIL\"".into()));
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(LoadError(format!("unsupported version {version}")));
        }
        let mut dims = [0usize; 6];
        for d in dims.iter_mut() {
            *d = r.u32()? as usize;
        }
        let [dim, hidden_dim, n_layers, n_heads, vocab_size, max_seq_len] = dims;
        let config = Config {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            vocab_size,
            max_seq_len,
            norm_eps: r.f32()?,
            rope_theta: r.f32()?,
        };
        if n_heads == 0 || !dim.is_multiple_of(n_heads) || !config.head_dim().is_multiple_of(2) {
            return Err(LoadError(format!(
                "dim {dim} not divisible into {n_heads} even-sized heads"
            )));
        }

        let mut vocab = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let len = r.bytes(1)?[0] as usize;
            let s = std::str::from_utf8(r.bytes(len)?).map_err(|e| LoadError(e.to_string()))?;
            vocab.push(s.to_string());
        }

        let tok_embeddings = r.f32s(vocab_size * dim)?;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(Layer {
                attention_norm: r.f32s(dim)?,
                wq: r.linear(dim, dim)?,
                wk: r.linear(dim, dim)?,
                wv: r.linear(dim, dim)?,
                wo: r.linear(dim, dim)?,
                ffn_norm: r.f32s(dim)?,
                w1: r.linear(hidden_dim, dim)?,
                w2: r.linear(dim, hidden_dim)?,
                w3: r.linear(hidden_dim, dim)?,
            });
        }
        let norm = r.f32s(dim)?;
        if r.pos != buf.len() {
            return Err(LoadError(format!("{} trailing bytes", buf.len() - r.pos)));
        }
        let output = Linear::F32 {
            w: tok_embeddings.clone(),
            rows: vocab_size,
            cols: dim,
        };

        Ok(Transformer {
            state: RunState::new(&config),
            config,
            vocab,
            weights: Weights {
                tok_embeddings,
                layers,
                norm,
                output,
            },
        })
    }

    /// Convert every linear layer to int8. The embedding table used for the
    /// lookup stays f32; the (tied) output projection is quantized.
    pub fn quantize(mut self) -> Self {
        let w = &mut self.weights;
        for l in w.layers.iter_mut() {
            for lin in [
                &mut l.wq, &mut l.wk, &mut l.wv, &mut l.wo, &mut l.w1, &mut l.w2, &mut l.w3,
            ] {
                let owned = std::mem::replace(
                    lin,
                    Linear::F32 {
                        w: Vec::new(),
                        rows: 0,
                        cols: 0,
                    },
                );
                *lin = owned.quantize();
            }
        }
        let out = std::mem::replace(
            &mut w.output,
            Linear::F32 {
                w: Vec::new(),
                rows: 0,
                cols: 0,
            },
        );
        w.output = out.quantize();
        self
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self.weights.output, Linear::Q8(_))
    }

    /// Bytes of weights touched per generated token (the embedding table is
    /// only read one row at a time, so it is counted separately).
    pub fn weight_bytes(&self) -> (usize, usize) {
        let w = &self.weights;
        let mut linear = w.output.size_bytes();
        let mut other = w.tok_embeddings.len() * 4 + w.norm.len() * 4;
        for l in &w.layers {
            linear += l.linears().iter().map(|x| x.size_bytes()).sum::<usize>();
            other += (l.attention_norm.len() + l.ffn_norm.len()) * 4;
        }
        (linear, other)
    }

    /// Run one token at position `pos` and return the logits for the next one.
    /// Positions must be fed in order 0, 1, 2, ... (the KV cache fills up as
    /// we go); start again from 0 to begin a new sequence.
    pub fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        let c = &self.config;
        assert!(token < c.vocab_size, "token {token} out of range");
        assert!(
            pos < c.max_seq_len,
            "position {pos} exceeds max_seq_len {}",
            c.max_seq_len
        );
        let (dim, hd, half) = (c.dim, c.head_dim(), c.head_dim() / 2);
        let w = &self.weights;
        let s = &mut self.state;

        s.x.copy_from_slice(&w.tok_embeddings[token * dim..(token + 1) * dim]);

        for (li, l) in w.layers.iter().enumerate() {
            // ---- attention ----
            rmsnorm(&mut s.xb, &s.x, &l.attention_norm, c.norm_eps);

            let kv_off = (li * c.max_seq_len + pos) * dim;
            l.wq.forward(&mut s.q, &s.xb, &mut s.xq_dim);
            l.wk.forward(&mut s.key_cache[kv_off..kv_off + dim], &s.xb, &mut s.xq_dim);
            l.wv.forward(
                &mut s.value_cache[kv_off..kv_off + dim],
                &s.xb,
                &mut s.xq_dim,
            );

            // Rotary embeddings on q and the freshly cached k.
            let cos = &s.rope_cos[pos * half..(pos + 1) * half];
            let sin = &s.rope_sin[pos * half..(pos + 1) * half];
            for v in [&mut s.q[..], &mut s.key_cache[kv_off..kv_off + dim]] {
                for head in v.chunks_exact_mut(hd) {
                    for i in 0..half {
                        let (a, b) = (head[2 * i], head[2 * i + 1]);
                        head[2 * i] = a * cos[i] - b * sin[i];
                        head[2 * i + 1] = a * sin[i] + b * cos[i];
                    }
                }
            }

            // Causal attention of this position over positions 0..=pos.
            let layer_off = li * c.max_seq_len * dim;
            let scale = 1.0 / (hd as f32).sqrt();
            for h in 0..c.n_heads {
                let q = &s.q[h * hd..(h + 1) * hd];
                let att = &mut s.att[h * c.max_seq_len..h * c.max_seq_len + pos + 1];
                for (t, a) in att.iter_mut().enumerate() {
                    let k = &s.key_cache[layer_off + t * dim + h * hd..][..hd];
                    *a = crate::ops::dot(q, k) * scale;
                }
                softmax(att);
                let out = &mut s.xb[h * hd..(h + 1) * hd];
                out.fill(0.0);
                for (t, &a) in att.iter().enumerate() {
                    let v = &s.value_cache[layer_off + t * dim + h * hd..][..hd];
                    for (o, &vi) in out.iter_mut().zip(v) {
                        *o += a * vi;
                    }
                }
            }

            l.wo.forward(&mut s.xb2, &s.xb, &mut s.xq_dim);
            for (x, d) in s.x.iter_mut().zip(&s.xb2) {
                *x += d;
            }

            // ---- feed-forward (SwiGLU) ----
            rmsnorm(&mut s.xb, &s.x, &l.ffn_norm, c.norm_eps);
            l.w1.forward(&mut s.hb, &s.xb, &mut s.xq_dim);
            l.w3.forward(&mut s.hb2, &s.xb, &mut s.xq_dim);
            for (h, &g) in s.hb.iter_mut().zip(&s.hb2) {
                *h = silu(*h) * g;
            }
            l.w2.forward(&mut s.xb2, &s.hb, &mut s.xq_hidden);
            for (x, d) in s.x.iter_mut().zip(&s.xb2) {
                *x += d;
            }
        }

        rmsnorm(&mut s.xb, &s.x, &w.norm, c.norm_eps);
        w.output.forward(&mut s.logits, &s.xb, &mut s.xq_dim);
        &s.logits
    }
}
