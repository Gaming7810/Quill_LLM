//! quill: a from-scratch inference engine for a small Llama-style transformer.
//!
//! - [`model`]: file loading and the forward pass with a KV cache
//! - [`quant`]: group-wise int8 quantization and integer matmul
//! - [`ops`]: dot product, matmul, RMSNorm, softmax
//! - [`sampler`], [`tokenizer`]: turning text into tokens and logits into text

pub mod model;
pub mod ops;
pub mod quant;
pub mod sampler;
pub mod tokenizer;

#[cfg(target_arch = "wasm32")]
mod wasm;

pub use model::Transformer;
