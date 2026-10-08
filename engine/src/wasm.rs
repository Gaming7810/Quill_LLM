//! A tiny C-style API for running the engine from JavaScript (see docs/app.js).
//! No wasm-bindgen: the browser copies the model file into memory obtained
//! from `quill_alloc`, then drives generation one token at a time.

use std::cell::RefCell;

use crate::sampler::{sample, Rng};
use crate::Transformer;

thread_local! {
    static ENGINE: RefCell<Option<(Transformer, Rng)>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn quill_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Loads a model from `len` bytes at `ptr` (which is freed). Returns 0 on success.
///
/// # Safety
/// `ptr` must come from `quill_alloc(len)` and hold `len` initialized bytes.
#[no_mangle]
pub unsafe extern "C" fn quill_load(ptr: *mut u8, len: usize, quantize: u32) -> i32 {
    let bytes = Vec::from_raw_parts(ptr, len, len);
    match Transformer::from_bytes(&bytes) {
        Ok(model) => {
            let model = if quantize != 0 {
                model.quantize()
            } else {
                model
            };
            ENGINE.with(|e| *e.borrow_mut() = Some((model, Rng::new(1))));
            0
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn quill_seed(seed: u32) {
    ENGINE.with(|e| {
        if let Some((_, rng)) = e.borrow_mut().as_mut() {
            *rng = Rng::new(seed as u64);
        }
    });
}

#[no_mangle]
pub extern "C" fn quill_max_seq_len() -> u32 {
    ENGINE.with(|e| {
        e.borrow()
            .as_ref()
            .map_or(0, |(m, _)| m.config.max_seq_len as u32)
    })
}

/// Feeds `token` at position `pos` and returns a sampled next token
/// (or u32::MAX if no model is loaded).
#[no_mangle]
pub extern "C" fn quill_step(token: u32, pos: u32, temperature: f32) -> u32 {
    ENGINE.with(|e| match e.borrow_mut().as_mut() {
        Some((model, rng)) => {
            let logits = model.forward(token as usize, pos as usize);
            sample(logits, temperature, rng) as u32
        }
        None => u32::MAX,
    })
}
