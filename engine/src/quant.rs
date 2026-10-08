//! Symmetric group-wise int8 quantization ("Q8").
//!
//! Each group of `GROUP_SIZE` consecutive values shares one f32 scale:
//! `x ≈ q * scale` with `q ∈ [-127, 127]` and `scale = max|x| / 127`.
//! Both weights (once, at load time) and activations (on every matmul) are
//! quantized, so the inner loop of a matmul is an integer dot product.

use crate::ops::for_each_row;

pub const GROUP_SIZE: usize = 32;

/// A quantized row-major matrix (or vector, when `rows == 1`).
#[derive(Clone, Debug)]
pub struct QTensor {
    pub q: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl QTensor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        assert!(
            cols.is_multiple_of(GROUP_SIZE),
            "cols ({cols}) must be a multiple of {GROUP_SIZE}"
        );
        QTensor {
            q: vec![0; rows * cols],
            scales: vec![0.0; rows * cols / GROUP_SIZE],
            rows,
            cols,
        }
    }

    pub fn from_f32(x: &[f32], rows: usize, cols: usize) -> Self {
        let mut t = Self::zeros(rows, cols);
        t.quantize_from(x);
        t
    }

    /// Re-quantize `x` into this tensor's existing buffers (no allocation).
    pub fn quantize_from(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len(), self.q.len());
        for ((xg, qg), s) in x
            .chunks_exact(GROUP_SIZE)
            .zip(self.q.chunks_exact_mut(GROUP_SIZE))
            .zip(self.scales.iter_mut())
        {
            let max = xg.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = max / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for (q, &v) in qg.iter_mut().zip(xg) {
                *q = (v * inv).round() as i8;
            }
            *s = scale;
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        self.q
            .chunks_exact(GROUP_SIZE)
            .zip(&self.scales)
            .flat_map(|(qg, &s)| qg.iter().map(move |&q| q as f32 * s))
            .collect()
    }

    pub fn size_bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 4
    }
}

/// `out = W x` where both `W` (rows, cols) and `x` (1, cols) are quantized.
pub fn matmul_q8(out: &mut [f32], w: &QTensor, x: &QTensor) {
    let n = w.cols;
    let groups = n / GROUP_SIZE;
    debug_assert_eq!(x.q.len(), n);
    debug_assert_eq!(out.len(), w.rows);
    for_each_row(out, |i, o| {
        let wq = &w.q[i * n..(i + 1) * n];
        let ws = &w.scales[i * groups..(i + 1) * groups];
        let mut acc = 0.0f32;
        for g in 0..groups {
            let a = &wq[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            let b = &x.q[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            let mut isum = 0i32;
            for k in 0..GROUP_SIZE {
                isum += a[k] as i32 * b[k] as i32;
            }
            acc += isum as f32 * ws[g] * x.scales[g];
        }
        *o = acc;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matmul;

    fn pseudo_random(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn roundtrip_error_is_bounded_by_half_a_step() {
        let x = pseudo_random(4 * GROUP_SIZE, 1);
        let t = QTensor::from_f32(&x, 1, x.len());
        let y = t.dequantize();
        for (g, (xg, yg)) in x.chunks(GROUP_SIZE).zip(y.chunks(GROUP_SIZE)).enumerate() {
            let half_step = t.scales[g] / 2.0 + 1e-7;
            for (a, b) in xg.iter().zip(yg) {
                assert!((a - b).abs() <= half_step);
            }
        }
    }

    #[test]
    fn zero_group_stays_zero() {
        let t = QTensor::from_f32(&[0.0; GROUP_SIZE], 1, GROUP_SIZE);
        assert!(t.dequantize().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8_matmul_equals_f32_matmul_of_dequantized_values() {
        // The integer kernel must be exact up to f32 rounding: all the error
        // comes from quantization itself, none from the matmul.
        let (rows, cols) = (48, 4 * GROUP_SIZE);
        let (wq, xq) = (
            QTensor::from_f32(&pseudo_random(rows * cols, 7), rows, cols),
            QTensor::from_f32(&pseudo_random(cols, 9), 1, cols),
        );
        let mut want = vec![0.0; rows];
        matmul(&mut want, &wq.dequantize(), &xq.dequantize());
        let mut got = vec![0.0; rows];
        matmul_q8(&mut got, &wq, &xq);
        for (w, g) in want.iter().zip(&got) {
            assert!((w - g).abs() < 1e-4, "{w} vs {g}");
        }
    }

    #[test]
    fn q8_matmul_error_within_quantization_bound() {
        // |Σ w x − Σ w' x'| ≤ Σ (|w| δx + |x| δw + δw δx), δ = half a step.
        let (rows, cols) = (48, 4 * GROUP_SIZE);
        let w = pseudo_random(rows * cols, 7);
        let x = pseudo_random(cols, 9);
        let (wq, xq) = (
            QTensor::from_f32(&w, rows, cols),
            QTensor::from_f32(&x, 1, cols),
        );
        let mut exact = vec![0.0; rows];
        matmul(&mut exact, &w, &x);
        let mut approx = vec![0.0; rows];
        matmul_q8(&mut approx, &wq, &xq);
        for r in 0..rows {
            let bound: f32 = (0..cols)
                .map(|k| {
                    let dw = wq.scales[(r * cols + k) / GROUP_SIZE] / 2.0;
                    let dx = xq.scales[k / GROUP_SIZE] / 2.0;
                    w[r * cols + k].abs() * dx + x[k].abs() * dw + dw * dx
                })
                .sum();
            assert!((exact[r] - approx[r]).abs() <= bound + 1e-5);
        }
    }
}
