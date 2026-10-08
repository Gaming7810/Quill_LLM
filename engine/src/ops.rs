//! Basic tensor operations on flat `f32` slices.

/// Dot product with 8 independent accumulators so LLVM can vectorize it.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// `out = W x`, with `W` stored row-major as (out.len(), x.len()).
pub fn matmul(out: &mut [f32], w: &[f32], x: &[f32]) {
    let n = x.len();
    debug_assert_eq!(w.len(), out.len() * n);
    for_each_row(out, |i, o| *o = dot(&w[i * n..(i + 1) * n], x));
}

/// Calls `f(row_index, &mut out[row_index])` for every row, in parallel when
/// the `parallel` feature is enabled. Rows are handed out in chunks so that
/// small matrices don't drown in scheduling overhead.
#[inline]
pub fn for_each_row<F>(out: &mut [f32], f: F)
where
    F: Fn(usize, &mut f32) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        const ROWS_PER_TASK: usize = 32;
        out.par_chunks_mut(ROWS_PER_TASK)
            .enumerate()
            .for_each(|(c, chunk)| {
                for (j, o) in chunk.iter_mut().enumerate() {
                    f(c * ROWS_PER_TASK + j, o);
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    for (i, o) in out.iter_mut().enumerate() {
        f(i, o);
    }
}

/// RMSNorm: `out = x / sqrt(mean(x^2) + eps) * weight`.
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let ms = dot(x, x) / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(weight) {
        *o = xi * inv * wi;
    }
}

/// In-place numerically stable softmax.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_naive() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5 - 3.0).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32).sin()).collect();
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - naive).abs() < 1e-4);
    }

    #[test]
    fn matmul_small() {
        // [[1, 2], [3, 4], [5, 6]] * [1, -1] = [-1, -1, -1]
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 3];
        matmul(&mut out, &w, &[1.0, -1.0]);
        assert_eq!(out, [-1.0, -1.0, -1.0]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0, 1000.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[3] > 0.99);
    }
}
