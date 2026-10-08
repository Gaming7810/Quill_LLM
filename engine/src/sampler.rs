//! Turning logits into the next token.

/// xorshift64* — small, fast, deterministic for a given seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0
}

/// Sample from softmax(logits / temperature); temperature 0 means greedy.
pub fn sample(logits: &[f32], temperature: f32, rng: &mut Rng) -> usize {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max) / temperature).exp())
        .collect();
    let total: f32 = probs.iter().sum();
    let mut r = rng.next_f32() * total;
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        assert_eq!(sample(&[0.1, 3.0, -1.0], 0.0, &mut Rng::new(1)), 1);
    }

    #[test]
    fn sampling_follows_distribution() {
        // ln(3) vs 0: token 0 should come up ~75% of the time.
        let logits = [3f32.ln(), 0.0];
        let mut rng = Rng::new(42);
        let hits = (0..10_000)
            .filter(|_| sample(&logits, 1.0, &mut rng) == 0)
            .count();
        assert!((7_200..7_800).contains(&hits), "{hits}");
    }
}
