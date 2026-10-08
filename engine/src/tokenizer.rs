//! Character-level tokenizer: one token per vocabulary entry.

use std::collections::HashMap;

pub struct Tokenizer {
    vocab: Vec<String>,
    index: HashMap<char, usize>,
}

impl Tokenizer {
    pub fn new(vocab: &[String]) -> Self {
        let index = vocab
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.chars().next().map(|c| (c, i)))
            .collect();
        Tokenizer {
            vocab: vocab.to_vec(),
            index,
        }
    }

    /// Characters outside the vocabulary are skipped.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.index.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, token: usize) -> &str {
        &self.vocab[token]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let vocab: Vec<String> = ["\n", " ", "a", "b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let t = Tokenizer::new(&vocab);
        let ids = t.encode("ab ba\nz");
        assert_eq!(ids, vec![2, 3, 1, 3, 2, 0]);
        let back: String = ids.iter().map(|&i| t.decode(i)).collect();
        assert_eq!(back, "ab ba\n");
    }
}
