//! Command-line interface: generate text, benchmark, measure perplexity.
//!
//!     quill generate [--prompt TEXT] [--steps N] [--temp T] [--seed S] [--q8]
//!     quill bench    [--steps N] [--q8] [--threads N]
//!     quill ppl      [--windows N] [--q8]
//!     quill report   [--steps N] [--windows N] [--out results/benchmarks.json]
//!
//! Common options: --model models/shakespeare.bin  --data data/val.bin  --threads N

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use quill::sampler::{argmax, sample, Rng};
use quill::tokenizer::Tokenizer;
use quill::Transformer;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Args {
    command: String,
    opts: HashMap<String, String>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut it = std::env::args().skip(1);
        let command = it.next().unwrap_or_else(|| "help".into());
        let mut opts = HashMap::new();
        let rest: Vec<String> = it.collect();
        let mut i = 0;
        while i < rest.len() {
            let key = rest[i]
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument {}", rest[i]))?;
            if key == "q8" {
                opts.insert(key.into(), "true".into());
                i += 1;
            } else {
                let val = rest
                    .get(i + 1)
                    .ok_or_else(|| format!("missing value for --{key}"))?;
                opts.insert(key.into(), val.clone());
                i += 2;
            }
        }
        Ok(Args { command, opts })
    }

    fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T> {
        match self.opts.get(key) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("invalid value for --{key}: {v}").into()),
        }
    }

    fn str(&self, key: &str, default: &str) -> String {
        self.opts
            .get(key)
            .cloned()
            .unwrap_or_else(|| default.into())
    }

    fn q8(&self) -> bool {
        self.opts.contains_key("q8")
    }
}

fn load(path: &str, q8: bool) -> Result<Transformer> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let model = Transformer::from_bytes(&bytes)?;
    Ok(if q8 { model.quantize() } else { model })
}

fn with_threads<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> Result<T> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build()?;
    Ok(pool.install(f))
}

fn generate(model: &mut Transformer, prompt: &str, steps: usize, temp: f32, seed: u64) {
    let tok = Tokenizer::new(&model.vocab);
    let mut tokens = tok.encode(prompt);
    if tokens.is_empty() {
        tokens.push(0); // the vocabulary's first entry is "\n"
    }
    let mut rng = Rng::new(seed);
    let mut out = std::io::stdout().lock();
    for &t in &tokens {
        let _ = write!(out, "{}", tok.decode(t));
    }

    let max = model.config.max_seq_len;
    let start = Instant::now();
    let (mut pos, mut generated) = (0, 0);
    let mut i = 0;
    while generated < steps {
        if pos == max {
            // Context full: restart the cache and re-read the last half.
            let keep = tokens.len() - max / 2;
            tokens.drain(..keep);
            pos = 0;
            i = 0;
        }
        let logits = model.forward(tokens[i], pos);
        pos += 1;
        i += 1;
        if i == tokens.len() {
            let next = sample(logits, temp, &mut rng);
            tokens.push(next);
            let _ = write!(out, "{}", tok.decode(next));
            let _ = out.flush();
            generated += 1;
        }
    }
    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "\n\n[{steps} tokens in {secs:.2}s, {:.0} tok/s]",
        steps as f64 / secs
    );
}

/// Greedy decoding from an empty context; returns tokens per second.
fn bench(model: &mut Transformer, steps: usize) -> f64 {
    let steps = steps.min(model.config.max_seq_len);
    let run = |model: &mut Transformer| {
        let start = Instant::now();
        let mut token = 0;
        for pos in 0..steps {
            token = argmax(model.forward(token, pos));
        }
        steps as f64 / start.elapsed().as_secs_f64()
    };
    run(model); // warm-up
    let mut runs: Vec<f64> = (0..5).map(|_| run(model)).collect();
    runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    runs[runs.len() / 2]
}

struct Eval {
    loss: f64,
    /// Fraction of positions where the argmax matches `reference`'s argmax.
    agreement: Option<f64>,
}

/// Mean next-token cross-entropy over non-overlapping windows of the
/// validation set (same windows as train/export.py's reference loss).
fn evaluate(
    model: &mut Transformer,
    data: &[u8],
    windows: usize,
    mut reference: Option<&mut Transformer>,
) -> Eval {
    let t = model.config.max_seq_len;
    let windows = windows.min((data.len() - 1) / t);
    let (mut nll, mut agree) = (0.0f64, 0usize);
    for w in 0..windows {
        let chunk = &data[w * t..w * t + t + 1];
        for pos in 0..t {
            let logits = model.forward(chunk[pos] as usize, pos);
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = max
                + logits
                    .iter()
                    .map(|&l| (l as f64 - max).exp())
                    .sum::<f64>()
                    .ln();
            nll += lse - logits[chunk[pos + 1] as usize] as f64;
            if let Some(r) = reference.as_deref_mut() {
                let mine = argmax(logits);
                if argmax(r.forward(chunk[pos] as usize, pos)) == mine {
                    agree += 1;
                }
            }
        }
    }
    let n = (windows * t) as f64;
    Eval {
        loss: nll / n,
        agreement: reference.map(|_| agree as f64 / n),
    }
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1e6
}

fn report(args: &Args) -> Result<()> {
    let model_path = args.str("model", "models/shakespeare.bin");
    let data = std::fs::read(args.str("data", "data/val.bin"))?;
    let steps: usize = args.get("steps", 256)?;
    let windows: usize = args.get("windows", 64)?;
    let max_threads: usize = args.get("threads", std::thread::available_parallelism()?.get())?;
    let out_path = args.str("out", "results/benchmarks.json");

    let mut thread_counts = vec![1];
    let mut n = 2;
    while n <= max_threads {
        thread_counts.push(n);
        n *= 2;
    }
    if *thread_counts.last().unwrap() != max_threads {
        thread_counts.push(max_threads);
    }

    let mut rows = Vec::new();
    let mut fp32 = load(&model_path, false)?;
    for q8 in [false, true] {
        let mut model = load(&model_path, q8)?;
        let (linear, other) = model.weight_bytes();
        let eval = with_threads(max_threads, || {
            evaluate(
                &mut model,
                &data,
                windows,
                if q8 { Some(&mut fp32) } else { None },
            )
        })?;
        let mut speeds = Vec::new();
        for &th in &thread_counts {
            let tps = with_threads(th, || bench(&mut model, steps))?;
            eprintln!(
                "{} threads={th}: {tps:.0} tok/s",
                if q8 { "int8" } else { "fp32" }
            );
            speeds.push((th, tps));
        }
        rows.push((q8, linear, other, eval, speeds));
    }

    // JSON by hand to keep the crate dependency-free.
    let mut json = String::from("{\n  \"results\": [\n");
    for (i, (q8, linear, other, eval, speeds)) in rows.iter().enumerate() {
        let speeds_json: Vec<String> = speeds
            .iter()
            .map(|(t, s)| format!("{{\"threads\": {t}, \"tokens_per_s\": {s:.1}}}"))
            .collect();
        json += &format!(
            "    {{\"precision\": \"{}\", \"linear_weight_bytes\": {linear}, \"other_weight_bytes\": {other}, \
             \"val_loss\": {:.4}, \"perplexity\": {:.4}, \"argmax_agreement_with_fp32\": {}, \"speed\": [{}]}}{}\n",
            if *q8 { "int8" } else { "fp32" },
            eval.loss,
            eval.loss.exp(),
            eval.agreement.map_or("null".into(), |a| format!("{a:.4}")),
            speeds_json.join(", "),
            if i + 1 < rows.len() { "," } else { "" }
        );
    }
    json += &format!("  ],\n  \"steps\": {steps},\n  \"windows\": {windows}\n}}\n");
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&out_path, json)?;

    println!(
        "| precision | linear weights | val loss | perplexity | top-1 agreement vs fp32 | {} |",
        thread_counts
            .iter()
            .map(|t| format!("tok/s ({t} thr)"))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    println!(
        "|---|---|---|---|---|{}",
        "---|".repeat(thread_counts.len())
    );
    for (q8, linear, _, eval, speeds) in &rows {
        println!(
            "| {} | {:.2} MB | {:.4} | {:.3} | {} | {} |",
            if *q8 { "int8" } else { "fp32" },
            mb(*linear),
            eval.loss,
            eval.loss.exp(),
            eval.agreement
                .map_or("—".into(), |a| format!("{:.1}%", a * 100.0)),
            speeds
                .iter()
                .map(|(_, s)| format!("{s:.0}"))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }
    eprintln!("wrote {out_path}");
    Ok(())
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    let model_path = args.str("model", "models/shakespeare.bin");
    let threads: usize = args.get("threads", std::thread::available_parallelism()?.get())?;

    match args.command.as_str() {
        "generate" => {
            let mut model = load(&model_path, args.q8())?;
            let prompt = args.str("prompt", "ROMEO:\n");
            let (steps, temp, seed) = (
                args.get("steps", 500)?,
                args.get("temp", 0.8)?,
                args.get("seed", 42)?,
            );
            with_threads(threads, || generate(&mut model, &prompt, steps, temp, seed))?;
        }
        "bench" => {
            let mut model = load(&model_path, args.q8())?;
            let steps = args.get("steps", 256)?;
            let tps = with_threads(threads, || bench(&mut model, steps))?;
            let (linear, other) = model.weight_bytes();
            println!(
                "{}: {tps:.0} tok/s ({:.3} ms/token), {threads} threads, linear weights {:.2} MB (+{:.2} MB embeddings/norms)",
                if model.is_quantized() { "int8" } else { "fp32" },
                1000.0 / tps,
                mb(linear),
                mb(other)
            );
        }
        "ppl" => {
            let mut model = load(&model_path, args.q8())?;
            let data = std::fs::read(args.str("data", "data/val.bin"))?;
            let windows = args.get("windows", 64)?;
            let eval = with_threads(threads, || evaluate(&mut model, &data, windows, None))?;
            println!(
                "val loss {:.4}, perplexity {:.4}",
                eval.loss,
                eval.loss.exp()
            );
        }
        "report" => report(&args)?,
        _ => {
            eprintln!("usage: quill <generate|bench|ppl|report> [options]  (see src/main.rs)");
            std::process::exit(2);
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
