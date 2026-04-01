# CoT Compression Baseline Evaluation

Baseline evaluation framework for the 11-711 project on efficient Chain-of-Thought reasoning.
Implements and compares three methods for reducing CoT token usage while preserving accuracy:

1. **Standard CoT** -- Verbose step-by-step few-shot prompting
2. **Chain of Draft (CoD)** -- Concise-draft few-shot prompting ([Xu et al., 2025](https://arxiv.org/abs/2502.18600))
3. **TokenSkip** -- LoRA-based controllable CoT compression ([Xia et al., 2025](https://github.com/hemingkx/TokenSkip))

## Project Structure

```
.
├── evaluate.py              # Main evaluation entry point
├── run_baselines.sh         # Runs all 9 baseline evaluations
├── requirements.txt
├── benchmarks/
│   ├── gsm8k.py             # GSM8K test loader (1,319 examples)
│   ├── math500.py           # MATH-500 loader (500 sampled from competition_math)
│   └── svamp.py             # SVAMP test loader (~1,000 examples)
├── configs/
│   ├── gsm8k_cot.yaml       # 8-shot verbose CoT prompts
│   ├── gsm8k_cod.yaml       # 8-shot Chain of Draft prompts
│   ├── math500_cot.yaml     # 4-shot CoT with \boxed{} format
│   ├── math500_cod.yaml     # 4-shot CoD with \boxed{} format
│   ├── svamp_cot.yaml       # 8-shot CoT (GSM8K-style)
│   └── svamp_cod.yaml       # 8-shot CoD (GSM8K-style)
├── utils/
│   ├── answer_extraction.py  # Answer parsing (boxed, ####, numeric fallback)
│   └── metrics.py            # Accuracy, token counts, result saving
└── results/                  # Output directory (created automatically)
```

## Setup

Requires Python 3.10+ and a CUDA-capable GPU. Tested on a single A100 40GB.

```bash
pip install -r requirements.txt
```

Key dependencies: `vllm`, `torch`, `transformers`, `peft`, `datasets`.

## Usage

### Run all baselines at once

```bash
bash run_baselines.sh
```

This executes 9 evaluations:
- Standard CoT on GSM8K, MATH-500, SVAMP
- Chain of Draft on GSM8K, MATH-500, SVAMP
- TokenSkip on GSM8K at compression ratios 0.7, 0.5, 0.3

### Run individual evaluations

```bash
# Standard Chain-of-Thought
python evaluate.py --method cot --benchmark gsm8k --model Qwen/Qwen2.5-3B-Instruct

# Chain of Draft
python evaluate.py --method cod --benchmark gsm8k --model Qwen/Qwen2.5-3B-Instruct

# TokenSkip (requires --adapter and --compression_ratio)
python evaluate.py --method tokenskip --benchmark gsm8k \
    --model Qwen/Qwen2.5-3B-Instruct \
    --adapter hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K \
    --compression_ratio 0.5
```

### Quick smoke test

```bash
python evaluate.py --method cot --benchmark gsm8k \
    --model Qwen/Qwen2.5-3B-Instruct --max_examples 5
```

### CLI arguments

| Argument | Description | Default |
|---|---|---|
| `--method` | `cot`, `cod`, or `tokenskip` | required |
| `--benchmark` | `gsm8k`, `math500`, or `svamp` | required |
| `--model` | HuggingFace model ID | `Qwen/Qwen2.5-3B-Instruct` |
| `--adapter` | LoRA adapter path (TokenSkip only) | `None` |
| `--compression_ratio` | Token compression ratio (TokenSkip only) | `1.0` |
| `--max_new_tokens` | Max generation length | `512` |
| `--temperature` | Sampling temperature | `0.0` |
| `--seed` | Random seed | `42` |
| `--max_examples` | Limit examples for debugging | `None` (all) |
| `--output_dir` | Override output directory | auto-generated |

## Output

Each run produces a directory under `results/` containing:

**`metrics.json`** -- aggregate results:
```json
{
    "method": "cot",
    "benchmark": "gsm8k",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "n_samples": 1319,
    "accuracy": 0.72,
    "avg_cot_tokens": 245.3,
    "total_time_s": 120.5,
    "latency_per_sample_s": 0.091
}
```

**`predictions.jsonl`** -- per-example results with question, expected answer, predicted answer, full model output, and token count.

## Methods

### Standard CoT

Uses 8-shot (GSM8K/SVAMP) or 4-shot (MATH-500) few-shot prompts with verbose step-by-step reasoning examples. Answers are extracted via `####` separator (GSM8K/SVAMP) or `\boxed{}` (MATH-500).

### Chain of Draft

Same few-shot structure as CoT but with a modified system prompt: *"Think step by step, but only keep minimum draft for each thinking step, with 5 words at most."* Few-shot examples use concise arithmetic drafts instead of verbose explanations. This is a pure prompting method with no model changes.

### TokenSkip

Uses a pre-trained LoRA adapter from [hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K](https://huggingface.co/hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K) that was fine-tuned to skip low-utility reasoning tokens. A compression ratio (0.0--1.0) is embedded in the prompt as a special token to control how aggressively the model compresses its reasoning. Currently only supports GSM8K with the pre-trained adapter.

## Benchmarks

| Benchmark | Source | Size | Answer Format |
|---|---|---|---|
| GSM8K | `openai/gsm8k` | 1,319 test | Numeric after `####` |
| MATH-500 | `hendrycks/competition_math` | 500 sampled (seed=42) | `\boxed{}` |
| SVAMP | `ChilleD/SVAMP` | ~1,000 test | Numeric |

## References

- Xu et al. (2025). *Chain of Draft: Thinking Faster by Writing Less*. [arXiv:2502.18600](https://arxiv.org/abs/2502.18600)
- Xia et al. (2025). *TokenSkip: Controllable Chain-of-Thought Compression in LLMs*. [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.165/)
- Cobbe et al. (2021). *Training Verifiers to Solve Math Word Problems*. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- Hendrycks et al. (2021). *Measuring Mathematical Problem Solving With the MATH Dataset*. NeurIPS 2021.
- Patel et al. (2021). *Are NLP Models really Able to Solve Simple Math Word Problems?* [arXiv:2103.07191](https://arxiv.org/abs/2103.07191)
