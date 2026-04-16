"""Unified evaluation script for CoT compression baselines.

Supports four methods:
  - cot:       Standard Chain-of-Thought few-shot prompting
  - cod:       Chain of Draft (concise few-shot prompting)
  - tokenskip: TokenSkip with LoRA adapter + compression ratio control
  - grpo:      Our length-regularized GRPO LoRA (trained via train_grpo.py)

Usage:
  python evaluate.py --method cot --benchmark gsm8k --model Qwen/Qwen2.5-3B-Instruct
  python evaluate.py --method cod --benchmark gsm8k --model Qwen/Qwen2.5-3B-Instruct
  python evaluate.py --method tokenskip --benchmark gsm8k --model Qwen/Qwen2.5-3B-Instruct \
      --adapter hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K --compression_ratio 0.5
  python evaluate.py --method grpo --benchmark gsm8k --model Qwen/Qwen2.5-3B-Instruct \
      --adapter checkpoints/grpo_gsm8k_lam0.3
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

from benchmarks import BENCHMARK_LOADERS
from utils.answer_extraction import extract_gsm8k_answer, extract_math_answer
from utils.metrics import compute_accuracy, compute_avg_tokens, save_results
from utils.prompts import grpo_chat_prompt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompt_config(benchmark, method):
    """Load YAML prompt config for a given benchmark and method."""
    config_path = os.path.join("configs", f"{benchmark}_{method}.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_fewshot_prompt(config, question):
    """Build a few-shot prompt from a YAML config and a question."""
    prompt = config["system_prompt"].strip() + "\n"
    fmt = config["format"]
    for ex in config.get("fewshot", []):
        prompt += fmt.format(
            question=ex["question"].strip(),
            answer=ex["answer"].strip(),
        )
        prompt += "\n"
    prompt += fmt.format(question=question.strip(), answer="")
    return prompt


def build_tokenskip_prompt(question, compression_ratio):
    """Build a Qwen-style chat prompt with TokenSkip compression ratio token.

    Follows the TokenSkip repo format for Qwen models:
    <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|eot_id|>{ratio}<|eot_id|><|im_end|>\n<|im_start|>assistant\n
    """
    if compression_ratio < 1.0:
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "Please reason step by step, and put your final answer within \\boxed{}.\n"
            f"{question}<|eot_id|>{compression_ratio}<|eot_id|><|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "Please reason step by step, and put your final answer within \\boxed{}.\n"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return prompt


def build_grpo_prompt(question, model_path):
    """Apply the model's chat template to the zero-shot GRPO prompt. Matches training format."""
    from transformers import AutoTokenizer
    tok = getattr(build_grpo_prompt, "_tok", None)
    if tok is None or getattr(build_grpo_prompt, "_tok_path", None) != model_path:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        build_grpo_prompt._tok = tok
        build_grpo_prompt._tok_path = model_path
    return tok.apply_chat_template(
        grpo_chat_prompt(question),
        tokenize=False,
        add_generation_prompt=True,
    )


def build_prompts(data, method, benchmark, model_path=None, compression_ratio=1.0):
    """Build prompts for all examples."""
    if method == "tokenskip":
        return [build_tokenskip_prompt(ex["question"], compression_ratio) for ex in data]
    elif method == "grpo":
        return [build_grpo_prompt(ex["question"], model_path) for ex in data]
    else:
        config = load_prompt_config(benchmark, method)
        return [build_fewshot_prompt(config, ex["question"]) for ex in data]


def get_answer_extractor(benchmark, method):
    """Return the appropriate answer extraction function."""
    if method in ("tokenskip", "grpo") or benchmark == "math500":
        return extract_math_answer
    else:
        return extract_gsm8k_answer


def run_vllm_inference(prompts, model_path, method, adapter_path=None,
                       compression_ratio=1.0, max_new_tokens=512,
                       temperature=0.0):
    """Run batched inference using vLLM."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    use_lora = method in ("tokenskip", "grpo") and adapter_path is not None
    # For few-shot CoT/CoD, stop before the model hallucinates a follow-on Q&A pair
    stop_sequences = ["\nQ:"] if method in ("cot", "cod") else []

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
    )
    if use_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        llm_kwargs["tensor_parallel_size"] = gpu_count

    print(f"Loading model: {model_path}")
    if use_lora:
        print(f"LoRA adapter: {adapter_path}")
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=stop_sequences,
        n=1,
    )

    print(f"Running inference on {len(prompts)} prompts...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    if use_lora:
        lora_request = LoRARequest(method, 1, adapter_path)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    results = []
    for output in outputs:
        text = output.outputs[0].text
        token_count = len(output.outputs[0].token_ids)
        results.append({"text": text, "token_count": token_count})

    return results, total_time


def main():
    parser = argparse.ArgumentParser(description="Evaluate CoT compression baselines")
    parser.add_argument("--method", choices=["cot", "cod", "tokenskip", "grpo"], required=True)
    parser.add_argument("--benchmark", choices=["gsm8k", "math500", "svamp"], required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=str, default=None,
                        help="HuggingFace adapter path for TokenSkip")
    parser.add_argument("--compression_ratio", type=float, default=1.0,
                        help="Compression ratio for TokenSkip (0.0-1.0)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples for debugging")
    args = parser.parse_args()

    if args.method in ("tokenskip", "grpo") and args.adapter is None:
        parser.error(f"--adapter is required for {args.method} method")

    set_seed(args.seed)

    if args.method == "tokenskip" and args.benchmark == "math500":
        args.max_new_tokens = int(args.max_new_tokens * args.compression_ratio)

    print(f"Method: {args.method}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Model: {args.model}")
    if args.method == "tokenskip":
        print(f"Adapter: {args.adapter}")
        print(f"Compression ratio: {args.compression_ratio}")
    elif args.method == "grpo":
        print(f"Adapter: {args.adapter}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed}")
    print()

    print("Loading benchmark data...")
    loader = BENCHMARK_LOADERS[args.benchmark]
    data = loader()
    if args.max_examples:
        data = data[:args.max_examples]
    print(f"Loaded {len(data)} examples")

    print("Building prompts...")
    prompts = build_prompts(
        data,
        args.method,
        args.benchmark,
        model_path=args.model,
        compression_ratio=args.compression_ratio,
    )
    print(f"Example prompt (first):\n{'-'*40}\n{prompts[0][:500]}...\n{'-'*40}\n")

    raw_outputs, total_time = run_vllm_inference(
        prompts=prompts,
        model_path=args.model,
        method=args.method,
        adapter_path=args.adapter,
        compression_ratio=args.compression_ratio,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    extractor = get_answer_extractor(args.benchmark, args.method)
    predictions = []
    extracted_answers = []
    token_counts = []

    for i, (example, output) in enumerate(zip(data, raw_outputs)):
        pred_answer = extractor(output["text"])
        extracted_answers.append(pred_answer)
        token_counts.append(output["token_count"])
        predictions.append({
            "id": i,
            "question": example["question"],
            "expected_answer": example["answer"],
            "predicted_answer": pred_answer,
            "model_output": output["text"],
            "cot_tokens": output["token_count"],
        })

    reference_answers = [ex["answer"] for ex in data]
    accuracy = compute_accuracy(extracted_answers, reference_answers)
    avg_tokens = compute_avg_tokens(token_counts)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        if args.method == "tokenskip":
            suffix = f"_{args.compression_ratio}"
        elif args.method == "grpo":
            suffix = f"_{os.path.basename(args.adapter.rstrip('/'))}"
        else:
            suffix = ""
        output_dir = os.path.join("results", f"{args.method}_{args.benchmark}{suffix}")

    save_results(
        output_dir=output_dir,
        method=args.method,
        benchmark=args.benchmark,
        model=args.model,
        accuracy=accuracy,
        avg_cot_tokens=avg_tokens,
        total_time_s=total_time,
        n_samples=len(data),
        predictions=predictions,
        compression_ratio=args.compression_ratio if args.method == "tokenskip" else None,
    )


if __name__ == "__main__":
    main()
