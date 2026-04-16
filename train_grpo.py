"""GRPO training with an explicit CoT-length regularizer.

Implements the multi-objective loss
    reward = 1{answer correct} - LAMBDA * (completion_length / max_completion_length)
via TRL's GRPOTrainer with two reward functions and reward_weights=[1.0, -LAMBDA].

Usage:
  python train_grpo.py --lambda_len 0.1 --output_dir checkpoints/grpo_lam0.1
  python train_grpo.py --lambda_len 0.3 --max_steps 500 --output_dir checkpoints/grpo_lam0.3_short

Then evaluate with:
  python evaluate.py --method grpo --benchmark gsm8k \
      --model Qwen/Qwen2.5-3B-Instruct --adapter checkpoints/grpo_lam0.1
"""

import argparse
import os

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from benchmarks import TRAIN_LOADERS
from utils.prompts import grpo_chat_prompt
from utils.rewards import accuracy_reward, make_length_reward


def build_dataset(benchmark, max_examples=None):
    loader = TRAIN_LOADERS[benchmark]
    examples = loader()
    if max_examples:
        examples = examples[:max_examples]
    return Dataset.from_list([
        {"prompt": grpo_chat_prompt(ex["question"]), "answer": ex["answer"]}
        for ex in examples
    ])


def main():
    parser = argparse.ArgumentParser(description="GRPO training with length-penalized reward")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--benchmark", choices=list(TRAIN_LOADERS.keys()), default="gsm8k")
    parser.add_argument("--lambda_len", type=float, required=True,
                        help="Weight on the length penalty. 0 = pure accuracy; 1 = length weighs as much as correctness.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Group size G in GRPO.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Override num_train_epochs with a fixed step count.")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature for rollouts. Higher = more exploration.")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL coefficient. TRL default is 0.0 (no KL); 0.04 adds a small anchor to the reference.")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3,
                        help="Fraction of GPU memory vLLM may use during colocated rollouts.")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Truncate training set (for smoke tests).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--report_to", type=str, default="none",
                        help="Logger: 'wandb', 'tensorboard', or 'none'.")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Benchmark: {args.benchmark}")
    print(f"LAMBDA (length weight): {args.lambda_len}")
    print(f"Output: {args.output_dir}")

    dataset = build_dataset(args.benchmark, max_examples=args.max_examples)
    print(f"Train examples: {len(dataset)}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name or os.path.basename(args.output_dir.rstrip("/")),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        reward_weights=[1.0, -args.lambda_len],
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        seed=args.seed,
        report_to=args.report_to,
        gradient_checkpointing=True,
    )

    reward_funcs = [
        accuracy_reward,
        make_length_reward(args.max_completion_length),
    ]
    reward_funcs[0].__name__ = "accuracy"
    reward_funcs[1].__name__ = "length_norm"

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
