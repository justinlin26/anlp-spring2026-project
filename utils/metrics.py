"""Metrics computation and result saving."""

import json
import os
from typing import List, Dict, Any

from utils.answer_extraction import answers_equal


def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute exact-match accuracy between predicted and reference answers."""
    if not predictions:
        return 0.0
    correct = sum(
        answers_equal(pred, ref) for pred, ref in zip(predictions, references)
    )
    return correct / len(predictions)


def compute_avg_tokens(token_counts: List[int]) -> float:
    """Compute mean token count."""
    if not token_counts:
        return 0.0
    return sum(token_counts) / len(token_counts)


def save_results(
    output_dir: str,
    method: str,
    benchmark: str,
    model: str,
    accuracy: float,
    avg_cot_tokens: float,
    total_time_s: float,
    n_samples: int,
    predictions: List[Dict[str, Any]],
    compression_ratio: float = None,
):
    """Save evaluation metrics and per-example predictions."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "method": method,
        "benchmark": benchmark,
        "model": model,
        "n_samples": n_samples,
        "accuracy": round(accuracy, 5),
        "avg_cot_tokens": round(avg_cot_tokens, 2),
        "total_time_s": round(total_time_s, 2),
        "latency_per_sample_s": round(total_time_s / max(n_samples, 1), 4),
    }
    if compression_ratio is not None:
        metrics["compression_ratio"] = compression_ratio

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(output_dir, "predictions.jsonl"), "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Results for {method} on {benchmark}")
    print(f"{'='*60}")
    print(f"  Accuracy:           {accuracy*100:.2f}%")
    print(f"  Avg CoT tokens:     {avg_cot_tokens:.1f}")
    print(f"  Total time:         {total_time_s:.1f}s")
    print(f"  Latency/sample:     {total_time_s/max(n_samples,1):.4f}s")
    if compression_ratio is not None:
        print(f"  Compression ratio:  {compression_ratio}")
    print(f"  Saved to:           {output_dir}")
    print(f"{'='*60}\n")

    return metrics
