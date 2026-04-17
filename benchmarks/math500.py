import random
from datasets import load_dataset

from utils.answer_extraction import extract_boxed_answers, strip_string


_MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def _parse_math_items(items):
    examples = []
    for item in items:
        boxed = extract_boxed_answers(item["solution"])
        if boxed:
            answer = strip_string(boxed[-1])
        else:
            answer = item["solution"].strip()
        examples.append({
            "question": item["problem"],
            "answer": answer,
            "level": item.get("level", ""),
            "type": item.get("type", ""),
            "raw_solution": item["solution"],
        })
    return examples


def load_math500(seed=42):
    """Load MATH-500: 500 problems sampled from EleutherAI/hendrycks_math test splits.
    Returns list of {"question": str, "answer": str, "level": str, "type": str}.
    """
    all_items = []
    for config in _MATH_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="test")
        all_items.extend(list(ds))

    rng = random.Random(seed)
    sampled = rng.sample(all_items, min(500, len(all_items)))
    return _parse_math_items(sampled)


def load_math_train():
    """Load the full MATH train split (~7,500 problems) for GRPO training.
    Same schema as load_math500; answers are already \\boxed{}-normalized.
    """
    all_items = []
    for config in _MATH_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="train")
        all_items.extend(list(ds))
    return _parse_math_items(all_items)
