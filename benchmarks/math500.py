import random
from datasets import load_dataset

from utils.answer_extraction import extract_boxed_answers, strip_string


def load_math500(seed=42):
    """Load MATH-500: 500 problems sampled from EleutherAI/hendrycks_math test splits.
    Returns list of {"question": str, "answer": str, "level": str, "type": str}.
    """
    configs = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    all_items = []
    for config in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="test")
        all_items.extend(list(ds))

    rng = random.Random(seed)
    sampled = rng.sample(all_items, min(500, len(all_items)))

    examples = []
    for item in sampled:
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
