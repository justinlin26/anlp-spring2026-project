import re
from datasets import load_dataset


def _parse_gsm8k_split(split):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    examples = []
    for item in ds:
        answer_text = item["answer"]
        match = re.search(r"####\s*(.+)", answer_text)
        if match:
            answer = match.group(1).strip().replace(",", "")
        else:
            answer = answer_text.strip()
        examples.append({
            "question": item["question"],
            "answer": answer,
            "raw_answer": answer_text,
        })
    return examples


def load_gsm8k():
    """Load GSM8K test split (1,319 examples).
    Returns list of {"question": str, "answer": str} where answer is the numeric value.
    """
    return _parse_gsm8k_split("test")


def load_gsm8k_train():
    """Load GSM8K train split (7,473 examples). Same schema as load_gsm8k."""
    return _parse_gsm8k_split("train")
