from datasets import load_dataset


def load_svamp():
    """Load SVAMP test split (~1,000 examples).
    Returns list of {"question": str, "answer": str}.
    """
    ds = load_dataset("ChilleD/SVAMP", split="test")
    examples = []
    for item in ds:
        body = item.get("Body", "")
        question_text = item.get("Question", "")
        full_question = f"{body} {question_text}".strip() if body else question_text
        answer = str(item.get("Answer", "")).strip()
        answer = answer.replace(",", "")
        if answer.endswith(".0"):
            answer = answer[:-2]
        examples.append({
            "question": full_question,
            "answer": answer,
        })
    return examples
