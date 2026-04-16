"""Shared prompt format used by GRPO training and eval.

Keeping this in one place ensures the trained adapter is evaluated with the
same prompt distribution it was trained on.
"""

GRPO_SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the problem step by step and put your "
    "final answer within \\boxed{}."
)


def grpo_chat_prompt(question):
    """Return the conversational prompt (list of messages) for GRPO train/eval."""
    return [
        {"role": "system", "content": GRPO_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
