"""Reward functions for GRPO training.

TRL's GRPOTrainer sums reward functions with equal weight. We bake LAMBDA
directly into the length penalty so the trainer's implicit sum yields
    total_reward = 1{correct} - LAMBDA * (len / max_completion_length)

Signature follows TRL's GRPOTrainer contract: reward functions accept `completions`,
`completion_ids`, and any dataset columns (e.g. `answer`) as kwargs, and return a
list of floats with length equal to the batch.
"""

from utils.answer_extraction import answers_equal, extract_math_answer


def _completion_text(completion):
    """Extract text from a completion (conversational or standard format)."""
    if isinstance(completion, list):
        return completion[0].get("content", "")
    return completion


def accuracy_reward(completions, answer, **kwargs):
    """1.0 if the extracted boxed answer matches the ground-truth answer, else 0.0.

    `answer` is a list of ground-truth strings passed in from the dataset column.
    """
    rewards = []
    for comp, gt in zip(completions, answer):
        pred = extract_math_answer(_completion_text(comp))
        rewards.append(1.0 if answers_equal(pred, gt) else 0.0)
    return rewards


def make_length_penalty(lambda_len, max_completion_length, tokenizer):
    """Build a length penalty reward of -lambda * (len / max_completion_length).

    Tokenizes each completion with the supplied tokenizer to measure length in
    tokens (TRL 0.14's reward-fn API passes `completions` but not `completion_ids`).
    When summed with accuracy_reward under TRL's default equal weighting, the
    total reward is `correct - lambda_len * (len / max_completion_length)`.
    """

    def length_penalty(completions, **kwargs):
        penalties = []
        for comp in completions:
            n_tokens = len(tokenizer.encode(_completion_text(comp), add_special_tokens=False))
            penalties.append(-lambda_len * min(n_tokens / max_completion_length, 1.0))
        return penalties

    return length_penalty
