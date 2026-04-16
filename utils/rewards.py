"""Reward functions for GRPO training.

Used together with `reward_weights=[1.0, -LAMBDA]` in GRPOConfig to implement
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


def make_length_reward(max_completion_length):
    """Build a length reward that returns normalized completion length in [0, 1].

    Combined with reward_weights=[..., -LAMBDA], yields a penalty of
    LAMBDA * (len / max_completion_length) per sample.
    """

    def length_reward(completion_ids, **kwargs):
        return [min(len(ids) / max_completion_length, 1.0) for ids in completion_ids]

    return length_reward
