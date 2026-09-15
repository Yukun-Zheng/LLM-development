from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GroupRelativeResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_penalty: torch.Tensor
    advantages: torch.Tensor
    ratios: torch.Tensor
    clip_fraction: torch.Tensor


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize verifier rewards within each prompt group.

    Expected shape is ``[batch, group]``: each row contains multiple rollouts
    sampled for the same prompt. This isolates *relative* quality inside a prompt
    instead of comparing reward scales across unrelated prompts.
    """

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [batch, group]")
    if rewards.shape[1] < 2:
        raise ValueError("group-relative advantages require at least two rollouts")
    if eps <= 0:
        raise ValueError("eps must be positive")

    rewards = rewards.float()
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    centered = rewards - mean
    return torch.where(std > eps, centered / std.clamp_min(eps), torch.zeros_like(centered))


def grpo_style_objective(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    reference_logprobs: torch.Tensor | None = None,
    kl_beta: float = 0.0,
) -> GroupRelativeResult:
    """Small inspectable GRPO-style clipped policy objective.

    This is a *teaching primitive*, not a claim to reproduce every detail of a
    production GRPO/RLVR recipe. Inputs are already-reduced rollout log-probs,
    which keeps the policy-ratio math visible:

        ratio = exp(log pi_new - log pi_old)
        advantage = normalize(reward within prompt group)
        policy loss = -mean(min(ratio*A, clip(ratio)*A))

    If ``reference_logprobs`` is supplied, an always-nonnegative KL estimator is
    added using ``exp(log_ref-log_new) - (log_ref-log_new) - 1``.
    """

    if clip_epsilon <= 0:
        raise ValueError("clip_epsilon must be positive")
    if kl_beta < 0:
        raise ValueError("kl_beta must be non-negative")
    if new_logprobs.shape != old_logprobs.shape or new_logprobs.shape != rewards.shape:
        raise ValueError("new/old logprobs and rewards must have matching [batch, group] shape")
    if reference_logprobs is not None and reference_logprobs.shape != new_logprobs.shape:
        raise ValueError("reference_logprobs must match policy log-prob shape")

    advantages = group_relative_advantages(rewards)
    log_ratio = new_logprobs - old_logprobs
    ratios = torch.exp(log_ratio)
    clipped_ratios = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped = ratios * advantages
    clipped = clipped_ratios * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    if reference_logprobs is None or kl_beta == 0:
        kl_penalty = torch.zeros((), device=new_logprobs.device, dtype=new_logprobs.dtype)
    else:
        ref_minus_policy = reference_logprobs - new_logprobs
        kl_estimate = torch.exp(ref_minus_policy) - ref_minus_policy - 1.0
        kl_penalty = kl_estimate.mean()

    loss = policy_loss + kl_beta * kl_penalty
    clip_fraction = (ratios.ne(clipped_ratios)).float().mean()
    return GroupRelativeResult(
        loss=loss,
        policy_loss=policy_loss,
        kl_penalty=kl_penalty,
        advantages=advantages,
        ratios=ratios,
        clip_fraction=clip_fraction,
    )
