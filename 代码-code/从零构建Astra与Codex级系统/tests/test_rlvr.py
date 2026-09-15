from __future__ import annotations

import torch

from astra_codex.rlvr import group_relative_advantages, grpo_style_objective


def test_group_relative_advantages_are_centered_per_prompt() -> None:
    rewards = torch.tensor([[1.0, 0.0, -1.0], [5.0, 5.0, 5.0]])
    advantages = group_relative_advantages(rewards)

    torch.testing.assert_close(advantages[0].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    assert advantages[0, 0] > advantages[0, 1] > advantages[0, 2]
    torch.testing.assert_close(advantages[1], torch.zeros(3))


def test_grpo_style_objective_is_zero_value_but_has_useful_gradient_at_old_policy() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1, 3))
    old = torch.zeros(1, 3)
    rewards = torch.tensor([[1.0, 0.0, -1.0]])

    result = grpo_style_objective(parameter, old, rewards, kl_beta=0.0)
    torch.testing.assert_close(result.loss.detach(), torch.tensor(0.0), atol=1e-6, rtol=0)

    result.loss.backward()
    assert parameter.grad is not None
    assert parameter.grad[0, 0] < 0  # gradient descent raises high-reward log-prob
    assert parameter.grad[0, 2] > 0  # gradient descent lowers low-reward log-prob


def test_grpo_style_clipping_and_kl_are_explicit() -> None:
    new = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
    old = torch.zeros(1, 3)
    reference = torch.zeros(1, 3)
    rewards = torch.tensor([[1.0, 0.0, -1.0]])

    result = grpo_style_objective(
        new,
        old,
        rewards,
        clip_epsilon=0.2,
        reference_logprobs=reference,
        kl_beta=0.1,
    )

    assert result.clip_fraction > 0
    assert result.kl_penalty > 0
    assert torch.isfinite(result.loss)
