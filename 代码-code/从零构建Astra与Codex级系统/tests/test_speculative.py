from __future__ import annotations

import torch

from astra_codex.speculative import (
    one_step_output_distribution,
    residual_distribution,
    verify_speculative_block,
)


def test_one_step_speculative_marginal_exactly_matches_target() -> None:
    target = torch.tensor([0.55, 0.25, 0.20])
    draft = torch.tensor([0.20, 0.50, 0.30])

    output = one_step_output_distribution(target, draft)

    torch.testing.assert_close(output, target, atol=1e-6, rtol=1e-6)


def test_residual_distribution_only_places_mass_where_target_exceeds_draft() -> None:
    target = torch.tensor([0.60, 0.10, 0.30])
    draft = torch.tensor([0.20, 0.50, 0.30])
    residual = residual_distribution(target, draft)

    torch.testing.assert_close(residual, torch.tensor([1.0, 0.0, 0.0]))


def test_rejection_emits_correction_and_stops_verifying_later_draft_tokens() -> None:
    draft_tokens = torch.tensor([1, 2])
    draft_probs = torch.tensor(
        [
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
        ]
    )
    target_probs = torch.tensor(
        [
            [0.90, 0.05, 0.05],
            [0.20, 0.20, 0.60],
            [0.30, 0.30, 0.40],
        ]
    )

    result = verify_speculative_block(
        draft_tokens,
        draft_probs,
        target_probs,
        uniforms=torch.tensor([0.99, 0.0]),
        generator=torch.Generator().manual_seed(0),
    )

    assert result.rejected_at == 0
    assert result.accepted_draft_tokens == 0
    assert not result.used_target_bonus_token
    # After rejecting token 1, residual target-draft mass exists only on token 0.
    assert result.output_tokens.tolist() == [0]


def test_all_accepted_block_adds_one_target_bonus_token() -> None:
    draft_tokens = torch.tensor([0, 1])
    draft_probs = torch.tensor(
        [
            [0.70, 0.30],
            [0.40, 0.60],
        ]
    )
    # Draft candidate probabilities are no larger than target probabilities for
    # the sampled tokens, so acceptance probability is exactly 1 for both.
    target_probs = torch.tensor(
        [
            [0.80, 0.20],
            [0.30, 0.70],
            [1.00, 0.00],
        ]
    )

    result = verify_speculative_block(
        draft_tokens,
        draft_probs,
        target_probs,
        uniforms=torch.tensor([1.0, 1.0]),
        generator=torch.Generator().manual_seed(0),
    )

    assert result.rejected_at is None
    assert result.accepted_draft_tokens == 2
    assert result.used_target_bonus_token
    assert result.output_tokens.tolist() == [0, 1, 0]


def test_matching_draft_and_target_has_no_correction_bias() -> None:
    target = torch.tensor([0.1, 0.2, 0.7])
    output = one_step_output_distribution(target, target.clone())
    torch.testing.assert_close(output, target, atol=1e-6, rtol=1e-6)
