from __future__ import annotations

import copy
import math

import torch

from astra_codex.config import ModelConfig
from astra_codex.model import DecoderOnlyTransformer
from astra_codex.posttraining import (
    IGNORE_INDEX,
    causal_lm_loss,
    dpo_batch_loss,
    dpo_from_logprobs,
    dpo_step,
    sequence_logprobs,
    sft_step,
)


def tiny_model(seed: int = 17) -> DecoderOnlyTransformer:
    torch.manual_seed(seed)
    return DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=32,
            hidden_size=24,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=48,
            max_seq_len=16,
        )
    )


def test_causal_lm_loss_masks_prompt_targets() -> None:
    logits = torch.zeros(1, 4, 5)
    labels = torch.tensor([[1, IGNORE_INDEX, 3, 4]])

    loss = causal_lm_loss(logits, labels)
    expected = torch.tensor(math.log(5.0))
    torch.testing.assert_close(loss, expected)

    logprobs = sequence_logprobs(logits, labels)
    assert logprobs.token_count.tolist() == [2]
    torch.testing.assert_close(logprobs.total, torch.tensor([-2.0 * math.log(5.0)]))


def test_sft_step_updates_tiny_model_parameters() -> None:
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]])
    before = model.lm_head.weight.detach().clone()

    loss = sft_step(model, optimizer, input_ids, labels, max_grad_norm=1.0)

    assert math.isfinite(loss)
    assert not torch.equal(before, model.lm_head.weight.detach())


def test_dpo_formula_is_log_two_when_policy_equals_reference() -> None:
    chosen = torch.tensor([-2.0, -3.0])
    rejected = torch.tensor([-4.0, -3.5])
    result = dpo_from_logprobs(chosen, rejected, chosen, rejected, beta=0.2)

    torch.testing.assert_close(result.loss, torch.tensor(math.log(2.0)))
    torch.testing.assert_close(result.preference_margin, torch.zeros(2))
    torch.testing.assert_close(result.preference_accuracy, torch.tensor(0.0))


def test_dpo_step_updates_policy_but_not_reference() -> None:
    policy = tiny_model()
    reference = copy.deepcopy(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-3)

    chosen_ids = torch.tensor([[1, 2, 3, 4, 5]])
    rejected_ids = torch.tensor([[1, 2, 7, 8, 9]])
    chosen_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]])
    rejected_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 7, 8, 9]])

    policy_before = policy.lm_head.weight.detach().clone()
    reference_before = reference.lm_head.weight.detach().clone()

    initial = dpo_batch_loss(
        policy,
        reference,
        chosen_input_ids=chosen_ids,
        chosen_labels=chosen_labels,
        rejected_input_ids=rejected_ids,
        rejected_labels=rejected_labels,
        beta=0.2,
    )
    torch.testing.assert_close(initial.loss, torch.tensor(math.log(2.0)), atol=1e-6, rtol=1e-6)

    result = dpo_step(
        policy,
        reference,
        optimizer,
        chosen_input_ids=chosen_ids,
        chosen_labels=chosen_labels,
        rejected_input_ids=rejected_ids,
        rejected_labels=rejected_labels,
        beta=0.2,
        max_grad_norm=1.0,
    )

    assert math.isfinite(float(result.loss))
    assert not torch.equal(policy_before, policy.lm_head.weight.detach())
    assert torch.equal(reference_before, reference.lm_head.weight.detach())
