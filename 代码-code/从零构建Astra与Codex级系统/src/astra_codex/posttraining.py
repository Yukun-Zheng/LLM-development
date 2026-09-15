from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class SequenceLogProbs:
    total: torch.Tensor
    token_count: torch.Tensor

    @property
    def mean(self) -> torch.Tensor:
        return self.total / self.token_count.clamp_min(1)


@dataclass(frozen=True, slots=True)
class DPOResult:
    loss: torch.Tensor
    chosen_reward: torch.Tensor
    rejected_reward: torch.Tensor
    preference_margin: torch.Tensor
    preference_accuracy: torch.Tensor


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Next-token cross entropy with explicit causal shift.

    ``logits[:, t]`` predicts token ``labels[:, t + 1]``. Setting a label to
    ``ignore_index`` removes that target token from the supervised objective,
    which is how an instruction-tuning batch can mask prompt/user tokens while
    training only on assistant tokens.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if labels.ndim != 2 or labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [B, T] matching logits")
    if logits.shape[1] < 2:
        raise ValueError("causal LM loss requires at least two sequence positions")

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
    )


def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> SequenceLogProbs:
    """Return summed supervised-token log-probability for each sequence."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if labels.ndim != 2 or labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [B, T] matching logits")
    if logits.shape[1] < 2:
        raise ValueError("sequence_logprobs requires at least two positions")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != ignore_index
    safe_labels = shifted_labels.masked_fill(~mask, 0)

    token_logprobs = F.log_softmax(shifted_logits.float(), dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_logprobs = token_logprobs * mask
    return SequenceLogProbs(
        total=token_logprobs.sum(dim=-1),
        token_count=mask.sum(dim=-1),
    )


def sft_batch_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    output = model(input_ids)
    return causal_lm_loss(output.logits, labels)


def sft_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_grad_norm: float | None = None,
) -> float:
    """One inspectable supervised fine-tuning optimization step."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = sft_batch_loss(model, input_ids, labels)
    loss.backward()
    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.detach())


def dpo_from_logprobs(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    *,
    beta: float = 0.1,
) -> DPOResult:
    """Direct Preference Optimization objective from sequence log-probs.

    For a chosen response y_w and rejected response y_l, DPO optimizes the
    logistic margin between policy and frozen-reference preference ratios.
    This function exposes that scalar math separately from model execution so it
    can be hand-checked and parity-tested.
    """

    if beta <= 0:
        raise ValueError("beta must be positive")
    tensors = [policy_chosen, policy_rejected, reference_chosen, reference_rejected]
    if any(tensor.shape != policy_chosen.shape for tensor in tensors[1:]):
        raise ValueError("all log-prob tensors must have matching shape")

    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin).mean()
    return DPOResult(
        loss=loss,
        chosen_reward=chosen_reward,
        rejected_reward=rejected_reward,
        preference_margin=margin,
        preference_accuracy=(margin > 0).float().mean(),
    )


def dpo_batch_loss(
    policy: nn.Module,
    reference: nn.Module,
    *,
    chosen_input_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
) -> DPOResult:
    policy_chosen = sequence_logprobs(
        policy(chosen_input_ids).logits, chosen_labels
    ).total
    policy_rejected = sequence_logprobs(
        policy(rejected_input_ids).logits, rejected_labels
    ).total

    reference.eval()
    with torch.no_grad():
        reference_chosen = sequence_logprobs(
            reference(chosen_input_ids).logits, chosen_labels
        ).total
        reference_rejected = sequence_logprobs(
            reference(rejected_input_ids).logits, rejected_labels
        ).total

    return dpo_from_logprobs(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=beta,
    )


def dpo_step(
    policy: nn.Module,
    reference: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    chosen_input_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
    max_grad_norm: float | None = None,
) -> DPOResult:
    """One from-scratch DPO optimization step with a frozen reference model."""

    policy.train()
    reference.eval()
    optimizer.zero_grad(set_to_none=True)
    result = dpo_batch_loss(
        policy,
        reference,
        chosen_input_ids=chosen_input_ids,
        chosen_labels=chosen_labels,
        rejected_input_ids=rejected_input_ids,
        rejected_labels=rejected_labels,
        beta=beta,
    )
    result.loss.backward()
    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return DPOResult(
        loss=result.loss.detach(),
        chosen_reward=result.chosen_reward.detach(),
        rejected_reward=result.rejected_reward.detach(),
        preference_margin=result.preference_margin.detach(),
        preference_accuracy=result.preference_accuracy.detach(),
    )
