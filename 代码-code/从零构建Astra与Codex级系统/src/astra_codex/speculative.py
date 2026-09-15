from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SpeculativeVerification:
    output_tokens: torch.Tensor
    accepted_draft_tokens: int
    rejected_at: int | None
    used_target_bonus_token: bool


def normalize_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 1:
        raise ValueError("probabilities must have shape [V]")
    if torch.any(probabilities < 0):
        raise ValueError("probabilities must be non-negative")
    total = probabilities.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("probabilities must have positive finite mass")
    return probabilities / total


def residual_distribution(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Normalized positive residual used after speculative rejection.

    r(x) ∝ max(p(x) - q(x), 0).

    If p == q exactly, the rejection branch has zero probability and therefore
    this residual is never needed. We reject such calls explicitly rather than
    inventing an arbitrary fallback distribution.
    """

    p = normalize_probabilities(target_probabilities)
    q = normalize_probabilities(draft_probabilities)
    if p.shape != q.shape:
        raise ValueError("target and draft distributions must have the same shape")
    residual = torch.clamp(p - q, min=0)
    mass = residual.sum()
    if mass <= 0:
        raise ValueError("residual distribution has zero mass; target and draft match")
    return residual / mass


def acceptance_probability(target_probability: torch.Tensor, draft_probability: torch.Tensor) -> torch.Tensor:
    """Return min(1, p(y)/q(y)) for a sampled draft token y."""

    if target_probability.ndim != 0 or draft_probability.ndim != 0:
        raise ValueError("acceptance_probability expects scalar tensors")
    if target_probability < 0 or draft_probability <= 0:
        raise ValueError("target must be >=0 and draft must be >0")
    return torch.minimum(
        torch.ones((), device=target_probability.device, dtype=target_probability.dtype),
        target_probability / draft_probability,
    )


def one_step_output_distribution(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Analytically marginalize one speculative draft step.

    This function is deliberately deterministic and is useful for proving the
    key invariant of speculative sampling: after accept/reject correction, the
    marginal distribution of the emitted token is exactly the target p.
    """

    p = normalize_probabilities(target_probabilities).float()
    q = normalize_probabilities(draft_probabilities).float()
    if p.shape != q.shape:
        raise ValueError("target and draft distributions must have the same shape")

    accepted_mass = torch.minimum(p, q)
    rejection_mass = 1.0 - accepted_mass.sum()
    if rejection_mass <= torch.finfo(p.dtype).eps:
        return accepted_mass / accepted_mass.sum()

    residual = torch.clamp(p - q, min=0)
    residual = residual / residual.sum()
    return accepted_mass + rejection_mass * residual


def _sample(probabilities: torch.Tensor, generator: torch.Generator | None) -> int:
    probabilities = normalize_probabilities(probabilities).float()
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def verify_speculative_block(
    draft_tokens: torch.Tensor,
    draft_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    uniforms: torch.Tensor | None = None,
) -> SpeculativeVerification:
    """Reference accept/reject verifier for one draft block.

    Args:
        draft_tokens: ``[K]`` candidate token ids sampled sequentially from q.
        draft_probabilities: ``[K, V]`` q distributions used for those samples.
        target_probabilities: ``[K+1, V]`` target distributions. Row i scores
            draft token i given the accepted prefix; row K is the bonus target
            distribution used if every draft token is accepted.
        uniforms: optional deterministic ``[K]`` U(0,1) values for tests.

    Returns:
        Accepted draft prefix plus exactly one correction token after the first
        rejection, or accepted draft block plus one target bonus token when all
        candidates are accepted.

    This function implements the probability correction only. A model-facing
    engine must additionally manage draft/target KV states and truncate caches
    to the accepted prefix after a rejection.
    """

    if draft_tokens.ndim != 1:
        raise ValueError("draft_tokens must have shape [K]")
    if draft_probabilities.ndim != 2:
        raise ValueError("draft_probabilities must have shape [K, V]")
    if target_probabilities.ndim != 2:
        raise ValueError("target_probabilities must have shape [K+1, V]")
    k = int(draft_tokens.shape[0])
    if k <= 0:
        raise ValueError("draft block must contain at least one token")
    if draft_probabilities.shape[0] != k:
        raise ValueError("draft probability row count must equal K")
    if target_probabilities.shape[0] != k + 1:
        raise ValueError("target probability row count must equal K+1")
    if draft_probabilities.shape[1] != target_probabilities.shape[1]:
        raise ValueError("draft/target vocab sizes must match")
    if uniforms is not None and (uniforms.ndim != 1 or uniforms.shape[0] != k):
        raise ValueError("uniforms must have shape [K]")

    accepted: list[int] = []
    for index in range(k):
        token = int(draft_tokens[index])
        if token < 0 or token >= draft_probabilities.shape[1]:
            raise ValueError(f"draft token out of vocabulary: {token}")
        q = normalize_probabilities(draft_probabilities[index])
        p = normalize_probabilities(target_probabilities[index])
        q_token = q[token]
        if q_token <= 0:
            raise ValueError("draft token has zero probability under its recorded q")
        accept_prob = acceptance_probability(p[token], q_token)
        draw = (
            float(uniforms[index])
            if uniforms is not None
            else float(torch.rand((), generator=generator))
        )
        if not 0.0 <= draw <= 1.0:
            raise ValueError("uniform acceptance draws must lie in [0,1]")

        if draw <= float(accept_prob):
            accepted.append(token)
            continue

        correction = _sample(residual_distribution(p, q), generator)
        accepted.append(correction)
        return SpeculativeVerification(
            output_tokens=torch.tensor(accepted, dtype=torch.long),
            accepted_draft_tokens=index,
            rejected_at=index,
            used_target_bonus_token=False,
        )

    bonus = _sample(target_probabilities[k], generator)
    accepted.append(bonus)
    return SpeculativeVerification(
        output_tokens=torch.tensor(accepted, dtype=torch.long),
        accepted_draft_tokens=k,
        rejected_at=None,
        used_target_bonus_token=True,
    )
