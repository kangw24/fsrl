"""Diagnostics for presentation-side symmetry of ranking decisions."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LeftRightEquivarianceReport:
    mean_probability_error: float
    max_probability_error: float
    mean_margin_error: float
    max_margin_error: float


def left_right_equivariance_report(
    original_logits: torch.Tensor,
    mirrored_logits: torch.Tensor,
) -> LeftRightEquivarianceReport:
    """Compare original P(left) with mirrored P(right).

    A perfectly presentation-equivariant ranker swaps its two output channels
    when the query items swap sides.  Margin error is also reported because it
    is invariant to adding an arbitrary common offset to both logits.
    """

    if original_logits.shape != mirrored_logits.shape:
        raise ValueError("original and mirrored logits must have the same shape")
    if original_logits.shape[-1] != 2:
        raise ValueError("left-right diagnostics require two-choice logits")

    original_prob = torch.softmax(original_logits, dim=-1)[..., 0]
    mirrored_prob = torch.softmax(mirrored_logits, dim=-1)[..., 1]
    probability_error = torch.abs(original_prob - mirrored_prob)

    original_margin = original_logits[..., 0] - original_logits[..., 1]
    mirrored_margin = mirrored_logits[..., 0] - mirrored_logits[..., 1]
    margin_error = torch.abs(original_margin + mirrored_margin)
    return LeftRightEquivarianceReport(
        mean_probability_error=float(probability_error.mean().detach().cpu()),
        max_probability_error=float(probability_error.max().detach().cpu()),
        mean_margin_error=float(margin_error.mean().detach().cpu()),
        max_margin_error=float(margin_error.max().detach().cpu()),
    )
