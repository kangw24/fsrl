"""Permutation distributions for constructive ranking.

Provides Mallows and Plackett-Luce samplers that operate over the
support-consistent linear extensions of the Liu 2026 partial order.
These distributions replace the ad-hoc "continuous rank scores + Gaussian
noise + hard ranking" eval path with a structured sampling of mental
permutations, which is more isomorphic to the constructive ranking account.
"""

from typing import List, Sequence, Tuple

import torch

from fsrl.device import DEVICE
from fsrl.utils.linear_extensions import generate_linear_extensions


def _kendall_tau_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Kendall tau distance between two sets of permutations.

    Args:
        p1: [..., n] permutation(s) as item order from strongest to weakest.
        p2: [..., n] permutation(s) with the same shape as p1.

    Returns:
        distance: [...] number of discordant unordered pairs divided by 2.
    """
    n = p1.shape[-1]
    # pos[b, item] = position of item in the permutation
    pos1 = torch.argsort(p1, dim=-1)
    pos2 = torch.argsort(p2, dim=-1)
    # Compare relative order of every item pair (i, j)
    pos1_i = pos1.unsqueeze(-1)  # [..., n, 1]
    pos1_j = pos1.unsqueeze(-2)  # [..., 1, n]
    pos2_i = pos2.unsqueeze(-1)
    pos2_j = pos2.unsqueeze(-2)
    cmp1 = pos1_i < pos1_j
    cmp2 = pos2_i < pos2_j
    diff = (cmp1 != cmp2).float().sum(dim=(-2, -1)) / 2.0
    return diff


def _permutation_to_scores(perm: torch.Tensor) -> torch.Tensor:
    """Convert an item permutation to rank scores.

    Args:
        perm: [..., n] item order from strongest to weakest.

    Returns:
        scores: [..., n] where strongest item gets n-1 and weakest gets 0.
    """
    positions = torch.argsort(perm, dim=-1)
    return (perm.shape[-1] - 1 - positions).float()


class PermutationDistribution:
    """Sample support-consistent permutations for eval-time constructive ranking.

    Two modes are supported:
      - ``mallows``: weighted sampling from the set of linear extensions using
        Kendall tau distance to a center permutation.
      - ``plackett_luce``: sequential sampling using item strengths, projected
        back to the support-consistent extension set.
    """

    def __init__(
        self,
        nbcues: int,
        support_pairs: Sequence[Tuple[int, int]],
        distribution: str = "mallows",
        max_extensions: int = 10000,
        seed: int = 42,
    ):
        if distribution not in ("mallows", "plackett_luce"):
            raise ValueError(f"Unknown distribution: {distribution}")
        self.nbcues = nbcues
        self.distribution = distribution
        self.support_pairs = list(support_pairs)
        self.max_extensions = max_extensions
        self._seed = seed

        # Pre-compute all (or a large sample of) support-consistent linear extensions.
        self.extensions: List[List[int]] = generate_linear_extensions(
            nbcues,
            support_pairs,
            max_extensions=max_extensions,
            seed=seed,
        )
        self.extensions_t = torch.tensor(
            self.extensions, dtype=torch.long, device=DEVICE
        )  # [E, n]

    def _project_to_extensions(self, perms: torch.Tensor) -> torch.Tensor:
        """Project arbitrary permutations to the nearest support-consistent extension.

        Args:
            perms: [bs, n] arbitrary permutations.

        Returns:
            projected: [bs, n] permutations from the pre-computed extension set.
        """
        perms = perms.to(self.extensions_t.device)
        bs = perms.shape[0]
        E = self.extensions_t.shape[0]
        # Compute Kendall tau distance from each batch permutation to each extension.
        perms_exp = perms.unsqueeze(1).expand(bs, E, self.nbcues)
        ext_exp = self.extensions_t.unsqueeze(0).expand(bs, E, self.nbcues)
        distances = _kendall_tau_distance(perms_exp.reshape(-1, self.nbcues),
                                          ext_exp.reshape(-1, self.nbcues))
        distances = distances.reshape(bs, E)
        nearest_idx = distances.argmin(dim=-1)
        return self.extensions_t[nearest_idx]

    def sample_mallows(
        self,
        center_perms: torch.Tensor,
        phi: float,
    ) -> torch.Tensor:
        """Sample permutations from a Mallows-like distribution over linear extensions.

        Args:
            center_perms: [bs, n] center permutation(s) (item order, strongest first).
            phi: dispersion parameter. Larger phi -> more concentrated around center.
                 phi <= 0 returns the center permutation unchanged.

        Returns:
            sampled_perms: [bs, n] sampled support-consistent permutations.
        """
        if phi <= 0.0:
            return center_perms.clone()

        center_perms = center_perms.to(self.extensions_t.device)
        bs = center_perms.shape[0]
        E = self.extensions_t.shape[0]
        # [bs, E] distances from each center to each extension
        center_exp = center_perms.unsqueeze(1).expand(bs, E, self.nbcues)
        ext_exp = self.extensions_t.unsqueeze(0).expand(bs, E, self.nbcues)
        distances = _kendall_tau_distance(
            center_exp.reshape(-1, self.nbcues),
            ext_exp.reshape(-1, self.nbcues),
        ).reshape(bs, E)

        # Mallows weights: w_k \propto exp(-phi * distance_k)
        # Normalize per batch element.
        log_weights = -phi * distances
        weights = torch.softmax(log_weights, dim=-1)

        # Multinomial sample one extension index per batch element.
        sampled_idx = torch.multinomial(weights, num_samples=1).squeeze(-1)
        return self.extensions_t[sampled_idx]

    def sample_plackett_luce(
        self,
        strengths: torch.Tensor,
        scale: float = 1.0,
        max_rejection_loops: int = 20,
    ) -> torch.Tensor:
        """Sample permutations using a Plackett-Luce model.

        Args:
            strengths: [bs, n] positive item strengths. Higher -> more likely to
                       appear earlier in the permutation.
            scale: temperature-like parameter. Smaller -> more deterministic.
            max_rejection_loops: number of rejection loops before projecting to
                                 the nearest valid extension.

        Returns:
            sampled_perms: [bs, n] sampled support-consistent permutations.
        """
        strengths = strengths.to(self.extensions_t.device)
        bs, n = strengths.shape
        eps = 1e-10
        # Gumbel-max sequential selection
        remaining = torch.ones(bs, n, dtype=torch.bool, device=strengths.device)
        perm = torch.zeros(bs, n, dtype=torch.long, device=strengths.device)
        neg_inf = torch.full((bs, n), float("-inf"), device=strengths.device)
        for pos in range(n):
            gumbel = -torch.log(-torch.log(
                torch.rand(bs, n, device=strengths.device).clamp(min=eps, max=1.0 - eps)
            ) + eps)
            masked = torch.where(
                remaining,
                strengths / max(scale, eps) + gumbel,
                neg_inf,
            )
            idx = masked.argmax(dim=-1)
            perm[:, pos] = idx
            remaining.scatter_(1, idx.unsqueeze(-1), False)

        # Validate / project to support-consistent extensions
        valid_mask = self._is_valid_extension(perm)
        if not valid_mask.all():
            invalid = ~valid_mask
            for _ in range(max_rejection_loops):
                if not invalid.any():
                    break
                # Deterministic fallback: use the ranking induced by strengths
                sub_strengths = strengths[invalid]
                perm[invalid] = torch.argsort(-sub_strengths, dim=-1)
                valid_mask = self._is_valid_extension(perm)
                invalid = ~valid_mask
            if invalid.any():
                perm[invalid] = self._project_to_extensions(perm[invalid])
        return perm

    def _is_valid_extension(self, perms: torch.Tensor) -> torch.Tensor:
        """Check whether permutations respect all support pairs.

        Args:
            perms: [bs, n] item order from strongest to weakest.

        Returns:
            valid: [bs] boolean mask.
        """
        perms = perms.to(self.extensions_t.device)
        pos = torch.argsort(perms, dim=-1)  # [bs, n]
        valid = torch.ones(perms.shape[0], dtype=torch.bool, device=perms.device)
        for strong, weak in self.support_pairs:
            valid = valid & (pos[:, strong] < pos[:, weak])
        return valid

    def sample(
        self,
        center_or_strengths: torch.Tensor,
        parameter: float,
    ) -> torch.Tensor:
        """Dispatch to the selected distribution.

        Args:
            center_or_strengths: [bs, n] center permutation for Mallows or item
                                  strengths for Plackett-Luce.
            parameter: phi for Mallows or scale for Plackett-Luce.

        Returns:
            permutations: [bs, n]
        """
        if self.distribution == "mallows":
            return self.sample_mallows(center_or_strengths, parameter)
        elif self.distribution == "plackett_luce":
            return self.sample_plackett_luce(center_or_strengths, parameter)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def to_scores(
        self,
        perms: torch.Tensor,
    ) -> torch.Tensor:
        """Convert sampled permutations to rank scores.

        Args:
            perms: [bs, n]

        Returns:
            scores: [bs, n]
        """
        return _permutation_to_scores(perms)
