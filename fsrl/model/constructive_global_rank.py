"""ConstructiveGlobalRank-v1 (CGR-v1) and its classical magnitude controls.

Pre-registered in README section 8.  The candidate (CGR-v1) reads only the sign
of each learned pair (who is higher) and commits early to one self-consistent
global total order, with the particular order selected by a low-dimensional
idiosyncratic anchor prior + presentation order + encoding noise.  The classical
controls (Q-learning, Beta-Q, sign-only Hodge) are *not* the candidate: they use
the true distance magnitude (as in the paper) and are expected to converge to
the shared true ranking -- exactly the contrast the constructive account needs.

Every model exposes the same minimal interface used by the synthetic runner:

    run_subject(subject_task, rng) -> ModelRun
        ModelRun.order_strong_to_weak : list[int]   # committed/reconstructed cue order
        ModelRun.choice_prob           : dict[(i,j) i<j] -> float  # P(i beats j)

The runner turns ``choice_prob`` into sampled 10-block TestResponses with a
shared decision lapse, so all models are scored by the same analysis pipeline.
No model reads true_rank, the query answer, or any human choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelRun:
    order_strong_to_weak: list[int]
    choice_prob: dict[tuple[int, int], float] = field(default_factory=dict)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _order_from_scores(scores: np.ndarray) -> list[int]:
    """Strong-to-weak cue order; ties broken stably by cue index."""
    return list(np.argsort(-np.asarray(scores, dtype=float), kind="mergesort"))


def _displayed_magnitude(observation) -> float:
    """Read validated model-visible magnitude, never generator metadata."""

    magnitude = getattr(observation, "magnitude", None)
    if magnitude is None:
        raise ValueError("magnitude-aware model requires displayed magnitude")
    magnitude = float(magnitude)
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("displayed magnitude must be finite and positive")
    return magnitude


# --------------------------------------------------------------------------- #
# Candidate: ConstructiveGlobalRank-v1 (sign only, constructive commitment)   #
# --------------------------------------------------------------------------- #


class ConstructiveGlobalRank:
    """Sign-only constructive global ranking.

    Shared (population, frozen a priori, not fit to humans) hyperparameters:
      sigma_a : spread of the idiosyncratic per-item anchor prior N(0, sigma_a^2)
      eta0    : initial enforcement step
      kappa   : commitment hardening (step decays as eta0 / (1 + kappa * t))
      delta   : required separation margin on a learned sign
      sigma_enc: per-step encoding noise on positions
    Per-virtual-individual randomness (the only idiosyncratic source, drawn from
    the shared population, never from true ranks): the anchor draw, the block
    presentation order, and the encoding-noise stream.
    """

    def __init__(
        self,
        n_items: int,
        *,
        sigma_a: float = 0.30,
        eta0: float = 2.0,
        kappa: float = 0.5,
        delta: float = 0.40,
        sigma_enc: float = 0.01,
        beta: float = 12.0,
    ) -> None:
        self.n_items = n_items
        self.sigma_a = sigma_a
        self.eta0 = eta0
        self.kappa = kappa
        self.delta = delta
        self.sigma_enc = sigma_enc
        self.beta = beta

    def run_subject(self, subject_task, rng) -> ModelRun:
        # idiosyncratic anchor prior: the only per-individual parameter, drawn
        # from the shared population N(0, sigma_a^2).  It is blind to true rank.
        s = rng.normal(0.0, self.sigma_a, self.n_items)
        for t, trial in enumerate(subject_task.support_trials):
            obs = trial.observation
            # sign == +1  -> right cue is higher ; sign == -1 -> left cue is higher
            # generator: sign=+1 -> left is higher ; sign=-1 -> right is higher
            if obs.sign >= 0:
                high_cue, low_cue = obs.left_cue, obs.right_cue
            else:
                high_cue, low_cue = obs.right_cue, obs.left_cue
            eta = self.eta0 / (1.0 + self.kappa * float(t))
            # push toward a uniform constructive spacing (the model does not know
            # the true magnitude, so it imposes a constant margin -- the
            # inductive bias).  Only the underdetermined, non-learned relations
            # remain free for the idiosyncratic anchor to flip.
            gap = s[high_cue] - s[low_cue]
            error = max(0.0, self.delta - gap)
            s[high_cue] += eta * error * 0.5
            s[low_cue]  -= eta * error * 0.5
            if self.sigma_enc > 0.0:
                s += rng.normal(0.0, self.sigma_enc, self.n_items)
        order = _order_from_scores(s)
        choice_prob = {}
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                # P(i beats j) from the committed scalar positions
                choice_prob[(i, j)] = float(_sigmoid(np.array([self.beta * (s[i] - s[j])]))[0])
        return ModelRun(order_strong_to_weak=order, choice_prob=choice_prob)


# --------------------------------------------------------------------------- #
# Control 1: Q-learning with true magnitude (paper Eq. 3-6) -- converges       #
# --------------------------------------------------------------------------- #


class QLearningControl:
    """Classical independent-value Q-learning using the true distance magnitude.

    This is the paper's baseline.  It is allowed to read the magnitude (derived
    from positions) because it is a *control*, not the candidate; the point is
    that magnitude-driven value updating converges to the shared true ranking.
    """

    def __init__(self, n_items: int, *, alpha: float = 0.35, gamma: float = 4.0) -> None:
        self.n_items = n_items
        self.alpha = alpha
        self.gamma = gamma

    def run_subject(self, subject_task, rng) -> ModelRun:
        q = np.zeros(self.n_items, dtype=float)
        for trial in subject_task.support_trials:
            obs = trial.observation
            d = _displayed_magnitude(obs)
            # generator: sign=+1 -> left is higher ; sign=-1 -> right is higher
            if obs.sign >= 0:
                high_cue, low_cue = obs.left_cue, obs.right_cue
            else:
                high_cue, low_cue = obs.right_cue, obs.left_cue
            delta = d - (q[high_cue] - q[low_cue]) / 2.0
            q[high_cue] += self.alpha * delta
            q[low_cue] -= self.alpha * delta
        order = _order_from_scores(q)
        choice_prob = {}
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                choice_prob[(i, j)] = float(_sigmoid(np.array([self.gamma * (q[i] - q[j])]))[0])
        return ModelRun(order_strong_to_weak=order, choice_prob=choice_prob)


# --------------------------------------------------------------------------- #
# Control 2: sign-only global Hodge (Phase-5g style) -- also converges         #
# --------------------------------------------------------------------------- #


class SignOnlyHodgeControl:
    """Deterministic sign-only global least squares (no anchor, no commitment).

    Demonstrates that sign-only information *alone* does not produce
    idiosyncrasy: on the fixed 8-edge graph the global least-squares solution is
    unique up to affine, so every virtual subject recovers the same ranking.
    """

    def __init__(self, n_items: int, *, gamma: float = 4.0) -> None:
        self.n_items = n_items
        self.gamma = gamma

    def run_subject(self, subject_task, rng) -> ModelRun:
        # accumulate sign evidence on cue edges, averaged across blocks
        edge_sum = np.zeros((self.n_items, self.n_items), dtype=float)
        edge_cnt = np.zeros((self.n_items, self.n_items), dtype=int)
        for trial in subject_task.support_trials:
            obs = trial.observation
            # generator: sign=+1 -> left is higher ; sign=-1 -> right is higher
            if obs.sign >= 0:
                high_cue, low_cue = obs.left_cue, obs.right_cue
            else:
                high_cue, low_cue = obs.right_cue, obs.left_cue
            edge_sum[low_cue, high_cue] += 1.0
            edge_cnt[low_cue, high_cue] += 1
        # least squares: minimize sum (s_i - s_j - y_ij)^2 over oriented edges
        # build B (edges x items) and y (+1 for high>low)
        rows, cols, data, y = [], [], [], []
        for i in range(self.n_items):
            for j in range(self.n_items):
                if edge_cnt[i, j] > 0:
                    e = len(y)
                    rows.extend([e, e])
                    cols.extend([j, i])  # s_j - s_i
                    data.extend([1.0, -1.0])
                    y.append(edge_sum[i, j] / edge_cnt[i, j])
        if not y:
            s = np.zeros(self.n_items)
        else:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import lsqr
            B = csr_matrix((data, (rows, cols)), shape=(len(y), self.n_items))
            s = np.asarray(lsqr(B, np.asarray(y, dtype=float))[0], dtype=float)
        order = _order_from_scores(s)
        choice_prob = {}
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                choice_prob[(i, j)] = float(_sigmoid(np.array([self.gamma * (s[i] - s[j])]))[0])
        return ModelRun(order_strong_to_weak=order, choice_prob=choice_prob)


# --------------------------------------------------------------------------- #
# Control 3: Beta-Q with true magnitude (paper Eq. 7-13) -- converges          #
# --------------------------------------------------------------------------- #


class BetaQControl:
    """Distributional value updating with the true magnitude (paper Eq. 7-13)."""

    def __init__(self, n_items: int, *, alpha: float = 0.35) -> None:
        self.n_items = n_items
        self.alpha = alpha  # bias factor toward boundary

    def run_subject(self, subject_task, rng) -> ModelRun:
        u = np.ones(self.n_items, dtype=float)
        l = np.ones(self.n_items, dtype=float)
        for trial in subject_task.support_trials:
            obs = trial.observation
            d = _displayed_magnitude(obs)
            # generator: sign=+1 -> left is higher ; sign=-1 -> right is higher
            if obs.sign >= 0:
                high_cue, low_cue = obs.left_cue, obs.right_cue
            else:
                high_cue, low_cue = obs.right_cue, obs.left_cue
            vh = u[high_cue] / (u[high_cue] + l[high_cue])
            vl = u[low_cue] / (u[low_cue] + l[low_cue])
            dv = vh - vl  # DeltaV(m,n) = V_m - V_n  (m = high)
            # paper Eq 8-11: U_m, L_n take the full factor; L_m, U_n take alpha
            u[high_cue] += d - dv            # Eq 8
            l[high_cue] += self.alpha * (-d + dv)   # Eq 9
            u[low_cue]  += self.alpha * (-d + dv)   # Eq 10
            l[low_cue]  += d - dv            # Eq 11
        means = u / (u + l)
        order = _order_from_scores(means)
        # choice probability by independent beta sampling approximation: use mean
        choice_prob = {}
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                choice_prob[(i, j)] = float(means[i] / (means[i] + means[j] + 1e-9))
        return ModelRun(order_strong_to_weak=order, choice_prob=choice_prob)


# --------------------------------------------------------------------------- #
# Candidate v1.1: order-independent constructive projection (anchor-driven)   #
# --------------------------------------------------------------------------- #


class ConstructiveGlobalRankProjection:
    """Order-independent constructive global ranking.

    The learned signs enter only as inequality constraints (s_high - s_low >=
    margin) in a single global objective, so presentation order does not affect
    the solution.  Idiosyncrasy therefore comes only from the anchor prior, not
    from trial chronology -- addressing the v1 lesion finding that v1's
    divergence was too order-driven.  The committed total order is the argsort of
    the minimizer; incomparable pairs (not pinned by any sign path) are set by
    the idiosyncratic anchor, giving divergence + bimodality while learned pairs
    stay satisfied (distance effect / accuracy come from the constrained part).

    Minimises  sum_edges max(0, margin - (s_high - s_low))^2  +  lam * ||s - a||^2
    by projected gradient descent (deterministic given the anchor draw).
    """

    def __init__(
        self,
        n_items: int,
        *,
        sigma_a: float = 0.30,
        margin: float = 0.40,
        lam: float = 0.10,
        beta: float = 12.0,
        n_iter: int = 400,
        lr: float = 0.2,
    ) -> None:
        self.n_items = n_items
        self.sigma_a = sigma_a
        self.margin = margin
        self.lam = lam
        self.beta = beta
        self.n_iter = n_iter
        self.lr = lr

    def run_subject(self, subject_task, rng) -> ModelRun:
        a = rng.normal(0.0, self.sigma_a, self.n_items)
        edges = []  # (high_cue, low_cue)
        for trial in subject_task.support_trials:
            obs = trial.observation
            # generator: sign=+1 -> left is higher ; sign=-1 -> right is higher
            if obs.sign >= 0:
                high_cue, low_cue = obs.left_cue, obs.right_cue
            else:
                high_cue, low_cue = obs.right_cue, obs.left_cue
            edges.append((high_cue, low_cue))
        s = a.copy()
        for _ in range(self.n_iter):
            grad = 2.0 * self.lam * (s - a)
            for high_cue, low_cue in edges:
                viol = self.margin - (s[high_cue] - s[low_cue])
                if viol > 0.0:
                    grad[high_cue] += -2.0 * viol
                    grad[low_cue] += 2.0 * viol
            s = s - self.lr * grad
        order = _order_from_scores(s)
        choice_prob = {}
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                choice_prob[(i, j)] = float(_sigmoid(np.array([self.beta * (s[i] - s[j])]))[0])
        return ModelRun(order_strong_to_weak=order, choice_prob=choice_prob)

# --------------------------------------------------------------------------- #
# Candidate v2: magnitude-aware constructive ranking (anchor bias on a global #
# scalar fit to the true distances).  Order-independent, closed-form ridge.   #
# --------------------------------------------------------------------------- #


class ConstructiveGlobalRankMagnitude:
    """Magnitude-aware constructive global ranking (CGR-v2).

    Unlike v1/v1.1 (sign-only), v2 uses the displayed distance magnitude
    D=(high-low)/(n-1) -- exactly what the screen shows and what humans use --
    to fit a global scalar, then adds an idiosyncratic per-item anchor bias.
    The anchor acts on a SINGLE global scalar (not independent per-item values),
    so idiosyncratic distortions remain self-consistent (a total order), unlike
    Q-learning + per-item noise which breaks transitivity.  Order-independent
    closed-form ridge solve -> idiosyncrasy comes only from the anchor.

    Minimises  sum_edges (s_high - s_low - D)^2  +  lam * ||s - a||^2
    => (B^T B + lam I) s = B^T y + lam a   (closed form).
    """

    def __init__(self, n_items: int, *, sigma_a: float = 0.35, lam: float = 0.05,
                 beta: float = 12.0) -> None:
        self.n_items = n_items
        self.sigma_a = sigma_a
        self.lam = lam
        self.beta = beta

    def run_subject(self, subject_task, rng) -> ModelRun:
        a = rng.normal(0.0, self.sigma_a, self.n_items)
        rows, cols, data, y = [], [], [], []
        for trial in subject_task.support_trials:
            obs = trial.observation
            d = _displayed_magnitude(obs)
            if obs.sign >= 0:
                high_cue, low_cue = obs.left_cue, obs.right_cue
            else:
                high_cue, low_cue = obs.right_cue, obs.left_cue
            e = len(y)
            rows.extend([e, e]); cols.extend([high_cue, low_cue]); data.extend([1.0, -1.0])
            y.append(d)
        import numpy as np
        E = len(y)
        # B (E x n): row e has +1 at high_cue, -1 at low_cue
        B = np.zeros((E, self.n_items))
        for e in range(E):
            B[e, cols[2*e]] += data[2*e]      # high_cue
            B[e, cols[2*e+1]] += data[2*e+1]  # low_cue
        BtB = B.T @ B + self.lam * np.eye(self.n_items)
        rhs = B.T @ np.asarray(y, dtype=float) + self.lam * a
        s = np.linalg.solve(BtB, rhs)
        order = _order_from_scores(s)
        choice_prob = {}
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                choice_prob[(i, j)] = float(_sigmoid(np.array([self.beta * (s[i] - s[j])]))[0])
        return ModelRun(order_strong_to_weak=order, choice_prob=choice_prob)


# --------------------------------------------------------------------------- #
# Candidate v3: distance-targeted constructive compression.                  #
# --------------------------------------------------------------------------- #


class ConstructiveGlobalRankCompression:
    """Distance-targeted constructive compression (CGR-v3).

    Two stages, both order-independent:
      1. Fit a global scalar from the displayed magnitudes (closed-form LS) ->
         recovers the mostly-true order, so FAR (large-gap) pairs stay correct
         (accuracy / distance effect).
      2. Add an idiosyncratic per-item anchor jitter, then COMMIT to the argsort
         total order.  The jitter is distance-targeted by construction: it is
         large relative to small (near) gaps -> swaps near pairs idiosyncratically
         (bimodality on hard pairs) but small relative to large (far) gaps ->
         leaves easy pairs intact.

    Because the idiosyncrasy is read off a single committed total order (not
    independent per-item noisy values), errors remain self-consistent (zero
    circular triads), unlike Q-learning + per-item noise.  The anchor is the only
    idiosyncratic source (order-independent closed-form fit).
    """

    def __init__(self, n_items: int, *, sigma_a: float = 0.30, beta: float = 12.0,
                 ridge: float = 1e-6) -> None:
        self.n_items = n_items
        self.sigma_a = sigma_a
        self.beta = beta
        self.ridge = ridge

    def run_subject(self, subject_task, rng) -> ModelRun:
        import numpy as np
        a = rng.normal(0.0, self.sigma_a, self.n_items)
        cols, data, y = [], [], []
        for trial in subject_task.support_trials:
            obs = trial.observation
            d = _displayed_magnitude(obs)
            if obs.sign >= 0:
                high_cue, low_cue = obs.left_cue, obs.right_cue
            else:
                high_cue, low_cue = obs.right_cue, obs.left_cue
            cols.extend([high_cue, low_cue]); data.extend([1.0, -1.0]); y.append(d)
        E = len(y)
        B = np.zeros((E, self.n_items))
        for e in range(E):
            B[e, cols[2*e]] += 1.0; B[e, cols[2*e+1]] += -1.0
        s_fit = np.linalg.solve(B.T @ B + self.ridge * np.eye(self.n_items),
                                B.T @ np.asarray(y, dtype=float))
        s = s_fit + a                 # distance-targeted idiosyncratic jitter
        order = _order_from_scores(s)
        choice_prob = {}
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                choice_prob[(i, j)] = float(_sigmoid(np.array([self.beta * (s[i] - s[j])]))[0])
        return ModelRun(order_strong_to_weak=order, choice_prob=choice_prob)


# --------------------------------------------------------------------------- #
# Candidate v3.1: finite relation memory + constrained global commitment.     #
# --------------------------------------------------------------------------- #


class ConstructiveGlobalRankMemoryConstrained:
    """Memory-constrained constructive global ranking (CGR-v3.1).

    CGR-v3 applies one item-level perturbation to every learned and inferred
    relation.  That makes directly experienced pairs too fragile.  V3.1 gives
    direct evidence and constructive inference distinct computational roles:

      1. Repeated presentations of each support relation are compressed into a
         finite-memory trace.  A relation is recalled when at least one of its
         presentations is encoded.
      2. Recalled magnitudes are integrated into a global scalar axis.
      3. An idiosyncratic item prior perturbs that axis.
      4. The final committed order is the prior-preferred *linear extension* of
         the recalled relations.  Thus remembered support signs constrain the
         same global order used for novel inference; they are not a separate
         pair-specific override that could create circular preferences.

    ``encoding_probability`` is per presentation.  With repeated exposure, a
    relation's recall probability is ``1 - (1-p)**n_presentations``.  Sampling
    is performed after grouping trials by relation, in canonical cue-pair
    order, so the model is invariant to presentation order for a fixed RNG.

    The model reads cue identity, sign, and the displayed relative magnitude
    carried by the support observation.  It never reads ``position_pair``,
    ``true_rank``, query
    targets, participant identity, or human choices.
    """

    def __init__(
        self,
        n_items: int,
        *,
        sigma_a: float = 0.30,
        encoding_probability: float = 0.45,
        beta: float = 12.0,
        ridge: float = 1e-6,
    ) -> None:
        if n_items < 2:
            raise ValueError("n_items must be at least two")
        if sigma_a < 0.0:
            raise ValueError("sigma_a must be non-negative")
        if not 0.0 <= encoding_probability <= 1.0:
            raise ValueError("encoding_probability must lie in [0, 1]")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        self.n_items = int(n_items)
        self.sigma_a = float(sigma_a)
        self.encoding_probability = float(encoding_probability)
        self.beta = float(beta)
        self.ridge = float(ridge)

    def _group_support(self, subject_task):
        """Return order-invariant unique relation summaries.

        Each entry is ``(cue_pair, high, low, mean_magnitude, repetitions)``.
        Contradictory signs for the same cue pair are rejected because a single
        recalled ordering constraint cannot represent both directions.
        """

        grouped: dict[tuple[int, int], dict[str, object]] = {}
        for trial in subject_task.support_trials:
            obs = trial.observation
            if obs.sign >= 0:
                high_cue, low_cue = int(obs.left_cue), int(obs.right_cue)
            else:
                high_cue, low_cue = int(obs.right_cue), int(obs.left_cue)
            if high_cue == low_cue:
                raise ValueError("support relation must contain two distinct cues")
            if not (0 <= high_cue < self.n_items and 0 <= low_cue < self.n_items):
                raise ValueError("support cue is outside the model item range")
            key = tuple(sorted((high_cue, low_cue)))
            magnitude = _displayed_magnitude(obs)
            entry = grouped.setdefault(
                key,
                {"high": high_cue, "low": low_cue, "magnitudes": []},
            )
            if entry["high"] != high_cue or entry["low"] != low_cue:
                raise ValueError("contradictory support signs for one cue pair")
            entry["magnitudes"].append(magnitude)

        summaries = []
        for key in sorted(grouped):
            entry = grouped[key]
            magnitudes = entry["magnitudes"]
            summaries.append(
                (
                    key,
                    int(entry["high"]),
                    int(entry["low"]),
                    float(np.mean(magnitudes)),
                    len(magnitudes),
                )
            )
        return summaries

    def _sample_recalled_relations(self, summaries, rng):
        recalled = []
        for _key, high, low, magnitude, repetitions in summaries:
            encoded_count = int(
                rng.binomial(repetitions, self.encoding_probability)
            )
            if encoded_count > 0:
                recalled.append((high, low, magnitude, encoded_count))
        return recalled

    def _fit_recalled_axis(self, recalled) -> np.ndarray:
        if not recalled:
            return np.zeros(self.n_items, dtype=float)
        B = np.zeros((len(recalled), self.n_items), dtype=float)
        y = np.zeros(len(recalled), dtype=float)
        for row, (high, low, magnitude, encoded_count) in enumerate(recalled):
            weight = float(np.sqrt(encoded_count))
            B[row, high] = weight
            B[row, low] = -weight
            y[row] = weight * magnitude
        return np.linalg.solve(
            B.T @ B + self.ridge * np.eye(self.n_items), B.T @ y
        )

    def _preferred_linear_extension(self, scores, recalled) -> list[int]:
        """Priority topological sort: recalled high cues precede low cues."""

        outgoing = [set() for _ in range(self.n_items)]
        indegree = [0] * self.n_items
        for high, low, _magnitude, _count in recalled:
            if low not in outgoing[high]:
                outgoing[high].add(low)
                indegree[low] += 1

        available = {cue for cue, degree in enumerate(indegree) if degree == 0}
        order: list[int] = []
        while available:
            cue = max(available, key=lambda item: (float(scores[item]), -item))
            available.remove(cue)
            order.append(cue)
            for lower in sorted(outgoing[cue]):
                indegree[lower] -= 1
                if indegree[lower] == 0:
                    available.add(lower)
        if len(order) != self.n_items:
            raise ValueError("recalled support relations contain a directed cycle")
        return order

    def run_subject(self, subject_task, rng) -> ModelRun:
        summaries = self._group_support(subject_task)
        recalled = self._sample_recalled_relations(summaries, rng)
        s_fit = self._fit_recalled_axis(recalled)
        anchor = rng.normal(0.0, self.sigma_a, self.n_items)
        proposed_scores = s_fit + anchor
        order = self._preferred_linear_extension(proposed_scores, recalled)

        # The committed order is the common source for learned and novel
        # choices.  Rank distance controls confidence; beta=0 is a soft-readout
        # lesion with chance choice on every pair.
        committed_value = np.empty(self.n_items, dtype=float)
        denominator = float(self.n_items - 1)
        for rank, cue in enumerate(order):
            committed_value[cue] = float(self.n_items - 1 - rank) / denominator
        choice_prob = {}
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                logit = self.beta * (committed_value[i] - committed_value[j])
                choice_prob[(i, j)] = float(_sigmoid(np.array([logit]))[0])
        return ModelRun(order_strong_to_weak=order, choice_prob=choice_prob)


@dataclass
class ConstructiveGlobalRankOnlineState:
    """Inspectable within-episode state for CGR-v3.2."""

    anchor: np.ndarray
    encoding_seed: int
    precision_inverse: np.ndarray
    axis: np.ndarray
    exposure_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    encoded_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    relation_direction: dict[tuple[int, int], tuple[int, int]] = field(
        default_factory=dict
    )
    order_strong_to_weak: list[int] = field(default_factory=list)
    support_steps: int = 0
    order_trajectory: list[tuple[int, ...]] = field(default_factory=list)
    encoded_relation_trajectory: list[int] = field(default_factory=list)
    block_trajectory: list[int] = field(default_factory=list)


class ConstructiveGlobalRankOnlineMemory(
    ConstructiveGlobalRankMemoryConstrained
):
    """Online memory-constrained constructive ranking (CGR-v3.2).

    V3.1 is a batch endpoint model: it groups the complete support set before
    sampling memory and constructing a ranking.  V3.2 implements the experiment
    as an episode process:

      ``initialize_state -> support_step x T -> query_probability x Q``.

    Every encoded support presentation performs a recursive least-squares
    update of the global scalar axis.  The current global order is reconstructed
    after every trial as the anchor-preferred linear extension of all relations
    recalled so far.  The order can therefore be inspected after any prefix and
    revised by later evidence.  Query reads are side-effect free.

    A fixed episode-level encoding seed generates a separate deterministic
    Bernoulli stream for each cue pair and exposure number.  Reordering support
    trials therefore changes the intermediate trajectory but not which repeated
    observations are encoded, preventing RNG assignment from masquerading as a
    cognitive presentation-order effect.

    Shared parameters remain fixed within and across evaluation episodes.  No
    query target, reward, or meta-update is available inside the human-like
    episode.
    """

    def initialize_state(self, rng) -> ConstructiveGlobalRankOnlineState:
        anchor = rng.normal(0.0, self.sigma_a, self.n_items)
        if hasattr(rng, "integers"):
            encoding_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
        else:
            encoding_seed = int(rng.randint(0, 2**31 - 1))
        initial_order = self._preferred_linear_extension(anchor, [])
        return ConstructiveGlobalRankOnlineState(
            anchor=np.asarray(anchor, dtype=float),
            encoding_seed=encoding_seed,
            precision_inverse=np.eye(self.n_items, dtype=float) / self.ridge,
            axis=np.zeros(self.n_items, dtype=float),
            order_strong_to_weak=initial_order,
            order_trajectory=[tuple(initial_order)],
            encoded_relation_trajectory=[0],
            block_trajectory=[-1],
        )

    def _parse_support_trial(self, trial):
        obs = trial.observation
        if obs.sign >= 0:
            high_cue, low_cue = int(obs.left_cue), int(obs.right_cue)
        else:
            high_cue, low_cue = int(obs.right_cue), int(obs.left_cue)
        if high_cue == low_cue:
            raise ValueError("support relation must contain two distinct cues")
        if not (0 <= high_cue < self.n_items and 0 <= low_cue < self.n_items):
            raise ValueError("support cue is outside the model item range")
        magnitude = _displayed_magnitude(obs)
        return high_cue, low_cue, magnitude

    @staticmethod
    def _encoding_uniform(
        encoding_seed: int, key: tuple[int, int], exposure_number: int
    ) -> float:
        seed = np.random.SeedSequence(
            [int(encoding_seed), int(key[0]), int(key[1]), int(exposure_number)]
        )
        return float(np.random.default_rng(seed).random())

    def _recalled_relations(self, state: ConstructiveGlobalRankOnlineState):
        recalled = []
        for key in sorted(state.encoded_counts):
            count = state.encoded_counts[key]
            if count <= 0:
                continue
            high, low = state.relation_direction[key]
            recalled.append((high, low, 0.0, count))
        return recalled

    def support_step(
        self, state: ConstructiveGlobalRankOnlineState, trial
    ) -> ConstructiveGlobalRankOnlineState:
        high, low, magnitude = self._parse_support_trial(trial)
        key = tuple(sorted((high, low)))
        existing = state.relation_direction.get(key)
        if existing is not None and existing != (high, low):
            raise ValueError("contradictory support signs for one cue pair")
        state.relation_direction[key] = (high, low)

        exposure = state.exposure_counts.get(key, 0) + 1
        state.exposure_counts[key] = exposure
        encoded = (
            self._encoding_uniform(state.encoding_seed, key, exposure)
            < self.encoding_probability
        )
        if encoded:
            state.encoded_counts[key] = state.encoded_counts.get(key, 0) + 1
            observation = np.zeros(self.n_items, dtype=float)
            observation[high] = 1.0
            observation[low] = -1.0

            precision_times_observation = state.precision_inverse @ observation
            denominator = 1.0 + float(
                observation @ precision_times_observation
            )
            gain = precision_times_observation / denominator
            prediction_error = magnitude - float(observation @ state.axis)
            state.axis = state.axis + gain * prediction_error
            state.precision_inverse = state.precision_inverse - np.outer(
                gain, observation @ state.precision_inverse
            )
            # Numerical drift from repeated rank-one updates should not create
            # an artificial order effect.
            state.precision_inverse = 0.5 * (
                state.precision_inverse + state.precision_inverse.T
            )

        proposed_scores = state.axis + state.anchor
        state.order_strong_to_weak = self._preferred_linear_extension(
            proposed_scores, self._recalled_relations(state)
        )
        state.support_steps += 1
        state.order_trajectory.append(tuple(state.order_strong_to_weak))
        state.encoded_relation_trajectory.append(len(state.encoded_counts))
        state.block_trajectory.append(int(trial.block_index))
        return state

    def query_probability(
        self,
        state: ConstructiveGlobalRankOnlineState,
        left_cue: int,
        right_cue: int,
    ) -> float:
        left_cue, right_cue = int(left_cue), int(right_cue)
        if left_cue == right_cue:
            raise ValueError("query must contain two distinct cues")
        if not (
            0 <= left_cue < self.n_items and 0 <= right_cue < self.n_items
        ):
            raise ValueError("query cue is outside the model item range")
        rank = {cue: index for index, cue in enumerate(state.order_strong_to_weak)}
        left_value = float(self.n_items - 1 - rank[left_cue]) / float(
            self.n_items - 1
        )
        right_value = float(self.n_items - 1 - rank[right_cue]) / float(
            self.n_items - 1
        )
        return float(
            _sigmoid(np.array([self.beta * (left_value - right_value)]))[0]
        )

    def query_step(self, state: ConstructiveGlobalRankOnlineState, trial) -> float:
        """Return P(left cue is higher) for one read-only experimental query."""

        obs = trial.observation
        return self.query_probability(state, obs.left_cue, obs.right_cue)

    def _model_run_from_state(
        self, state: ConstructiveGlobalRankOnlineState
    ) -> ModelRun:
        choice_prob = {}
        for first in range(self.n_items):
            for second in range(first + 1, self.n_items):
                choice_prob[(first, second)] = self.query_probability(
                    state, first, second
                )
        return ModelRun(
            order_strong_to_weak=list(state.order_strong_to_weak),
            choice_prob=choice_prob,
        )

    def run_subject_with_state(self, subject_task, rng):
        state = self.initialize_state(rng)
        for trial in subject_task.support_trials:
            self.support_step(state, trial)
        return self._model_run_from_state(state), state

    def run_subject(self, subject_task, rng) -> ModelRun:
        run, _state = self.run_subject_with_state(subject_task, rng)
        return run
