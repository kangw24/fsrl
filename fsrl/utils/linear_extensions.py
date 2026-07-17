"""Support-partial-order linear extensions for rank-codebook initialization."""

import random
from typing import List, Sequence, Tuple


def _random_topological_sort(
    graph, rng: random.Random
) -> List[int]:
    """Random priority topological sort for a networkx DiGraph."""
    import heapq

    nodes = list(graph.nodes())
    priorities = {node: rng.random() for node in nodes}
    in_degree = {node: graph.in_degree(node) for node in nodes}
    heap = [(priorities[node], node) for node in nodes if in_degree[node] == 0]
    heapq.heapify(heap)

    result = []
    while heap:
        _, node = heapq.heappop(heap)
        result.append(node)
        for succ in graph.successors(node):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                heapq.heappush(heap, (priorities[succ], succ))

    if len(result) != len(nodes):
        raise ValueError("Graph contains a cycle; cannot produce topological sort.")
    return result


def _backtrack_extensions(
    nbcues: int,
    support_pairs: Sequence[Tuple[int, int]],
    max_extensions: int,
    rng: random.Random,
) -> List[List[int]]:
    """Fallback DFS enumerator / sampler of linear extensions."""
    adj = [[] for _ in range(nbcues)]
    indeg = [0] * nbcues
    for i, j in support_pairs:
        adj[i].append(j)
        indeg[j] += 1

    result: List[List[int]] = []
    used = [False] * nbcues
    path: List[int] = []

    def dfs() -> None:
        if len(result) >= max_extensions:
            return
        if len(path) == nbcues:
            result.append(path.copy())
            return
        available = [i for i in range(nbcues) if not used[i] and indeg[i] == 0]
        rng.shuffle(available)
        for node in available:
            used[node] = True
            for v in adj[node]:
                indeg[v] -= 1
            path.append(node)
            dfs()
            path.pop()
            for v in adj[node]:
                indeg[v] += 1
            used[node] = False
            if len(result) >= max_extensions:
                return

    dfs()
    return result


def generate_linear_extensions(
    nbcues: int,
    support_pairs: Sequence[Tuple[int, int]],
    max_extensions: int = 128,
    seed: int = None,
) -> List[List[int]]:
    """
    Generate global rankings (linear extensions) that respect ``support_pairs``.

    Each returned list orders cues from strongest to weakest.  For example,
    ``support_pairs=[(0, 5)]`` means cue 0 must appear before cue 5.

    Parameters
    ----------
    nbcues : int
        Number of cues (nodes).
    support_pairs : sequence of (int, int)
        Pairs ``(strong, weak)`` defining the partial order.
    max_extensions : int
        Maximum number of linear extensions to return.  If the total count
        exceeds this, a random sample is returned.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    List[List[int]]
        Linear extensions, each a permutation of ``range(nbcues)``.
    """
    rng = random.Random(seed)

    # Validate that all node indices are in range.
    for i, j in support_pairs:
        if not (0 <= i < nbcues and 0 <= j < nbcues):
            raise ValueError(f"Support pair ({i}, {j}) out of range [0, {nbcues}).")

    # Prefer networkx random-priority sampling when available.
    try:
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(range(nbcues))
        graph.add_edges_from(support_pairs)

        extensions_set = set()
        max_attempts = max_extensions * 50
        attempts = 0
        while len(extensions_set) < max_extensions and attempts < max_attempts:
            attempts += 1
            ext = tuple(_random_topological_sort(graph, rng))
            extensions_set.add(ext)

        extensions = [list(ext) for ext in extensions_set]
    except Exception:
        extensions = _backtrack_extensions(nbcues, support_pairs, max_extensions, rng)

    # Identity is only a valid extension when it satisfies the supplied edges.
    # Never inject a ground-truth ordering that contradicts the observations.
    identity = list(range(nbcues))
    identity_position = {item: pos for pos, item in enumerate(identity)}
    identity_valid = all(
        identity_position[strong] < identity_position[weak]
        for strong, weak in support_pairs
    )
    if identity_valid:
        if identity in extensions:
            extensions.remove(identity)
        extensions.insert(0, identity)

    # Cap at the requested number.
    return extensions[:max_extensions]
