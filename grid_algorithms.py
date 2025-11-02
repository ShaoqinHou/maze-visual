from __future__ import annotations

import numpy as np

from grid_generate_data import _xy, GraphSample
from generate_data import push_states


def astar(sample: GraphSample):
    """
    A* trace on a grid sample, outputting arrays compatible with repo generators.

    Returns:
      node_states: [T, n, 2] with (in_queue, in_tree)
      edge_states: [T, n, n, 2] with (pointers, self_loops)
      scalars:     [T, n, n, 1] edge weights, self-loops carry node g-scores
    """
    n = sample.n
    width, height = sample.width, sample.height
    start, target = sample.start, sample.target

    # Build adjacency matrix with weights (1.0 for unweighted)
    adj = np.zeros((n, n), dtype=float)
    if sample.weights is None:
        for (u, v) in sample.edges:
            adj[u, v] = 1.0
    else:
        for (u, v) in sample.edges:
            adj[u, v] = float(sample.weights.get((u, v), 1.0))

    node_states = []
    edge_states = []
    scalars = []

    in_queue = np.zeros(n, dtype=np.int32)
    in_tree = np.zeros(n, dtype=np.int32)

    pointers = np.eye(n, dtype=np.int32)
    self_loops = np.eye(n, dtype=np.int32)

    # g-scores (distance from start)
    g = np.full(n, np.inf, dtype=float)
    g[start] = 0.0

    # heuristic: Manhattan distance to target, scaled by min edge weight
    tx, ty = _xy(target, width)
    if sample.weights is None:
        min_w = 1.0
    else:
        # consider only present edges
        present = [w for (e, w) in sample.weights.items()]
        min_w = float(min(present)) if present else 1.0

    h = np.zeros(n, dtype=float)
    for u in range(n):
        ux, uy = _xy(u, width)
        h[u] = (abs(ux - tx) + abs(uy - ty)) * min_w

    # Build edge index including self-loops to match repository convention
    ei = np.stack(np.nonzero(adj + np.eye(n, dtype=adj.dtype)))

    def compute_current_scalars(g_scores):
        # Per-edge scalars: edge weights; self-loops carry f-scores (g + h)
        # with a tiny deterministic tie-break by node id to stabilize argmin.
        s = adj[ei[0], ei[1]].copy()
        f_scores = g_scores + h
        # avoid inf in supervision by using a large finite sentinel for undiscovered nodes
        mask = ~np.isfinite(g_scores)
        if mask.any():
            f_scores = f_scores.copy()
            f_scores[mask] = h[mask] + 1e6
        # add tiny tie-break proportional to node id (stable, does not change optimality)
        # tie-break is per node (match diagonal/self-loops), not per edge
        n_nodes = int(g_scores.shape[0])
        if n_nodes > 1:
            eps = 1e-4
            tie = (np.arange(n_nodes, dtype=float) / (n_nodes - 1)) * eps
            f_scores = f_scores + tie
        s[ei[0] == ei[1]] = f_scores
        return s

    in_queue[start] = 1

    push_states(
        node_states,
        edge_states,
        scalars,
        (in_queue, in_tree),
        (pointers, self_loops),
        (compute_current_scalars(g),),
    )

    # Expand up to n-1 steps similar to Dijkstra; selection uses g+h
    for _ in range(1, n):
        # select next from open by minimal f=g+h; mask out not-in-queue
        mask = (in_queue == 1).astype(float)
        if mask.sum() == 0:
            # no more nodes to expand; still push last state to keep length n
            push_states(
                node_states,
                edge_states,
                scalars,
                (in_queue, in_tree),
                (pointers, self_loops),
                (compute_current_scalars(g),),
            )
            continue

        # selection by f with the same tiny deterministic tie-break by node id
        f = g + h + (1.0 - mask) * 1e9
        if n > 1:
            eps = 1e-4
            tie = (np.arange(n, dtype=float) / (n - 1)) * eps
            f = f + tie
        node = int(np.argmin(f))

        # settle node
        in_tree[node] = 1
        in_queue[node] = 0

        # relax neighbors
        outs = np.nonzero(adj[node] > 0.0)[0]
        for v in outs:
            if in_tree[v] == 1:
                continue
            w = adj[node, v] if adj[node, v] > 0.0 else 1.0
            nd = g[node] + w
            better = (nd < g[v]) or (nd == g[v] and node < np.argmax(pointers[v]))
            if (in_queue[v] == 0) or better:
                pointers[v] = 0
                pointers[v, node] = 1
                g[v] = nd
                in_queue[v] = 1

        push_states(
            node_states,
            edge_states,
            scalars,
            (in_queue, in_tree),
            (pointers, self_loops),
            (compute_current_scalars(g),),
        )

    return np.array(node_states), np.array(edge_states), np.array(scalars)
