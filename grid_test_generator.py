
"""
Determinism, optimality, and trace tester for grid generators.

Notes for A* alignment (current design):
- Self-loop scalars carry f-scores (g + h), with a tiny tie-break by node id, and undiscovered uses ~1e6 + h sentinel.
- Edge scalars carry edge weights (constant across steps).
- Final pointer matrix represents a shortest-path tree (admissible/consistent h, non-negative weights, no early-exit).
"""
# Usage: python test_generator.py [runs] [width] [height] [wall_pct] [seed_base] [ensure_connected]
# Defaults: runs=200, width=30, height=30, wall_pct=0.20, seed_base=20251011, ensure_connected=1
## Force connectivity (default): unreachable stays 0
# python test_generator.py 50 40 40 0.98 20251011 1
# If ensure_connected=1 (default), the generator resamples until (s,t) is reachable.
# If ensure_connected=0, it uses the first sample drawn; you will then see unreachable>0
# at high wall percentages.
#
from typing import Optional, Dict, Tuple, List
import sys
from grid_generate_data import sample_grid, shortest_path, edge_index, GraphSample
from grid_algorithms import astar as grid_astar

def count_shortest_paths_unweighted(sample: GraphSample) -> int:
    from collections import deque
    n = sample.n
    s, t = sample.start, sample.target
    adj = sample.adj
    INF = 10**18
    dist = [INF]*n
    ways = [0]*n
    q = deque([s])
    dist[s] = 0
    ways[s] = 1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == INF:
                dist[v] = dist[u] + 1
                ways[v] = ways[u]
                q.append(v)
            elif dist[v] == dist[u] + 1:
                ways[v] += ways[u]
                if ways[v] > 1_000_000_000:
                    ways[v] = 1_000_000_000
    return ways[t]

def dijkstra_dist(sample: GraphSample) -> Tuple[List[float], List[int]]:
    import heapq
    n = sample.n
    s = sample.start
    adj = sample.adj
    wmap = sample.weights
    INF = float('inf')
    dist = [INF]*n
    parent = [-1]*n
    dist[s] = 0.0
    parent[s] = s
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]: continue
        for v in adj[u]:
            w = 1.0 if wmap is None else wmap.get((u,v), 1.0)
            nd = d + w
            if nd < dist[v] or (nd == dist[v] and u < parent[v] if parent[v] != -1 else True):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, parent

def count_shortest_paths_weighted(sample: GraphSample) -> int:
    dist, _ = dijkstra_dist(sample)
    n = sample.n
    s, t = sample.start, sample.target
    if dist[t] == float('inf'):
        return 0
    order = sorted(range(n), key=lambda u: (dist[u], u))
    ways = [0]*n
    ways[s] = 1
    for u in order:
        if ways[u] == 0 or dist[u] == float('inf'):
            continue
        du = dist[u]
        for v in sample.adj[u]:
            w = 1.0 if sample.weights is None else sample.weights.get((u,v), 1.0)
            if abs(du + w - dist[v]) < 1e-12:
                ways[v] += ways[u]
                if ways[v] > 1_000_000_000:
                    ways[v] = 1_000_000_000
    return ways[t]

def run_suite(runs: int, width: int, height: int, wall_pct: float, seed_base: int, ensure_connected: bool) -> None:
    def repro_check(algo: str, weighted: bool, label: str, count_paths_fn):
        multi = 0
        unreachable = 0
        for i in range(runs):
            seed = seed_base + i
            s1 = sample_grid(width, height, wall_pct, connectivity=4, seed=seed, algo=algo, weighted=weighted, ensure_connected=ensure_connected)
            s2 = sample_grid(width, height, wall_pct, connectivity=4, seed=seed, algo=algo, weighted=weighted, ensure_connected=ensure_connected)
            p1 = shortest_path(s1, algo)
            p2 = shortest_path(s2, algo)
            if (p1 is None) or (p2 is None):
                unreachable += 1
                continue
            assert p1 == p2, f"{label}: paths differ for seed {seed}"
            k = count_paths_fn(s1)
            if k > 1:
                multi += 1
        return multi, unreachable, runs

    def astar_path(sample: GraphSample) -> Optional[List[int]]:
        """Reconstruct s->t path from the final A* pointer matrix."""
        node_fts, edge_fts, scalars = grid_astar(sample)
        # edge_fts: [T, n, n, 2] with channel0=pointers
        ptr = edge_fts[-1, :, :, 0]
        n = ptr.shape[0]
        parent = [-1] * n
        for v in range(n):
            parent[v] = int(ptr[v].argmax())
        s, t = sample.start, sample.target
        if parent[t] == -1 or parent[s] == -1:
            return None
        if s == t:
            return [s]
        # detect unreachable: walk until s or loop
        path = [t]
        seen = set([t])
        cur = t
        while cur != s:
            cur = parent[cur]
            if cur in seen or cur < 0:
                return None
            seen.add(cur)
            path.append(cur)
        path.reverse()
        return path

    def path_cost(sample: GraphSample, path: List[int]) -> float:
        if path is None:
            return float("inf")
        if sample.weights is None:
            return float(len(path) - 1)
        return float(sum(sample.weights.get((u, v), 1.0) for u, v in zip(path, path[1:])))

    def repro_check_astar(weighted: bool, label: str, count_paths_fn):
        multi = 0
        unreachable = 0
        for i in range(runs):
            seed = seed_base + i
            s1 = sample_grid(width, height, wall_pct, connectivity=4, seed=seed, algo="dijkstra", weighted=weighted, ensure_connected=ensure_connected)
            s2 = sample_grid(width, height, wall_pct, connectivity=4, seed=seed, algo="dijkstra", weighted=weighted, ensure_connected=ensure_connected)
            # Determinism of samples
            a1 = astar_path(s1)
            a2 = astar_path(s2)
            if (a1 is None) or (a2 is None):
                unreachable += 1
                continue
            assert a1 == a2, f"{label}: A* paths differ for seed {seed}"
            # Optimality vs Dijkstra
            from grid_test_generator import dijkstra_dist as _dd
            dist, _ = _dd(s1)
            target = s1.target
            assert dist[target] != float('inf')
            ca = path_cost(s1, a1)
            assert abs(ca - dist[target]) < 1e-9, f"{label}: A* path not optimal cost for seed {seed}: got {ca}, want {dist[target]}"
            k = count_paths_fn(s1)
            if k > 1:
                multi += 1
        return multi, unreachable, runs

    def repro_check_astar_trace(weighted: bool, label: str):
        import numpy as np

        def heuristic(sample: GraphSample) -> np.ndarray:
            w = sample.width
            tx, ty = (sample.target % w, sample.target // w)
            if sample.weights is None:
                min_w = 1.0
            else:
                present = list(sample.weights.values())
                min_w = float(min(present)) if present else 1.0
            h = np.zeros(sample.n, dtype=float)
            for u in range(sample.n):
                ux, uy = (u % w, u // w)
                h[u] = (abs(ux - tx) + abs(uy - ty)) * min_w
            return h

        def simulate_trace(sample: GraphSample):
            n = sample.n
            # Build adjacency with weights
            adj = np.zeros((n, n), dtype=float)
            if sample.weights is None:
                for (u, v) in sample.edges:
                    adj[u, v] = 1.0
            else:
                for (u, v) in sample.edges:
                    adj[u, v] = float(sample.weights.get((u, v), 1.0))

            in_queue = np.zeros(n, dtype=np.int32)
            in_tree = np.zeros(n, dtype=np.int32)
            pointers = np.eye(n, dtype=np.int32)

            g = np.full(n, np.inf, dtype=float)
            g[sample.start] = 0.0
            h = heuristic(sample)

            node_states = []
            edge_states = []
            g_traj = []

            in_queue[sample.start] = 1
            node_states.append(np.stack((in_queue.copy(), in_tree.copy()), axis=-1))
            edge_states.append(np.stack((pointers.copy(), np.eye(n, dtype=np.int32)), axis=-1))
            g_traj.append(g.copy())

            for _ in range(1, n):
                mask = (in_queue == 1)
                if mask.sum() == 0:
                    node_states.append(np.stack((in_queue.copy(), in_tree.copy()), axis=-1))
                    edge_states.append(np.stack((pointers.copy(), np.eye(n, dtype=np.int32)), axis=-1))
                    g_traj.append(g.copy())
                    continue
                f = g + h
                f_masked = np.where(mask, f, 1e18)
                node = int(np.argmin(f_masked))
                in_tree[node] = 1
                in_queue[node] = 0
                outs = np.nonzero(adj[node] > 0.0)[0]
                for v in outs:
                    if in_tree[v] == 1:
                        continue
                    w = adj[node, v] if adj[node, v] > 0.0 else 1.0
                    nd = g[node] + w
                    cur_parent = int(np.argmax(pointers[v]))
                    better = (nd < g[v]) or (nd == g[v] and node < cur_parent)
                    if (in_queue[v] == 0) or better:
                        pointers[v] = 0
                        pointers[v, node] = 1
                        g[v] = nd
                        in_queue[v] = 1
                node_states.append(np.stack((in_queue.copy(), in_tree.copy()), axis=-1))
                edge_states.append(np.stack((pointers.copy(), np.eye(n, dtype=np.int32)), axis=-1))
                g_traj.append(g.copy())
            return np.array(node_states), np.array(edge_states), np.array(g_traj)

        for i in range(runs):
            seed = seed_base + i
            s1 = sample_grid(width, height, wall_pct, connectivity=4, seed=seed, algo="dijkstra", weighted=weighted, ensure_connected=ensure_connected)
            s2 = sample_grid(width, height, wall_pct, connectivity=4, seed=seed, algo="dijkstra", weighted=weighted, ensure_connected=ensure_connected)

            n = s1.n
            n1_nodes, n1_edges, s1_scal = grid_astar(s1)
            n2_nodes, n2_edges, s2_scal = grid_astar(s2)

            # Exact determinism of full traces from generator
            assert np.array_equal(n1_nodes, n2_nodes), f"{label}: node state trace differs for seed {seed}"
            assert np.array_equal(n1_edges, n2_edges), f"{label}: edge state trace differs for seed {seed}"

            # Cross-check against independently simulated A* trace
            sim_nodes, sim_edges, g_traj = simulate_trace(s1)
            assert np.array_equal(n1_nodes, sim_nodes), f"{label}: node trace deviates from simulated A* for seed {seed}"
            assert np.array_equal(n1_edges, sim_edges), f"{label}: edge/pointer trace deviates from simulated A* for seed {seed}"

            # Scalar sanity checks: map flat scalars [E] back to [n,n] via ei
            n = s1.n
            # Build adjacency and ei consistent with astar/grid_data_loader
            adj = np.zeros((n, n), dtype=float)
            if s1.weights is None:
                for (u, v) in s1.edges:
                    adj[u, v] = 1.0
            else:
                for (u, v) in s1.edges:
                    adj[u, v] = float(s1.weights.get((u, v), 1.0))
            ei = np.stack(np.nonzero(adj + np.eye(n, dtype=adj.dtype)))

            # 1) Edge scalars equal weights (constant across steps)
            S0_flat = s1_scal[0, :, 0]
            M = np.zeros_like(adj)
            M[ei[0], ei[1]] = S0_flat
            for (u, v) in s1.edges:
                expected = 1.0 if s1.weights is None else float(s1.weights.get((u, v), 1.0))
                assert abs(M[u, v] - expected) < 1e-9, f"{label}: edge scalar mismatch at ({u},{v}) seed {seed}: {M[u,v]} vs {expected}"
            # 2) Self-loop scalars equal f=g+h (+ tiny tie-break) with sentinel for inf
            T = n1_nodes.shape[0]
            for t in range(T):
                S_flat = s1_scal[t, :, 0]
                Mt = np.zeros_like(adj)
                Mt[ei[0], ei[1]] = S_flat
                diag = np.diag(Mt)
                g = g_traj[t].copy()
                h = heuristic(s1)
                f = g + h
                mask = ~np.isfinite(g)
                if mask.any():
                    f[mask] = h[mask] + 1e6
                # incorporate tiny deterministic tie-break used in generator
                n = s1.n
                if n > 1:
                    eps = 1e-6
                    tie = (np.arange(n, dtype=float) / (n - 1)) * eps
                    f = f + tie
                assert np.allclose(diag, f, atol=1e-9), f"{label}: f-scalar (with tie-break) mismatch at step {t} seed {seed}"

    print(f"Config: runs={runs}, grid={width}x{height}, wall_pct={wall_pct}, seed_base={seed_base}, ensure_connected={int(ensure_connected)}")

    mA, uA, totA = repro_check("bfs", False, "BFS(unweighted)", count_shortest_paths_unweighted)
    print(f"[A] BFS(unweighted): multi-shortest cases = {mA}/{totA - uA} (unreachable={uA}) -> determinism OK")

    mB, uB, totB = repro_check("dijkstra", False, "Dijkstra(unweighted)", count_shortest_paths_unweighted)
    print(f"[B] Dijkstra(unweighted): multi-shortest cases = {mB}/{totB - uB} (unreachable={uB}) -> determinism OK")

    mC, uC, totC = repro_check("dijkstra", True, "Dijkstra(weighted)", count_shortest_paths_weighted)
    print(f"[C] Dijkstra(weighted): multi-shortest cases = {mC}/{totC - uC} (unreachable={uC}) -> determinism OK")

    mD, uD, totD = repro_check_astar(False, "A*(unweighted)", count_shortest_paths_unweighted)
    print(f"[D] A*(unweighted): multi-shortest cases = {mD}/{totD - uD} (unreachable={uD}) -> deterministic + optimal vs Dijkstra")

    mE, uE, totE = repro_check_astar(True, "A*(weighted)", count_shortest_paths_weighted)
    print(f"[E] A*(weighted): multi-shortest cases = {mE}/{totE - uE} (unreachable={uE}) -> deterministic + optimal vs Dijkstra")

    # Deep trace-level verification
    repro_check_astar_trace(False, "A*(unweighted) trace")
    print("[F] A*(unweighted) trace: deterministic + canonical expansion order")
    repro_check_astar_trace(True, "A*(weighted) trace")
    print("[G] A*(weighted) trace: deterministic + canonical expansion order")

def main():
    runs = 200
    width, height = 30, 30
    wall_pct = 0.20
    seed_base = 20251011
    ensure_connected = True
    if len(sys.argv) >= 2:
        runs = int(sys.argv[1])
    if len(sys.argv) >= 4:
        width = int(sys.argv[2]); height = int(sys.argv[3])
    if len(sys.argv) >= 5:
        wall_pct = float(sys.argv[4])
    if len(sys.argv) >= 6:
        seed_base = int(sys.argv[5])
    if len(sys.argv) >= 7:
        ensure_connected = bool(int(sys.argv[6]))
    run_suite(runs, width, height, wall_pct, seed_base, ensure_connected)

if __name__ == "__main__":
    main()
