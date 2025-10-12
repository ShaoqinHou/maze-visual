
# Determinism & tie-stress tester
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

    print(f"Config: runs={runs}, grid={width}x{height}, wall_pct={wall_pct}, seed_base={seed_base}, ensure_connected={int(ensure_connected)}")

    mA, uA, totA = repro_check("bfs", False, "BFS(unweighted)", count_shortest_paths_unweighted)
    print(f"[A] BFS(unweighted): multi-shortest cases = {mA}/{totA - uA} (unreachable={uA}) -> determinism OK")

    mB, uB, totB = repro_check("dijkstra", False, "Dijkstra(unweighted)", count_shortest_paths_unweighted)
    print(f"[B] Dijkstra(unweighted): multi-shortest cases = {mB}/{totB - uB} (unreachable={uB}) -> determinism OK")

    mC, uC, totC = repro_check("dijkstra", True, "Dijkstra(weighted)", count_shortest_paths_weighted)
    print(f"[C] Dijkstra(weighted): multi-shortest cases = {mC}/{totC - uC} (unreachable={uC}) -> determinism OK")

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
