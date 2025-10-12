
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import random

@dataclass
class GraphSample:
    n: int
    width: int
    height: int
    adj: List[List[int]]            # neighbors per node (sorted, deduped)
    edges: List[Tuple[int, int]]    # directed edges (u,v) for undirected graphs mirrored
    weights: Optional[Dict[Tuple[int, int], float]]  # None for unweighted
    start: int
    target: int
    passable: List[bool]            # len=n, True if cell is not a wall

def _node_id(x: int, y: int, w: int) -> int:
    return y * w + x

def _xy(node: int, w: int) -> Tuple[int, int]:
    return (node % w, node // w)

def _neighbors_4(x: int, y: int, w: int, h: int) -> List[Tuple[int,int]]:
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield (nx, ny)

def _neighbors_8(x: int, y: int, w: int, h: int) -> List[Tuple[int,int]]:
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                yield (nx, ny)

def set_seed(seed: int) -> None:
    random.seed(seed)

def generate_grid_graph(
    width: int,
    height: int,
    wall_pct: float,
    connectivity: int = 4,
    seed: int = 1337,
    ensure_connected: bool = True,
) -> Tuple[List[List[int]], List[Tuple[int,int]], List[bool]]:
    """
    Returns (adj, edges, passable).
    - adj: sorted neighbor lists for each node id 0..n-1 (only passable nodes have neighbors)
    - edges: list of directed edges (u,v), mirrored for undirected
    - passable: list of bools per node (True means not a wall)
    """
    assert 0 <= wall_pct < 1.0, "wall_pct in [0,1)"
    set_seed(seed)
    n = width * height
    # Sample passable cells
    # Ensure at least 2 passable cells
    attempts = 0
    while True:
        attempts += 1
        passable = [random.random() >= wall_pct for _ in range(n)]
        if sum(passable) >= 2:
            break
        if attempts > 1000:
            raise RuntimeError("Failed to sample passable cells with >=2 open tiles.")
    # Build adjacency (only between passable cells)
    adj = [[] for _ in range(n)]
    edges_set: Set[Tuple[int,int]] = set()
    neigh_fn = _neighbors_4 if connectivity == 4 else _neighbors_8
    for y in range(height):
        for x in range(width):
            u = _node_id(x, y, width)
            if not passable[u]:
                continue
            for nx, ny in neigh_fn(x, y, width, height):
                v = _node_id(nx, ny, width)
                if passable[v]:
                    # undirected graph -> mirror edges (store directed for convenience)
                    edges_set.add((u, v))
                    edges_set.add((v, u))
    # Canonicalize neighbor order: sorted, deduped
    for (u, v) in edges_set:
        adj[u].append(v)
    for u in range(n):
        if adj[u]:
            adj[u] = sorted(set(adj[u]))
    edges = sorted(edges_set)  # global deterministic order if needed downstream
    return adj, edges, passable

def _pick_start_target(passable: List[bool], seed: int) -> Tuple[int,int]:
    set_seed(seed)
    candidates = [i for i, ok in enumerate(passable) if ok]
    if len(candidates) < 2:
        raise ValueError("Need at least two passable nodes to pick start/target.")
    # Deterministic choice by shuffling with fixed seed then picking first two
    random.shuffle(candidates)
    return candidates[0], candidates[1]

def _bfs_path(adj: List[List[int]], start: int, target: int) -> Optional[List[int]]:
    from collections import deque
    n = len(adj)
    parent = [-1]*n
    q = deque()
    q.append(start)
    parent[start] = start
    while q:
        u = q.popleft()
        if u == target:
            break
        # neighbors are already sorted -> deterministic parent discovery
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                q.append(v)
    if parent[target] == -1:
        return None
    # reconstruct
    path = [target]
    cur = target
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path

def _dijkstra_path(adj: List[List[int]], weights: Optional[Dict[Tuple[int,int], float]], start: int, target: int) -> Optional[List[int]]:
    import heapq
    n = len(adj)
    INF = float('inf')
    dist = [INF]*n
    parent = [-1]*n
    dist[start] = 0.0
    parent[start] = start
    # Use (dist, node) so ties break by node id deterministically
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        if u == target:
            break
        for v in adj[u]:  # already sorted
            w = 1.0 if weights is None else weights.get((u,v), 1.0)
            nd = d + w
            if nd < dist[v] or (nd == dist[v] and u < parent[v] if parent[v] != -1 else True):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    if parent[target] == -1:
        return None
    # reconstruct
    path = [target]
    cur = target
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path

def _ensure_reachable(adj: List[List[int]], start: int, target: int) -> bool:
    return _bfs_path(adj, start, target) is not None

def _build_weights(edges: List[Tuple[int,int]], weighted: bool, seed: int, width: int) -> Optional[Dict[Tuple[int,int], float]]:
    if not weighted:
        return None
    set_seed(seed)
    # Integers 1..9 for stability; mirror edges get identical weights
    wmap: Dict[Tuple[int,int], float] = {}
    # To ensure undirected weight symmetry, assign only once per undirected pair
    for (u, v) in edges:
        if (v, u) in wmap:
            wmap[(u,v)] = wmap[(v,u)]
        else:
            w = float(random.randint(1, 9))
            wmap[(u,v)] = w
    return wmap

def sample_grid(
    width: int,
    height: int,
    wall_pct: float,
    connectivity: int = 4,
    seed: int = 1337,
    start_target_seed: Optional[int] = None,
    ensure_connected: bool = True,
    algo: str = "bfs",          # 'bfs' or 'dijkstra' (only used for testing convenience)
    weighted: bool = False,     # True -> weighted edges for Dijkstra
) -> GraphSample:
    """
    Minimal, deterministic grid-graph generator with canonical neighbor order.
    """
    adj, edges, passable = generate_grid_graph(width, height, wall_pct, connectivity, seed, ensure_connected=False)
    # pick s,t
    sts = seed if start_target_seed is None else start_target_seed
    start, target = _pick_start_target(passable, sts)
    if ensure_connected:
        # If not connected, resample walls with incremented seed deterministically until connected
        attempt = 0
        base_seed = seed
        while not _ensure_reachable(adj, start, target):
            attempt += 1
            if attempt > 512:
                raise RuntimeError("Failed to create a connected (s,t) pair after many attempts.")
            seed = base_seed + attempt
            adj, edges, passable = generate_grid_graph(width, height, wall_pct, connectivity, seed, ensure_connected=False)
            start, target = _pick_start_target(passable, sts + attempt)
    # weights
    weights = _build_weights(edges, weighted=weighted, seed=seed, width=width)
    n = width * height
    return GraphSample(n=n, width=width, height=height, adj=adj, edges=edges, weights=weights, start=start, target=target, passable=passable)

def shortest_path(sample: GraphSample, algo: str = "bfs") -> Optional[List[int]]:
    if algo == "bfs":
        return _bfs_path(sample.adj, sample.start, sample.target)
    elif algo == "dijkstra":
        return _dijkstra_path(sample.adj, sample.weights, sample.start, sample.target)
    else:
        raise ValueError("algo must be 'bfs' or 'dijkstra'")

def edge_index(sample: GraphSample) -> List[Tuple[int,int]]:
    """
    Convenience: returns a sorted directed edge list (u,v). Already stored as sample.edges.
    """
    return sample.edges[:]
