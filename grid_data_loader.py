from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import tqdm

from configs.grid_base_config import GridConfig
from grid_generate_data import sample_grid
from grid_algorithms import astar as grid_astar
from generate_data import ProblemInstance, ALGORITHMS


def _sample_to_problem_instance(sample, weighted: bool) -> ProblemInstance:
    import numpy as np

    n = sample.n
    # Build adjacency matrix (float) with optional weights
    adj = np.zeros((n, n), dtype=float)
    if weighted and sample.weights is None:
        raise ValueError("Weighted requested but sample.weights is None")
    for (u, v) in sample.edges:
        adj[u, v] = (sample.weights.get((u, v), 1.0) if weighted else 1.0)

    # ProblemInstance expects: adj (np array), start (int), weighted (bool), randomness (None)
    return ProblemInstance(adj=adj, start=sample.start, weighted=weighted, randomness=None)


def create_grid_dataloader(config: GridConfig, split: str, seed: int, device) -> DataLoader:
    np.random.seed(seed)

    datapoints = []
    algo = config.algorithm
    weighted = bool(config.grid_weighted)

    for i in tqdm.tqdm(range(config.num_samples[split]), desc=f"Generate grid samples for {split}"):
        # vary the base seed per sample deterministically
        base_seed = seed + i
        grid = sample_grid(
            width=config.width,
            height=config.height,
            wall_pct=config.wall_pct,
            connectivity=config.connectivity,
            seed=base_seed,
            ensure_connected=config.ensure_connected,
            weighted=weighted,
            algo=algo,
        )

        if algo == "astar":
            # Use grid-specific A* trace
            node_fts, edge_fts, scalars = grid_astar(grid)
            # Build adjacency for edge_index (include self-loops)
            n = grid.n
            adj = np.zeros((n, n), dtype=float)
            if weighted and grid.weights is not None:
                for (u, v) in grid.edges:
                    adj[u, v] = grid.weights.get((u, v), 1.0)
            else:
                for (u, v) in grid.edges:
                    adj[u, v] = 1.0
            ei = np.stack(np.nonzero(adj + np.eye(n, dtype=adj.dtype)))
            edge_index = torch.tensor(ei).contiguous()
        else:
            instance = _sample_to_problem_instance(grid, weighted=weighted)
            node_fts, edge_fts, scalars = ALGORITHMS[algo](instance)
            edge_index = torch.tensor(instance.edge_index).contiguous()

        node_fts = torch.transpose(torch.tensor(node_fts), 0, 1)
        edge_fts = torch.transpose(
            torch.tensor(edge_fts)[:, edge_index[0], edge_index[1]], 0, 1
        )
        scalars = torch.transpose(torch.tensor(scalars), 0, 1)

        output_fts = edge_fts if config.output_type == "pointer" else node_fts
        y = output_fts[:, -1, config.output_idx].clone().detach()

        datapoints.append(
            Data(
                node_fts=node_fts,
                edge_fts=edge_fts,
                scalars=scalars,
                edge_index=edge_index,
                y=y,
            ).to(device)
        )

    return DataLoader(datapoints, batch_size=config.batch_size, shuffle=True)
