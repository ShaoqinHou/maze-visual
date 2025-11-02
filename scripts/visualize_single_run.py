from __future__ import annotations

import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

import utils
import models
import generate_data as gd
from configs.grid_base_config import read_config, GridConfig
from grid_generate_data import sample_grid, _xy
from grid_algorithms import astar as grid_astar


def build_single_graph_data(cfg: GridConfig, seed: int, width: int, height: int, wall_pct: float) -> Tuple[gd.Data, dict]:
    # Sample a single grid and produce teacher features like in grid_data_loader
    grid = sample_grid(width=width, height=height, wall_pct=wall_pct,
                       connectivity=cfg.connectivity, seed=seed, ensure_connected=cfg.ensure_connected,
                       weighted=bool(cfg.grid_weighted), algo=cfg.algorithm)

    node_fts, edge_fts, scalars = grid_astar(grid)

    n = grid.n
    # Build adjacency for edge_index (include self-loops), consistent with dataloader
    adj = np.zeros((n, n), dtype=float)
    if cfg.grid_weighted and grid.weights is not None:
        for (u, v) in grid.edges:
            adj[u, v] = grid.weights.get((u, v), 1.0)
    else:
        for (u, v) in grid.edges:
            adj[u, v] = 1.0
    ei = np.stack(np.nonzero(adj + np.eye(n, dtype=adj.dtype)))

    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor(ei).contiguous()
    node_t = torch.transpose(torch.tensor(node_fts), 0, 1)
    edge_t = torch.transpose(torch.tensor(edge_fts)[:, edge_index[0], edge_index[1]], 0, 1)
    scal_t = torch.transpose(torch.tensor(scalars), 0, 1)
    # Target y: final pointer channel
    output_fts = edge_t
    y = output_fts[:, -1, cfg.output_idx].clone().detach()

    data = Data(node_fts=node_t, edge_fts=edge_t, scalars=scal_t, edge_index=edge_index, y=y)
    meta = {
        "grid": grid,
        "node_states": node_fts,  # [T, n, 2]
        "edge_states": edge_fts,  # [T, n, n, 2]
        "scalars": scalars,       # [T, E, 1] flattened over ei
        "edge_index_np": ei,
    }
    return data, meta


def reconstruct_f_diag(scalars_t: np.ndarray, ei: np.ndarray, n: int) -> np.ndarray:
    # scalars_t: [E] at a given step; map to [n,n] and read diagonal (self-loops)
    M = np.zeros((n, n), dtype=float)
    M[ei[0], ei[1]] = scalars_t
    return np.diag(M)


def visualize_steps(meta: dict, out_path: str, steps: List[int], eps: float = 1e-4):
    grid = meta["grid"]
    node_states = meta["node_states"]  # [T, n, 2]: (in_queue, in_tree)
    T, n, _ = node_states.shape
    width, height = grid.width, grid.height

    # Compute teacher final path from pointers at last step
    edge_states = meta["edge_states"]  # [T, n, n, 2], channel0=pointers
    ptr_last = edge_states[-1, :, :, 0]
    parent = np.argmax(ptr_last, axis=1)
    # reconstruct path s->t
    s, t = grid.start, grid.target
    path = [t]
    seen = {t}
    cur = t
    ok = True
    while cur != s:
        cur = int(parent[cur])
        if cur in seen or cur < 0:
            ok = False
            break
        seen.add(cur)
        path.append(cur)
    if ok:
        path = list(reversed(path))
    else:
        path = []

    # Prepare base obstacles map
    base = np.zeros((height, width, 3), dtype=float)
    # obstacles: black
    for node_id, passable in enumerate(grid.passable):
        x, y = _xy(node_id, width)
        if not passable:
            base[y, x] = (0, 0, 0)
        else:
            base[y, x] = (1, 1, 1)  # white

    fig, axes = plt.subplots(1, len(steps), figsize=(3 * len(steps), 3))
    if len(steps) == 1:
        axes = [axes]

    for ax, t_idx in zip(axes, steps):
        t_idx = max(0, min(T - 1, t_idx))
        img = base.copy()
        in_queue = node_states[t_idx, :, 0]
        in_tree = node_states[t_idx, :, 1]
        # overlay in_tree (blue), then in_queue (yellow)
        for node_id in range(n):
            x, y = _xy(node_id, width)
            if not grid.passable[node_id]:
                continue
            if in_tree[node_id] > 0.5:
                img[y, x] = (0.4, 0.6, 0.95)
            elif in_queue[node_id] > 0.5:
                img[y, x] = (0.95, 0.9, 0.4)
        # start/target
        sx, sy = _xy(grid.start, width)
        tx, ty = _xy(grid.target, width)
        img[sy, sx] = (0.3, 0.9, 0.3)
        img[ty, tx] = (0.9, 0.3, 0.3)
        # final path overlay (blue line)
        if path:
            for u, v in zip(path, path[1:]):
                x1, y1 = _xy(u, width)
                x2, y2 = _xy(v, width)
                # draw as thicker pixels
                img[y1, x1] = (0.1, 0.2, 0.8)
                img[y2, x2] = (0.1, 0.2, 0.8)
        ax.imshow(img)
        ax.set_title(f"step {t_idx}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"Saved steps visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize a single DNAR-A* run vs teacher")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=False, default=None)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--wall_pct", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=20251016)
    parser.add_argument("--out_steps", type=str, default="viz_steps.png")
    parser.add_argument("--out_f", type=str, default="viz_f_heatmap.png")
    parser.add_argument("--out_compare", type=str, default="viz_compare.png")
    parser.add_argument("--out_mismatch", type=str, default="viz_mismatch.png")
    parser.add_argument("--out_overlay", type=str, default="viz_paths_overlay.png")

    args = parser.parse_args()

    cfg: GridConfig = read_config(args.config_path)
    cfg.grid_weighted = False
    cfg.connectivity = 4
    cfg.ensure_connected = True
    cfg.output_type = "pointer"

    # Build data for a single graph
    data, meta = build_single_graph_data(cfg, seed=args.seed, width=args.width, height=args.height, wall_pct=args.wall_pct)

    # Teacher f heatmap at last step
    T, n, _ = meta["node_states"].shape
    f_last = reconstruct_f_diag(meta["scalars"][-1, :, 0], meta["edge_index_np"], n)
    # Draw f as heatmap (reshape to grid)
    width, height = meta["grid"].width, meta["grid"].height
    f_img = np.zeros((height, width), dtype=float)
    for node_id, f_val in enumerate(f_last):
        x, y = _xy(node_id, width)
        f_img[y, x] = f_val if np.isfinite(f_val) and f_val < 1e5 else np.nan
    plt.figure(figsize=(4, 4))
    im = plt.imshow(f_img, cmap="viridis")
    plt.colorbar(im, shrink=0.8)
    plt.title("Teacher f-score (last step)")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.savefig(args.out_f, dpi=180)
    print(f"Saved f heatmap to {args.out_f}")

    # Steps mosaic: first, mid, last
    steps = [0, max(0, T//4), max(0, T//2), max(0, 3*T//4), T-1]
    visualize_steps(meta, args.out_steps, steps)

    # If checkpoint provided, compare final pointers (per-node) vs teacher
    if args.checkpoint_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Align SPEC for A* before constructing model
        if cfg.algorithm == "astar" and "astar" not in gd.SPEC:
            gd.SPEC["astar"] = gd.SPEC["dijkstra"]
        model = models.Dnar(cfg).to(device)
        state = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        # Build a single-item batch
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([data.to(device)])
        with torch.no_grad():
            pred, _ = model(batch)
        # Compute node-level pointer accuracy on this single graph
        # Extract the predictions for this graph
        if model.output_type == "pointer":
            idx = batch.batch[batch.edge_index[0]] == 0
            pred_single = pred[idx]
        else:
            pred_single = pred[batch.batch == 0]
        acc = utils.pointer_accuracy(data, pred_single).item()
        print({
            "single_graph_pointer_accuracy": acc
        })

        # Build per-node parent for teacher and model (group by sender rows)
        # Teacher final pointers from edge_states[-1, :, :, 0]
        ptr_last = meta["edge_states"][-1, :, :, 0]
        parent_teacher = np.argmax(ptr_last, axis=1)
        # Model predicted parents: for each sender node v, choose edge with max score among edges with edge_index[0]==v
        ei = data.edge_index.cpu().numpy()
        pred_np = pred_single.detach().cpu().numpy()
        n = meta["grid"].n
        parent_model = np.zeros(n, dtype=int)
        for v in range(n):
            mask = (ei[0] == v)
            # fall back to self if no edges (should not happen for passable nodes)
            if not mask.any():
                parent_model[v] = v
            else:
                local_idx = np.argmax(pred_np[mask])
                # find the global index of the chosen edge among masked positions
                chosen_global = np.where(mask)[0][local_idx]
                parent_model[v] = int(ei[1, chosen_global])

        # Reconstruct teacher/model paths (s->t) for overlay
        def reconstruct_path(parents: np.ndarray, s: int, t: int, width: int, height: int) -> List[int]:
            seen = {t}
            cur = t
            path = [t]
            while cur != s:
                cur = int(parents[cur])
                if cur in seen or cur < 0:
                    return []
                seen.add(cur)
                path.append(cur)
            return list(reversed(path))

        s = meta["grid"].start
        t = meta["grid"].target
        w, h = meta["grid"].width, meta["grid"].height
        path_teacher = reconstruct_path(parent_teacher, s, t, w, h)
        path_model = reconstruct_path(parent_model, s, t, w, h)

        # Build side-by-side teacher vs model final overlays
        def build_overlay(path_nodes: List[int]) -> np.ndarray:
            base = np.zeros((h, w, 3), dtype=float)
            for node_id, passable in enumerate(meta["grid"].passable):
                x, y = _xy(node_id, w)
                base[y, x] = (1, 1, 1) if passable else (0, 0, 0)
            sx, sy = _xy(s, w)
            tx, ty = _xy(t, w)
            base[sy, sx] = (0.3, 0.9, 0.3)
            base[ty, tx] = (0.9, 0.3, 0.3)
            for u, v in zip(path_nodes, path_nodes[1:]):
                x1, y1 = _xy(u, w)
                x2, y2 = _xy(v, w)
                base[y1, x1] = (0.1, 0.2, 0.8)
                base[y2, x2] = (0.1, 0.2, 0.8)
            return base

        img_teacher = build_overlay(path_teacher)
        img_model = build_overlay(path_model)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(img_teacher)
        axes[0].set_title("Teacher final path")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[1].imshow(img_model)
        axes[1].set_title(f"Model final path (acc={acc*100:.1f}%)")
        axes[1].set_xticks([]); axes[1].set_yticks([])
        plt.tight_layout()
        plt.savefig(args.out_compare, dpi=180)
        print(f"Saved teacher vs model comparison to {args.out_compare}")

        # Mismatch heatmap (per-node)
        mismatch = (parent_model != parent_teacher)
        mm_img = np.zeros((h, w, 3), dtype=float)
        for node_id, passable in enumerate(meta["grid"].passable):
            x, y = _xy(node_id, w)
            if not passable:
                mm_img[y, x] = (0, 0, 0)
            else:
                # correct: light gray; mismatch: orange
                mm_img[y, x] = (0.9, 0.9, 0.9) if not mismatch[node_id] else (0.95, 0.6, 0.2)
        sx, sy = _xy(s, w)
        tx, ty = _xy(t, w)
        mm_img[sy, sx] = (0.3, 0.9, 0.3)
        mm_img[ty, tx] = (0.9, 0.3, 0.3)
        plt.figure(figsize=(4, 4))
        plt.imshow(mm_img)
        plt.title("Parent mismatch (model vs teacher)")
        plt.xticks([]); plt.yticks([])
        plt.tight_layout(); plt.savefig(args.out_mismatch, dpi=180)
        print(f"Saved mismatch heatmap to {args.out_mismatch}")

        # Combined overlay: teacher (blue) and model (magenta) on same grid
        overlay = np.zeros((h, w, 3), dtype=float)
        for node_id, passable in enumerate(meta["grid"].passable):
            x, y = _xy(node_id, w)
            overlay[y, x] = (1, 1, 1) if passable else (0, 0, 0)
        overlay[sy, sx] = (0.3, 0.9, 0.3)
        overlay[ty, tx] = (0.9, 0.3, 0.3)
        for u, v in zip(path_teacher, path_teacher[1:]):
            x1, y1 = _xy(u, w); x2, y2 = _xy(v, w)
            overlay[y1, x1] = (0.1, 0.2, 0.8)  # blue teacher
            overlay[y2, x2] = (0.1, 0.2, 0.8)
        for u, v in zip(path_model, path_model[1:]):
            x1, y1 = _xy(u, w); x2, y2 = _xy(v, w)
            overlay[y1, x1] = (0.8, 0.2, 0.8)  # magenta model
            overlay[y2, x2] = (0.8, 0.2, 0.8)
        plt.figure(figsize=(4, 4))
        plt.imshow(overlay)
        plt.title("Teacher (blue) vs Model (magenta)")
        plt.xticks([]); plt.yticks([])
        plt.tight_layout(); plt.savefig(args.out_overlay, dpi=180)
        print(f"Saved overlay comparison to {args.out_overlay}")


if __name__ == "__main__":
    torch.set_num_threads(5)
    torch.set_default_tensor_type(torch.DoubleTensor)
    main()
