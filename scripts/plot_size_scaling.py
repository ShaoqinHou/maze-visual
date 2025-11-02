from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import utils
import models
import generate_data as gd
from configs.grid_base_config import read_config, GridConfig
from grid_data_loader import create_grid_dataloader


@dataclass
class SizeSetting:
    width: int
    height: int


def eval_one(cfg: GridConfig, checkpoint_path: str, num_graphs: int, seed: int) -> dict:
    # Ensure SPEC is aligned for A*
    if cfg.algorithm == "astar" and "astar" not in gd.SPEC:
        gd.SPEC["astar"] = gd.SPEC["dijkstra"]

    cfg.num_samples = {"test": int(num_graphs)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.Dnar(cfg).to(device)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    loader = create_grid_dataloader(cfg, split="test", seed=int(seed), device=device)
    with torch.no_grad():
        scores = utils.evaluate(model, loader, utils.METRICS[cfg.output_type])
    return scores


def main():
    parser = argparse.ArgumentParser(description="Plot pointer accuracy vs grid size")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--sizes", type=str, default="4x4,6x6,10x10,30x30",
                        help="Comma-separated list like 4x4,10x10,30x30")
    parser.add_argument("--wall_pct", type=float, default=0.30)
    parser.add_argument("--num_graphs", type=int, default=1000,
                        help="Graphs per size for normal sizes")
    parser.add_argument("--num_graphs_large", type=int, default=20,
                        help="Graphs for large sizes (see --large_min_side)")
    parser.add_argument("--large_min_side", type=int, default=30,
                        help="If max(width,height) >= this, use num_graphs_large")
    parser.add_argument("--seed", type=int, default=20251012)
    parser.add_argument("--out", type=str, default="size_scaling.png")

    args = parser.parse_args()

    cfg: GridConfig = read_config(args.config_path)
    cfg.grid_weighted = False
    cfg.connectivity = 4
    cfg.ensure_connected = True
    cfg.output_type = "pointer"

    # Parse sizes
    sizes: List[SizeSetting] = []
    for token in args.sizes.split(","):
        token = token.strip().lower()
        if "x" not in token:
            raise ValueError(f"Invalid size token: {token}")
        w, h = token.split("x")
        sizes.append(SizeSetting(width=int(w), height=int(h)))

    x_labels: List[str] = []
    accs: List[float] = []
    counts: List[int] = []

    for s in sizes:
        # Override generation params
        cfg.width = s.width
        cfg.height = s.height
        cfg.wall_pct = float(args.wall_pct)

        # Choose count based on size (limit for large grids)
        num_graphs = int(args.num_graphs_large if max(s.width, s.height) >= args.large_min_side else args.num_graphs)

        import time
        t0 = time.perf_counter()
        scores = eval_one(cfg, args.checkpoint_path, num_graphs, args.seed)
        dt = time.perf_counter() - t0
        rate = (num_graphs / dt) if dt > 0 else float('inf')
        acc = float(scores.get("pointer_accuracy", 0.0))
        x_labels.append(f"{s.width}x{s.height}")
        accs.append(acc)
        counts.append(num_graphs)
        print({
            "size": (s.width, s.height),
            "wall_pct": cfg.wall_pct,
            "num_graphs": num_graphs,
            "seed": args.seed,
            "elapsed_sec": round(dt, 3),
            "rate_items_per_sec": round(rate, 2),
            "scores": scores,
        })

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(x_labels, accs, color="#4C78A8")
    # Sensible dynamic y-limits to show small differences clearly
    mn, mx = min(accs), max(accs)
    rng = max(0.02, mx - mn)
    y_min = max(0.0, mn - 0.2 * rng)
    y_max = min(1.0, mx + 0.1 * rng)
    if y_max - y_min < 0.05:
        y_min = max(0.0, y_min - 0.02)
        y_max = min(1.0, y_max + 0.02)
    plt.ylim(y_min, y_max)
    plt.ylabel("Pointer accuracy (node-level)")
    plt.xlabel("Grid size (W x H)")
    plt.title(f"DNAR-A* accuracy vs size (wall_pct={args.wall_pct})")
    for i, v in enumerate(accs):
        note = f"{v*100:.1f}%\n(n={counts[i]})"
        plt.text(i, min(y_max - 0.01, v + 0.01), note, ha="center")
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    torch.set_num_threads(5)
    torch.set_default_tensor_type(torch.DoubleTensor)
    main()
