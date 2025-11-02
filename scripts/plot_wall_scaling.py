from __future__ import annotations

import argparse
from typing import List

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


def eval_one(cfg: GridConfig, checkpoint_path: str, num_graphs: int, seed: int) -> dict:
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
    parser = argparse.ArgumentParser(description="Plot pointer accuracy vs wall percentage")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--densities", type=str, default="0.10,0.20,0.30,0.40",
                        help="Comma-separated list like 0.10,0.20,0.30,0.40")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--num_graphs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20251012)
    parser.add_argument("--out", type=str, default="wall_scaling.png")

    args = parser.parse_args()

    cfg: GridConfig = read_config(args.config_path)
    cfg.grid_weighted = False
    cfg.connectivity = 4
    cfg.ensure_connected = True
    cfg.output_type = "pointer"
    cfg.width = int(args.width)
    cfg.height = int(args.height)

    densities: List[float] = [float(x.strip()) for x in args.densities.split(",")]

    x_labels: List[str] = []
    accs: List[float] = []

    for p in densities:
        cfg.wall_pct = float(p)
        scores = eval_one(cfg, args.checkpoint_path, args.num_graphs, args.seed)
        acc = float(scores.get("pointer_accuracy", 0.0))
        x_labels.append(f"{p:.2f}")
        accs.append(acc)
        print({
            "size": (cfg.width, cfg.height),
            "wall_pct": cfg.wall_pct,
            "num_graphs": args.num_graphs,
            "seed": args.seed,
            "scores": scores,
        })

    plt.figure(figsize=(6, 4))
    plt.bar(x_labels, accs, color="#72B7B2")
    # Sensible dynamic y-limits
    mn, mx = min(accs), max(accs)
    rng = max(0.02, mx - mn)
    y_min = max(0.0, mn - 0.2 * rng)
    y_max = min(1.0, mx + 0.1 * rng)
    if y_max - y_min < 0.05:
        y_min = max(0.0, y_min - 0.02)
        y_max = min(1.0, y_max + 0.02)
    plt.ylim(y_min, y_max)
    plt.ylabel("Pointer accuracy (node-level)")
    plt.xlabel("Wall percentage")
    plt.title(f"DNAR-A* accuracy vs wall density (size={cfg.width}x{cfg.height})")
    for i, v in enumerate(accs):
        plt.text(i, min(y_max - 0.01, v + 0.01), f"{v*100:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    torch.set_num_threads(5)
    torch.set_default_tensor_type(torch.DoubleTensor)
    main()
