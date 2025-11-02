import argparse
from typing import Optional

import torch

import utils
import models
from configs.grid_base_config import read_config, GridConfig
from grid_data_loader import create_grid_dataloader
import generate_data as gd


def to_bool(x: Optional[str | int | bool], default: bool) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    try:
        return bool(int(x))
    except Exception:
        return str(x).lower() in {"true", "t", "yes", "y"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to grid_* YAML config used for training (for model hyperparams and algorithm)")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint (torch.save state_dict)")

    # Generation overrides
    parser.add_argument("--num_graphs", type=int, default=200, help="Number of eval graphs")
    parser.add_argument("--width", type=int, default=None, help="Grid width override")
    parser.add_argument("--height", type=int, default=None, help="Grid height override")
    parser.add_argument("--wall_pct", type=float, default=None, help="Wall probability in [0,1)")
    parser.add_argument("--connectivity", type=int, default=None, help="4 or 8 neighbors")
    parser.add_argument("--ensure_connected", default=None, help="1/0 to resample until (s,t) reachable")
    parser.add_argument("--grid_weighted", default=None, help="1/0 to use weighted edges (Dijkstra/A*)")
    parser.add_argument("--seed", type=int, default=1234, help="Eval data seed base")

    # Optional model/eval overrides (use with caution; should match training)
    parser.add_argument("--algo", type=str, default=None, help="Override algorithm (e.g., astar)")
    parser.add_argument("--h", type=int, default=None, help="Override model hidden size")
    parser.add_argument("--num_node_states", type=int, default=None, help="Override num_node_states")
    parser.add_argument("--num_edge_states", type=int, default=None, help="Override num_edge_states")
    parser.add_argument("--output_type", type=str, default=None, help="Override output_type (pointer|node_mask)")
    parser.add_argument("--output_idx", type=int, default=None, help="Override output_idx (channel index)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval batch size")

    args = parser.parse_args()

    # Load config and override generation params
    cfg: GridConfig = read_config(args.config_path)

    if args.width is not None:
        cfg.width = int(args.width)
    if args.height is not None:
        cfg.height = int(args.height)
    if args.wall_pct is not None:
        cfg.wall_pct = float(args.wall_pct)
    if args.connectivity is not None:
        cfg.connectivity = int(args.connectivity)
    if args.ensure_connected is not None:
        cfg.ensure_connected = to_bool(args.ensure_connected, cfg.ensure_connected)
    if args.grid_weighted is not None:
        cfg.grid_weighted = to_bool(args.grid_weighted, cfg.grid_weighted)

    # Create a dedicated eval split size
    cfg.num_samples = {"test": int(args.num_graphs)}

    # Align SPEC for A*
    if cfg.algorithm == "astar" and "astar" not in gd.SPEC:
        gd.SPEC["astar"] = gd.SPEC["dijkstra"]

    # Optional model overrides
    if args.algo is not None:
        cfg.algorithm = str(args.algo)
        if cfg.algorithm == "astar" and "astar" not in gd.SPEC:
            gd.SPEC["astar"] = gd.SPEC["dijkstra"]
    if args.h is not None:
        cfg.h = int(args.h)
    if args.num_node_states is not None:
        cfg.num_node_states = int(args.num_node_states)
    if args.num_edge_states is not None:
        cfg.num_edge_states = int(args.num_edge_states)
    if args.output_type is not None:
        cfg.output_type = str(args.output_type)
    if args.output_idx is not None:
        cfg.output_idx = int(args.output_idx)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.Dnar(cfg).to(device)
    state = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    loader = create_grid_dataloader(cfg, split="test", seed=int(args.seed), device=device)

    with torch.no_grad():
        scores = utils.evaluate(model, loader, utils.METRICS[cfg.output_type])
        print("Eval config override:")
        print({
            "width": cfg.width,
            "height": cfg.height,
            "wall_pct": cfg.wall_pct,
            "connectivity": cfg.connectivity,
            "ensure_connected": cfg.ensure_connected,
            "grid_weighted": cfg.grid_weighted,
            "num_graphs": args.num_graphs,
            "seed": args.seed,
        })
        print("Scores:", scores)


if __name__ == "__main__":
    torch.set_num_threads(5)
    torch.set_default_tensor_type(torch.DoubleTensor)
    main()
