from dataclasses import dataclass
import yaml


@dataclass
class GridConfig:
    # --- task ---
    algorithm: str = None

    # --- train ---
    batch_size: int = 2
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    num_iterations: int = 4
    eval_each: int = 4
    stepwise_training: bool = True
    processor_upper_t: float = 3.0
    processor_lower_t: float = 0.01
    use_noise: bool = True

    # --- data (counts) ---
    num_samples: dict | None = None  # {train, val, test}

    # --- grid generation ---
    width: int = 4
    height: int = 4
    wall_pct: float = 0.2
    connectivity: int = 4  # 4 or 8
    ensure_connected: bool = True
    grid_weighted: bool = False

    # --- model ---
    h: int = 128
    temp_on_eval: float = 0.0
    num_node_states: int = 1
    num_edge_states: int = 1
    output_type: str = "pointer"
    output_idx: int = 0

    # flags for compatibility with processors, not used for grid by default
    generate_random_numbers: bool = False

    # --- io ---
    models_directory: str = "models"
    tensorboard_logs: bool = True


def read_config(config_path: str) -> GridConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return GridConfig(**cfg)

