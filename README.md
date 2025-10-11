# Discrete Neural Algorithmic Reasoning
This repository contains the official implementation of the experiments from the paper
"Discrete Neural Algorithmic Reasoning".

## Setup
Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data generation
To generate supervised traces for a given algorithm, run:

```bash
python generate_data.py \
  --algorithm bfs \
  --output_dir data/bfs \
  --num_graphs 1000
```

Available algorithms include `bfs`, `dfs`, `dijkstra`, `mis`, and `mst`.

## Training
Train a single-task model by providing the relevant configuration file:

```bash
python train.py --config_path configs/bfs.yaml
```

## Evaluation
Evaluate a trained checkpoint with:

```bash
python eval.py --config_path configs/bfs.yaml --checkpoint_path runs/bfs/best.ckpt
```

## Repository structure
- `configs/`: Algorithm-specific training presets.
- `generate_data.py`: Utilities for building graph datasets with algorithm traces.
- `train.py`: Entry point for model optimisation.
- `eval.py`: Evaluation scripts for trained models.
- `processors.py`, `models.py`, `utils.py`: Core model and processing modules.
