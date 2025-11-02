DNAR Grid A* — Reproducible Setup, Training, and Evaluation

This project provides deterministic grid generation and A*-trace supervision, with training, evaluation, and visualization utilities.

Quickstart (Ubuntu, CPU-only)
- System packages:
  - `sudo apt update`
  - `sudo apt install python3 python3-pip python3-venv git`
- Create and activate a virtual environment:
  - `python3 -m venv dnar-env`
  - `source dnar-env/bin/activate`
- Clone the repo:
  - `git clone https://github.com/ShaoqinHou/maze-visual`
  - `cd maze-visual`
- Install Python dependencies (CPU wheels):
  - `pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu`
  - `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cpu.html`
  - `pip install torch-geometric==2.4.0`
  - `pip install numpy tqdm`

Training
- Unweighted A* (config C):
  - `python -u ./grid_train.py --config_path ./configs/grid_astar_C.yaml --num_seeds 1`

Evaluation and Figures
- 0: Eval a checkpoint on a small grid sample:
  - `python grid_eval.py --config_path configs/grid_astar_C.yaml --checkpoint_path models/grid_astar_uw_40_pointer_accuracy_best --num_graphs 2 --wall_pct 0.4 --seed 20251016 --width 6 --height 6`

- 1: Section 4, 5 charts (single-run visualizations: steps, f-heatmap, pointer overlays):
  - `python scripts/visualize_single_run.py --config_path configs/grid_astar_C.yaml --checkpoint_path models/grid_astar_uw_40_pointer_accuracy_best --width 30 --height 30 --wall_pct 0.30 --seed 20251016 --out_steps viz_steps.png --out_f viz_f_heatmap.png --out_compare viz_compare.png --out_mismatch viz_mismatch.png --out_overlay viz_overlay.png`

- 2: Size scaling plot (accuracy vs grid size):
  - `python scripts/plot_size_scaling.py --config_path configs/grid_astar_C.yaml --checkpoint_path models/grid_astar_uw_40_pointer_accuracy_best --sizes 4x4,6x6,10x10,30x30 --wall_pct 0.30 --num_graphs 2000 --num_graphs_large 20 --large_min_side 30 --seed 20251016 --out size_scaling.png`

- 3: Wall scaling plot (accuracy vs obstacle density):
  - `python scripts/plot_wall_scaling.py --config_path configs/grid_astar_C.yaml --checkpoint_path models/grid_astar_uw_40_pointer_accuracy_best --densities 0.10,0.20,0.30,0.40 --width 4 --height 4 --num_graphs 2000 --seed 20251016 --out wall_scaling.png`

Notes
- The checkpoint path above (`models/grid_astar_uw_40_pointer_accuracy_best`) refers to an A* model trained on unweighted grids. Replace with your own checkpoint as needed.
- CPU-only install shown; for CUDA, use the appropriate torch and PyG wheels.
- Windows users can consult `scripts/setup_windows_pyg.ps1` for PyG setup hints.

Grid-Specific Files (What They Do)
- `grid_algorithms.py`: Implements grid A* and emits full traces (node states, edge pointers, scalar fields) used for supervision.
- `grid_data_loader.py`: Builds grid datasets on the fly and packages features/labels compatible with the DNAR model.
- `grid_eval.py`: CLI to load a trained model and evaluate on freshly generated grid data with overrideable generation params.
- `grid_generate_data.py`: Deterministic grid generator (walls, neighbors, weights), shortest paths (BFS/Dijkstra), and utilities.
- `grid_test_generator.py`: Determinism and correctness suite (A* optimality vs. Dijkstra, trace equality, scalar semantics).
- `grid_train.py`: Training loop for grid configs, tensorboard logging, and checkpointing.
- `configs/grid_astar_*.yaml`: A* training presets (unweighted/weighted, sizes, training schedules). Notable: `grid_astar_C.yaml`.
- `configs/grid_bfs.yaml`, `configs/grid_dijkstra_*.yaml`: Alternative algorithms and baselines for comparison.
- `scripts/visualize_single_run.py`: Produces step-by-step A* visualizations and overlays.
- `scripts/plot_size_scaling.py`: Accuracy vs. grid size sweep.
- `scripts/plot_wall_scaling.py`: Accuracy vs. obstacle density sweep.
