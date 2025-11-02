import argparse
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import models
import utils
from configs.grid_base_config import read_config, GridConfig
from grid_data_loader import create_grid_dataloader
import generate_data as gd


def evaluate(model, val_data, test_data, metrics_list, model_saver, writer, steps):
    with torch.no_grad():
        model.eval()
        val_scores = utils.evaluate(model, val_data, metrics_list)
        test_scores = utils.evaluate(model, test_data, metrics_list)
        print("Eval after {} steps:".format(steps))
        print("Val scores: ", val_scores)
        print("Test scores: ", test_scores)
        model.train()
    if writer is not None:
        for stat in val_scores:
            writer.add_scalar(f"{stat}/val", val_scores[stat], steps)
            writer.add_scalar(f"{stat}/test", test_scores[stat], steps)
    model_saver.visit(model, val_scores)


def train(config: GridConfig, seed, resume_path: str | None = None, resume_last: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Provide SPEC entry for A* compatible with dijkstra states layout
    if config.algorithm == "astar" and "astar" not in gd.SPEC:
        gd.SPEC["astar"] = gd.SPEC["dijkstra"]

    model = models.Dnar(config).to(device)

    # Build model name and optionally resume weights
    model_name = f"grid_{config.algorithm}_{'w' if config.grid_weighted else 'uw'}_{seed}"
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume path does not exist: {resume_path}")
        state = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Resumed model weights from: {resume_path}")
    elif resume_last:
        auto_path = os.path.join(config.models_directory, f"{model_name}_last")
        if os.path.exists(auto_path):
            state = torch.load(auto_path, map_location="cpu")
            model.load_state_dict(state)
            print(f"Resumed model weights from: {auto_path}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # model_name already defined above
    model_saver = utils.ModelSaver(config.models_directory, model_name)

    train_data = create_grid_dataloader(config, "train", seed=seed, device=device)
    val_data = create_grid_dataloader(config, "val", seed=seed + 1, device=device)
    test_data = create_grid_dataloader(config, "test", seed=seed + 2, device=device)

    writer = SummaryWriter(comment=f"-{model_name}") if config.tensorboard_logs else None

    model.train()

    steps = 0
    while steps <= config.num_iterations:
        for batch in train_data:
            steps += 1

            _, loss = model(batch, writer, training_step=steps)
            assert not torch.isnan(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            if steps % config.eval_each == 1:
                evaluate(
                    model,
                    val_data,
                    test_data,
                    utils.METRICS[config.output_type],
                    model_saver,
                    writer,
                    steps,
                )

            if steps >= config.num_iterations:
                break
    model.eval()
    return model


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_num_threads(5)
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/grid_bfs.yaml")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_last", action="store_true", help="Resume from models/<auto_name>_last if present")

    options = parser.parse_args()

    print("Train with grid-config {}".format(options.config_path))

    for seed in range(40, 40 + options.num_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        config = read_config(options.config_path)
        model = train(config, seed, resume_path=options.resume_path, resume_last=options.resume_last)


