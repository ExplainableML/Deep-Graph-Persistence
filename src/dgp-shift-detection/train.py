import os
import torch
import shutil
import random
import argparse
import numpy as np

from load_model import load_model
from load_data import get_data_shape
from pytorch_lightning import Trainer
from load_data import load_dataset_for_training
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def _parse_arguments() -> argparse.Namespace:
    # Instantiate argument parser
    parser = argparse.ArgumentParser("MLP Trainer")

    # Add Model hyperparameters
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--hidden", type=int, help="Number of hidden units", default=50)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
    )
    parser.add_argument(
        "--initialization",
        type=str,
        default="normal",
    )
    parser.add_argument("--dropout", type=float, default=0.0)

    # Add Data hyperparameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
    )
    parser.add_argument("--batch", type=int, default=32)

    # Add Training hyperparameters
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_root_dir", type=str, default="./saved_models/")

    # Parse command line arguments
    hyperparameters = parser.parse_args()
    return hyperparameters


def single_model_run(hyperparameters: argparse.Namespace, name: str, run: int) -> None:
    # Load Data
    # Set fixed seeds for loading data
    # This is required because of the random splitting in train/validation portions
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train_data, validation_data = load_dataset_for_training(
        name=hyperparameters.dataset,
        batch_size=hyperparameters.batch,
    )
    input_size, num_classes = get_data_shape(hyperparameters.dataset)

    # Set seeds
    run_seed = hyperparameters.seed + run
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    # Define Logger & Callbacks
    log_path = os.path.join("./logs", name, f"run_{run}")
    shutil.rmtree(log_path, ignore_errors=True)
    os.makedirs(log_path, exist_ok=True)
    checkpoint_save_path = os.path.join(
        hyperparameters.checkpoint_root_dir, name, f"run_{run}"
    )
    shutil.rmtree(checkpoint_save_path, ignore_errors=True)
    os.makedirs(checkpoint_save_path, exist_ok=True)

    if len(os.listdir(log_path)) > 0:
        raise RuntimeError(f"Logging path {log_path} not empty")
    if len(os.listdir(checkpoint_save_path)) > 0:
        raise RuntimeError(f"Checkpoint save path {checkpoint_save_path} not empty")

    logger = CSVLogger(save_dir=log_path, name=name)
    checkpointer = ModelCheckpoint(
        dirpath=checkpoint_save_path,
        save_last=True,
        save_top_k=1,
        mode="min",
        save_on_train_epoch_end=False,
    )

    # Instantiate Model
    model = load_model(
        name=hyperparameters.model,
        in_size=input_size,
        num_classes=num_classes,
        parameters=hyperparameters,
    )

    # Make Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpointer],
        max_epochs=hyperparameters.epochs,
        check_val_every_n_epoch=1,
        enable_progress_bar=hyperparameters.verbose,
        enable_model_summary=hyperparameters.verbose,
        gradient_clip_val=None,
    )

    # Train Model
    trainer.fit(
        model=model, train_dataloaders=train_data, val_dataloaders=validation_data
    )


def experiment() -> None:
    hyperparameters = _parse_arguments()
    name = hyperparameters.name
    runs = hyperparameters.runs

    for run in range(runs):
        single_model_run(hyperparameters=hyperparameters, name=name, run=run)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    experiment()
