import os
import torch
import shutil
import random
import argparse
import numpy as np

from mlp import MLP
from data import DataModule
from data import MNISTDataModule
from data import EMNISTDataModule
from pytorch_lightning import Trainer
from data import FashionMNISTDataModule
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def _parse_arguments() -> argparse.Namespace:
    # Instantiate argument parser
    parser = argparse.ArgumentParser("Neural Persistence Model Trainer")

    # Add MLP hyperparameters
    parser.add_argument("--hidden", type=int, help="Number of hidden units", default=50)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--activation",
        type=str,
        choices=list(MLP.activation_functions.keys()),
        default="relu",
    )
    parser.add_argument(
        "--initialization",
        type=str,
        choices=list(MLP.initializations.keys()),
        default="normal",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batchnorm", action="store_true")

    # Add Data hyperparameters
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fashion-mnist", "emnist"],
        default="mnist",
    )
    parser.add_argument("--batch", type=int, default=32)

    # Add Training hyperparameters
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_root_dir", type=str, default="./checkpoints")

    # Parse command line arguments
    hyperparameters = parser.parse_args()
    return hyperparameters


def _make_experiment_name(hyperparameters: argparse.Namespace) -> str:
    # Convert Namespace to dictionary
    hyperparameters = vars(hyperparameters)
    # del hyperparameters["checkpoint_root_dir"]

    # Convert hyperparameters to string
    hyperparameters = [
        (parameter, str(hyperparameters[parameter]))
        for parameter in sorted(hyperparameters.keys())
        if parameter != "checkpoint_root_dir"
    ]
    return "_".join([argument + "=" + value for argument, value in hyperparameters])


def _load_dataset(hyperparameters: argparse.Namespace) -> DataModule:
    if hyperparameters.dataset == "mnist":
        data = MNISTDataModule(
            data_dir="./data/mnist", batch_size=hyperparameters.batch
        )
    elif hyperparameters.dataset == "fashion-mnist":
        data = FashionMNISTDataModule(
            data_dir="./data/fashion_mnist", batch_size=hyperparameters.batch
        )
    elif hyperparameters.dataset == "emnist":
        data = EMNISTDataModule(
            data_dir="./data/emnist", batch_size=hyperparameters.batch
        )
    else:
        raise ValueError(f"Unknown Dataset: {hyperparameters.dataset}")

    return data


def single_model_run(hyperparameters: argparse.Namespace, name: str, run: int) -> None:
    # Load Data
    # Set fixed seeds for loading data
    # This is required because of the random splitting in train/validation portions
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data = _load_dataset(hyperparameters=hyperparameters)
    data.setup()

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
        hyperparameters.checkpoint_root_dir, "saved_models/", name, f"run_{run}"
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
        monitor="validation_loss",
        save_last=True,
        save_top_k=-1,
        mode="min",
        save_on_train_epoch_end=False,
    )

    # Instantiate Model
    model = MLP(
        input_size=data.input_size,
        num_classes=data.num_classes,
        num_layers=hyperparameters.layers,
        hidden_size=hyperparameters.hidden,
        activation_function=hyperparameters.activation,
        initialization=hyperparameters.initialization,
        dropout=hyperparameters.dropout,
        batch_norm=hyperparameters.batchnorm,
        log_neural_persistence=True,
        log_sortedness=True,
        log_variance=True,
        log_dnp=True,
    )
    model = model.cuda()

    # Make Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        logger=logger,
        callbacks=[checkpointer],
        max_epochs=hyperparameters.epochs,
        val_check_interval=0.25,
        enable_progress_bar=hyperparameters.verbose,
        enable_model_summary=hyperparameters.verbose,
        gradient_clip_val=None,
    )

    # Save initial checkpoint
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyperparameters": vars(hyperparameters),
            "name": name,
            "run": run,
        },
        os.path.join(checkpoint_save_path, "initialization.ckpt"),
    )

    # Train Model
    trainer.fit(model=model, datamodule=data)


def experiment() -> None:
    hyperparameters = _parse_arguments()
    name = _make_experiment_name(hyperparameters=hyperparameters)
    runs = hyperparameters.runs

    for run in range(runs):
        single_model_run(hyperparameters=hyperparameters, name=name, run=run)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    experiment()
