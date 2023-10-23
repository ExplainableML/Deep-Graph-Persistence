import torch
import numpy as np
import torch.nn as nn

from dnp import get_dnp
from torch import Tensor
from typing import Tuple
from metrics import get_variance
from metrics import get_sortedness
from collections import OrderedDict
from metrics import get_neural_persistence
from pytorch_lightning import LightningModule


class MLP(LightningModule):
    activation_functions = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    initializations = {
        "zeros": nn.init.zeros_,
        "normal": nn.init.xavier_normal_,
        "uniform": nn.init.xavier_uniform_,
    }

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        activation_function: str,
        initialization: str,
        batch_norm: bool,
        dropout: float,
        log_neural_persistence: bool,
        log_sortedness: bool,
        log_variance: bool,
        log_dnp: bool,
    ) -> None:
        super().__init__()

        # Save Hyperparameters
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation_function = activation_function
        self.initialization = initialization
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.log_neural_persistence = log_neural_persistence
        self.log_sortedness = log_sortedness
        self.log_variance = log_variance
        self.log_dnp = log_dnp

        self._check_hyperparameters()
        self.save_hyperparameters()

        # Define Model Architecture
        self.linear_layers = nn.ModuleList()
        self.nonlinearities = nn.ModuleList()

        # Define Hidden Layers
        in_size = self.input_size

        for layer_index in range(self.num_layers):
            # Make Linear Layer
            linear = nn.Linear(in_size, self.hidden_size, bias=True)
            self.initializations[self.initialization](linear.weight)

            # Make Activation + Dropout + Batch Norm
            activation = self.activation_functions[self.activation_function]
            dropout = nn.Dropout(p=self.dropout)
            if self.batch_norm:
                batch_norm = nn.BatchNorm1d(num_features=self.hidden_size)
            else:
                batch_norm = nn.Identity()
            nonlinearity = nn.Sequential(activation, dropout, batch_norm)

            # Append Modules
            self.linear_layers.append(linear)
            self.nonlinearities.append(nonlinearity)

            # Update input size
            in_size = self.hidden_size

        # Define Output Layer
        output_linear = nn.Linear(in_size, self.num_classes, bias=True)
        self.initializations[self.initialization](output_linear.weight)
        self.linear_layers.append(output_linear)
        self.nonlinearities.append(nn.Identity())

        # Combine all Modules
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

        # Initialize containers for epoch metrics
        self._reset_epoch_statistics()

    def mlp(self, x: Tensor) -> Tensor:
        for linear, nonlinearity in zip(self.linear_layers, self.nonlinearities):
            x = linear(x)
            x = nonlinearity(x)
        return x

    def _reset_epoch_statistics(self) -> None:
        self.train_losses = []
        self.train_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

        self.variances = []
        self.neural_persistence_values = []
        self.layer_global_persistence_values = {
            layer_index: [] for layer_index in range(self.num_layers + 1)
        }
        self.sortedness_values = {
            layer_index: [] for layer_index in range(self.num_layers + 1)
        }

    def _check_hyperparameters(self) -> None:
        assert isinstance(self.input_size, int) and self.input_size > 0
        assert isinstance(self.num_classes, int) and self.num_classes > 0
        assert isinstance(self.num_layers, int) and self.num_layers >= 0
        assert isinstance(self.log_neural_persistence, bool)
        assert isinstance(self.log_sortedness, bool)
        assert isinstance(self.log_variance, bool)
        assert isinstance(self.log_dnp, bool)

        if self.num_layers > 0:
            assert isinstance(self.hidden_size, int) and self.hidden_size > 0
            assert self.activation_function in self.activation_functions
            assert self.initialization in self.initializations
            assert isinstance(self.batch_norm, bool)
            assert isinstance(self.dropout, float) and 0 <= self.dropout <= 1

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, labels = batch
        inputs = inputs.reshape(inputs.shape[0], -1)
        labels = labels.flatten()

        logits = self.mlp(inputs)
        losses = self.cross_entropy(logits, labels.flatten())
        batch_loss = losses.mean()

        # Save detached losses
        self.train_losses.extend(losses.detach().cpu().flatten().tolist())

        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        correct = torch.eq(predictions, labels)
        correct = correct.detach().cpu().tolist()
        self.train_accuracies.extend(correct)

        # Return batch loss
        return batch_loss

    def evaluation_step(self, batch: Tuple[Tensor, Tensor], stage: str) -> None:
        # Get Logits
        inputs, labels = batch
        inputs = inputs.reshape(inputs.shape[0], -1)
        labels = labels.flatten()

        logits = self.mlp(inputs)

        # Get losses
        losses = self.cross_entropy(logits, labels.flatten())
        losses = losses.detach().cpu().flatten().tolist()

        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        correct = torch.eq(predictions, labels)
        correct = correct.detach().cpu().tolist()

        if stage == "validation":
            self.validation_losses.extend(losses)
            self.validation_accuracies.extend(correct)
        elif stage == "test":
            self.test_losses.extend(losses)
            self.test_accuracies.extend(correct)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if dataloader_idx == 0:
            self.evaluation_step(batch=batch, stage="validation")
        elif dataloader_idx == 1:
            self.evaluation_step(batch=batch, stage="test")
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")

    def on_validation_epoch_end(self) -> None:
        # Log Data related metrics
        self.log("train_loss", np.mean(self.train_losses).item())
        self.log("train_accuracy", np.mean(self.train_accuracies).item())
        if self.validation_losses:
            self.log("validation_loss", np.mean(self.validation_losses).item())
        if self.validation_accuracies:
            self.log(
                "validation_accuracy",
                np.mean(self.validation_accuracies).item(),
                prog_bar=True,
            )
        if self.test_losses:
            self.log("test_loss", np.mean(self.test_losses).item())
        if self.test_accuracies:
            self.log("test_accuracy", np.mean(self.test_accuracies).item())

        # Log Weight related metrics
        weights = self.get_weights()

        if self.log_neural_persistence:
            global_persistence, local_persistence = get_neural_persistence(
                weights=weights
            )
            self.log("neural_persistence", global_persistence, prog_bar=True)

            for layer_index in sorted(weights.keys()):
                self.log(
                    f"layer_{layer_index}_persistence", local_persistence[layer_index]
                )

        if self.log_sortedness:
            sortedness = get_sortedness(weights=weights)

            for layer_index in sorted(weights.keys()):
                self.log(f"layer_{layer_index}_sortedness", sortedness[layer_index])

        if self.log_variance:
            variance = get_variance(weights=weights)
            self.log("variance", variance)

        if self.log_dnp:
            for key, value in get_dnp(weights=weights).items():
                self.log(key, value)

        # Reset Statistics
        self._reset_epoch_statistics()

    def get_weights(self) -> OrderedDict:
        weights = OrderedDict()
        for layer_index in range(self.num_layers + 1):
            linear = self.linear_layers[layer_index]
            weight_matrix = linear.weight.detach().cpu().numpy()
            weights[layer_index] = weight_matrix
        return weights
