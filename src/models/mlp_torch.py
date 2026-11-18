"""Small PyTorch MLP with validation-based early stopping."""

from __future__ import annotations

import copy
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.common import make_preprocessor


class TrafficMLP(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_mlp(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    target: str,
    seed: int = 42,
    max_epochs: int = 120,
    patience: int = 12,
) -> Tuple[np.ndarray, Dict[str, object]]:
    set_seeds(seed)
    preprocess = make_preprocessor(features, scale_numeric=True)
    x_train = preprocess.fit_transform(train[features]).astype("float32")
    x_val = preprocess.transform(val[features]).astype("float32")
    y_train = train[target].to_numpy(dtype="float32")
    y_val = val[target].to_numpy(dtype="float32")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = TrafficMLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=100.0)

    x_val_tensor = torch.from_numpy(x_val)
    y_val_tensor = torch.from_numpy(y_val)
    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    stale_epochs = 0
    epochs_run = 0

    for epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mae = torch.mean(torch.abs(model(x_val_tensor) - y_val_tensor)).item()
        epochs_run = epoch + 1
        if val_mae < best_mae - 0.25:
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(x_val_tensor).numpy().clip(min=0)
    details = {
        "hidden_layers": [128, 64],
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs_run": epochs_run,
        "best_validation_mae": round(best_mae, 4),
    }
    return predictions, details


def fit_predict_mlp_fixed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    epochs: int,
    seed: int = 42,
) -> np.ndarray:
    """Refit the chosen MLP schedule on train+validation and predict test once."""
    set_seeds(seed)
    preprocess = make_preprocessor(features, scale_numeric=True)
    x_train = preprocess.fit_transform(train[features]).astype("float32")
    x_test = preprocess.transform(test[features]).astype("float32")
    y_train = train[target].to_numpy(dtype="float32")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = TrafficMLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=100.0)
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x_test)).numpy().clip(min=0)
