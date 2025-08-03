import json
import argparse
from typing import Final
import torch
import torch.nn as nn
import torch.optim as optim
from generate_data import DEFAULT_DATA_PATH
from losses.relative_mse_loss import RelativeMSELoss
from models import BaseModel, SquareModel, Model
from utils import Scheduler

DEFAULT_EPOCHS: Final = 1000
DEFAULT_BATCH_SIZE: Final = 32


def train_model(model: Model, X, y, X_val, y_val, epochs: int, batch_size: int):
    loss_function: Final = RelativeMSELoss()
    optimizer: Final = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler: Final = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=7,
    )
    batch_scheduler: Final = Scheduler[float](
        lambda value, values: len(values) > 0 and value - min(values) > -1e-6,
        cache=5,
    )

    for epoch in range(epochs):
        permutation = torch.randperm(X.size(0))
        epoch_loss = 0.0

        for i in range(0, X.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X[indices], y[indices]

            variable_permutation = torch.randperm(X.size(1))
            batch_x = batch_x[:, variable_permutation]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / X.size(0)
        scheduler.step(avg_loss)
        batch_scheduler.step(avg_loss)
        if batch_scheduler.check():
            batch_size //= 2
            print(f"Batch size: {batch_size}")

        validation_outputs = model(X_val)
        validation_loss = loss_function(validation_outputs, y_val).item()
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Validation Loss: {validation_loss:.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on generated data.")
    parser.add_argument(
        "-e", "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to data file",
    )
    args = parser.parse_args()

    data_path: Final[str] = args.data_path
    batch_size: Final[int] = args.batch_size
    epochs: Final[int] = args.epochs

    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        inputs = data["inputs"]
        outputs = data["outputs"]
        validation_inputs = data["validation_inputs"]
        validation_outputs = data["validation_outputs"]

    X = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(outputs, dtype=torch.float32)
    X_val = torch.tensor(validation_inputs, dtype=torch.float32)
    y_val = torch.tensor(validation_outputs, dtype=torch.float32)

    print(f"Training data: {data_path} {X.shape} {y.shape}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")

    print("Training BaseModel...")
    base_model = BaseModel()
    try:
        train_model(base_model, X, y, X_val, y_val, epochs, batch_size)
    except KeyboardInterrupt:
        pass 

    # print("Training SquareModel...")
    # square_model = SquareModel()
    # train_model(square_model, X, y, X_val, y_val, epochs, batch_size)
