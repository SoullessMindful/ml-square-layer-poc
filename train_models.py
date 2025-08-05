import json
import argparse
from typing import Final
import torch
import torch.nn as nn
import torch.optim as optim
from generate_data import DEFAULT_VARIABLE_COUNT, default_data_path
from losses import HybridMSELoss
from losses import RelativeMSELoss
from models import BaseModel, SquareModel, Model
from utils import Scheduler

DEFAULT_EPOCHS: Final = 1000
DEFAULT_BATCH_SIZE: Final = 32


def train_model(
    model: Model,
    X: torch.Tensor,
    y: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    batch_size: int,
):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    mse_loss_function: Final = nn.MSELoss()
    rmse_loss_function: Final = RelativeMSELoss()
    training_loss_function: Final = HybridMSELoss(alpha=0.1)
    optimizer: Final = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler: Final = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=7,
    )
    batch_scheduler: Final = Scheduler[float](
        lambda value, values: len(values) > 0 and value - min(values) > -1e-6,
        cache=5,
        patience=12,
    )
    early_stoppping_scheduler: Final = Scheduler[float](
        lambda value, values: len(values) > 0 and value - min(values) > 0,
        cache=10,
        patience=20,
    )

    for epoch in range(epochs):
        permutation = torch.randperm(X.size(0))
        epoch_loss = 0.0
        rmse_epoch_loss = 0.0
        mse_epoch_loss = 0.0

        for i in range(0, X.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X[indices], y[indices]

            variable_permutation = torch.randperm(X.size(1))
            batch_x = batch_x[:, variable_permutation]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = training_loss_function(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

            with torch.no_grad():
                rmse_epoch_loss += rmse_loss_function(
                    outputs, batch_y
                ).item() * batch_x.size(0)
                mse_epoch_loss += mse_loss_function(
                    outputs, batch_y
                ).item() * batch_x.size(0)

        avg_loss = epoch_loss / X.size(0)
        rmse_avg_loss = rmse_epoch_loss / X.size(0)
        mse_avg_loss = mse_epoch_loss / X.size(0)

        scheduler.step(avg_loss)
        batch_scheduler.step(avg_loss)
        if batch_scheduler.check() and batch_size > 64:
            batch_size //= 2
            print(f"Batch size: {batch_size}")

        with torch.no_grad():
            validation_outputs = model(X_val)
            validation_loss = training_loss_function(validation_outputs, y_val).item()
            rmse_validation_loss = rmse_loss_function(validation_outputs, y_val).item()
            mse_validation_loss = mse_loss_function(validation_outputs, y_val).item()

        print(
            f"Epoch {epoch+1}/{epochs}, "
            + f"TL: {avg_loss:.6f}, VL: {validation_loss:.6f}, "
            + f"RMSE TL: {rmse_avg_loss:.6f}, RMSE VL: {rmse_validation_loss:.6f}, "
            + f"MSE TL: {mse_avg_loss:.6f}, MSE VL: {mse_validation_loss:.6f}"
        )
        early_stoppping_scheduler.step(validation_loss)
        if early_stoppping_scheduler.check():
            print("Early stopping")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on generated data.")
    parser.add_argument(
        "-v",
        "--variable_count",
        type=int,
        default=DEFAULT_VARIABLE_COUNT,
        help="Amount of multiplied variables",
    )
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
        default="",
        help="Path to data file",
    )
    args = parser.parse_args()
    variable_count: Final[int] = args.variable_count
    data_path: Final[str] = (
        args.data_path if args.data_path != "" else default_data_path(variable_count)
    )
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
    base_model = BaseModel(variable_count)
    try:
        train_model(base_model, X, y, X_val, y_val, epochs, batch_size)
    except KeyboardInterrupt:
        pass 

    # print("Training SquareModel...")
    # square_model = SquareModel()
    # train_model(square_model, X, y, X_val, y_val, epochs, batch_size)

    if input("Do you want to save the base model state? ") == "y":
        model_state_path: str = input("Base model state path: ")
        model_state_path = (
            model_state_path
            if model_state_path != ""
            else f"./model_states/base_model_state_{variable_count}vars.pt"
        )
        torch.save(base_model.state_dict(), model_state_path)
        print("Saved to " + model_state_path)
