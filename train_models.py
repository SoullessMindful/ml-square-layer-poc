import json
import sys
from typing import Final
import torch
import torch.nn as nn
import torch.optim as optim
from models import BaseModel, Model

DEFAULT_EPOCHS: Final = 500
DEFAULT_BATCH_SIZE: Final = 256

def train_model(model: Model, epochs: int, batch_size: int):
    loss_function: Final = nn.MSELoss()
    optimizer: Final = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        permutation = torch.randperm(X.size(0))
        epoch_loss = 0.0

        for i in range(0, X.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X[indices], y[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / X.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    DATA_PATH = "./data/data.json" if (len(sys.argv) <= 1) else sys.argv[1]

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
        inputs = data["inputs"]
        outputs = data["outputs"]

    X = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(outputs, dtype=torch.float32)

    base_model = BaseModel()
    train_model(base_model, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE)
