import torch
import torch.nn as nn


def unit_tests_3vars(model: nn.Module):
    print(f"1.5 * (-0.5) * 1.2 = -0.9 = {model(torch.tensor([1.5, -0.5, 1.2]))}")
    print(f"2.0 * 1.4 * (-0.1) = -0.28 = {model(torch.tensor([2.0, 1.4, -0.1]))}")
    print(f"(-0.3) * (-0.4) * (-1.1) = -0.132 = {model(torch.tensor([-0.3, -0.4, -1.1]))}")
    print(f"0.15 * 1.0 * 1.6 = 0.24 = {model(torch.tensor([0.15, 1.0, 1.6]))}")
    print(f"(-1.0) * 1.0 * 1.0 = -1.0 = {model(torch.tensor([-1.0, 1.0, 1.0]))}")


def unit_tests(model: nn.Module, variable_count: int):
    if variable_count == 3:
        unit_tests_3vars(model)
