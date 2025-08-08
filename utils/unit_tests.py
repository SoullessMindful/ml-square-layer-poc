from functools import reduce
from typing import List
import torch
import torch.nn as nn


def unit_tests_3vars(model: nn.Module):
    run_test(model, [1.5, -0.5, 1.2])
    run_test(model, [2.0, 1.4, -0.1])
    run_test(model, [-0.3, -0.4, -1.1])
    run_test(model, [0.15, 1.0, 1.6])
    run_test(model, [-1.0, 1.0, 1.0])


def unit_tests(model: nn.Module, variable_count: int):
    if variable_count == 3:
        unit_tests_3vars(model)


def run_test(model: nn.Module, numbers: List[float]):
    strings = [f"{n}" if n >= 0.0 else f"({n})" for n in numbers]
    product = reduce(lambda prod, n: prod * n, numbers, 1)
    product_estimate = float(model(torch.tensor(numbers))[0])
    print(f"{" * ".join(strings)} = {product:.6f} = {product_estimate:.6f}")
