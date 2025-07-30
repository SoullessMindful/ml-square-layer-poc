from functools import reduce
import json
import random
import sys
from typing import Final


VARIABLE_COUNT: Final = 10
INPUTS_COUNT: Final = 10000
DEFAULT_DATA_PATH: Final = "./data/data.json"


def generate_data():
    inputs: Final = [
        [random.uniform(-2, 2) for _ in range(VARIABLE_COUNT)]
        for _ in range(INPUTS_COUNT)
    ]
    outputs: Final = [
        [reduce(lambda product, element: product * element, input_row, 1.0)]
        for input_row in inputs
    ]
    return inputs, outputs


if __name__ == "__main__":
    inputs, outputs = generate_data()

    path: Final = './data/data.json' if (len(sys.argv) <= 1) else sys.argv[1]

    with open(path, "w", encoding="utf-8") as file:
        json.dump({ "inputs": inputs, "outputs": outputs}, file, indent=2)
