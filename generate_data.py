import argparse
from functools import reduce
import json
import random
from typing import Final


DEFAULT_VARIABLE_COUNT: Final = 2
DEFAULT_INPUTS_COUNT: Final = 10000
DEFAULT_VALIDATION_INPUTS_COUNT: Final = 2000


def default_data_path(variable_count):
    return f"./data/data_{variable_count}vars.json"


def generate_data(
    inputs_count: int,
    validation_inputs_count: int,
    variable_count: int,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    inputs: Final = [
        [random.uniform(-2, 2) for _ in range(variable_count)]
        for _ in range(inputs_count + validation_inputs_count)
    ]
    outputs: Final = [
        [reduce(lambda product, element: product * element, input_row, 1.0)]
        for input_row in inputs
    ]

    return (
        inputs[:inputs_count],
        outputs[:inputs_count],
        inputs[inputs_count:],
        outputs[inputs_count:],
    )


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
        "-s",
        "--inputs_count",
        type=int,
        default=DEFAULT_INPUTS_COUNT,
        help="Size of the training set",
    )
    parser.add_argument(
        "-vs",
        "--validation_inputs_count",
        type=int,
        default=DEFAULT_VALIDATION_INPUTS_COUNT,
        help="Size of the validation set",
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
    inputs_count: Final[int] = args.inputs_count
    validation_inputs_count: Final[int] = args.validation_inputs_count
    data_path: Final[str] = (
        args.data_path
        if args.data_path != ""
        else default_data_path(variable_count)
    )

    print(f"Generating data at {data_path}")
    print(f"Training set size: {inputs_count}")
    print(f"Validation set size: {validation_inputs_count}")

    inputs, outputs, validation_inputs, validation_outputs = generate_data(
        inputs_count,
        validation_inputs_count,
        variable_count,
    )

    with open(data_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "inputs": inputs,
                "outputs": outputs,
                "validation_inputs": validation_inputs,
                "validation_outputs": validation_outputs,
            },
            file,
            indent=2,
        )
