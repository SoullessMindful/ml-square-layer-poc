from typing import Callable


class Scheduler[T]:
    def __init__(
        self,
        condition: Callable[[T, list[T]], bool],
        patience: int = 10,
        cache: int = 1,
    ):
        self.condition = condition
        self.patience = patience
        self.cache = cache
        self.counter: int = 0
        self.values: list[T] = []

    def step(self, value: T) -> None:
        if self.condition(value, self.values):
            self.counter += 1
        else:
            self.counter = 0

        self.values.append(value)
        if len(self.values) > self.cache:
            self.values.pop(0)

    def check(self) -> bool:
        if self.counter >= self.patience:
            self.counter = 0
            return True
        return False
