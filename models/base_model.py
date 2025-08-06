import torch as torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, variable_count: int):
        super(BaseModel, self).__init__()

        activation = nn.Tanh
        # activation = nn.SiLU
        # activation = nn.ReLU

        self.seq1 = nn.Sequential(
            nn.Linear(variable_count, 32),
            activation(),
            nn.Linear(32, 32),
            activation(),
            nn.Linear(32, 32),
            activation(),
        )
        
        self.skip = nn.Linear(variable_count, 32)
        
        self.seq2 = nn.Sequential(
            nn.Linear(32, 32),
            activation(),
            nn.Linear(32, 32),
            activation(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.seq2(self.seq1(x) + self.skip(x))
