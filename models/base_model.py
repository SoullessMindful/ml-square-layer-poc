import torch.nn as nn

from generate_data import VARIABLE_COUNT


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

        activation = nn.Tanh
        # activation = nn.SiLU
        # activation = nn.ReLU

        self.model = nn.Sequential(
            nn.Linear(VARIABLE_COUNT, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 96),
            activation(),
            nn.Linear(96, 192),
            activation(),
            nn.Linear(192, 1),
        )

    def forward(self, x):
        return self.model(x)
