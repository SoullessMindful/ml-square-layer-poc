import torch.nn as nn

from generate_data import VARIABLE_COUNT


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

        activation = nn.Tanh
        # activation = nn.SiLU
        # activation = nn.ReLU

        self.model = nn.Sequential(
            nn.Linear(VARIABLE_COUNT, 32),
            activation(),
            nn.Linear(32, 32),
            activation(),
            nn.Linear(32, 64),
            activation(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)
