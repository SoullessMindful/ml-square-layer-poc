import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self, variable_count: int):
        super(BaseModel, self).__init__()

        activation = nn.Tanh
        # activation = nn.SiLU
        # activation = nn.ReLU

        self.model = nn.Sequential(
            nn.Linear(variable_count, 64),
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
