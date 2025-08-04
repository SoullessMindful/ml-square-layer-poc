import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self, variable_count: int):
        super(BaseModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(variable_count, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)
