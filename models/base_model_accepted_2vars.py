import torch.nn as nn

from generate_data import VARIABLE_COUNT

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(VARIABLE_COUNT, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)
