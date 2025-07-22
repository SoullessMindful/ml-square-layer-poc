import torch.nn as nn

from generate_data import VARIABLE_COUNT

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

        self.linear1 = nn.Linear(VARIABLE_COUNT, 64)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 32)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
