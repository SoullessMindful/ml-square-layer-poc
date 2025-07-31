import torch.nn as nn

from generate_data import VARIABLE_COUNT

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

        self.linear1 = nn.Linear(VARIABLE_COUNT, 128)
        self.activation1 = nn.SiLU()
        self.linear2 = nn.Linear(128, 128)
        self.activation2 = nn.SiLU()
        self.linear3 = nn.Linear(128, 64)
        self.activation3 = nn.SiLU()
        self.linear4 = nn.Linear(64, 32)
        self.activation4 = nn.SiLU()
        self.linear5 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.activation4(x)
        x = self.linear5(x)
        return x
