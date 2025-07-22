import torch

from generate_data import VARIABLE_COUNT

class BaseModel(torch.nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

        self.linear1 = torch.nn.Linear(VARIABLE_COUNT, 64)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, 32)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

base_model = BaseModel()
