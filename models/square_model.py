import torch.nn as nn

from layers import SquareLayer

class SquareModel(nn.Module):

    def __init__(self, variable_count: int):
        super(SquareModel, self).__init__()

        self.linear1 = nn.Linear(variable_count, 64)
        self.activation1 = nn.SiLU()
        self.square_layer1 = SquareLayer()
        self.linear2 = nn.Linear(128, 128)
        self.activation2 = nn.SiLU()
        self.linear3 = nn.Linear(128, 32)
        self.activation3 = nn.SiLU()
        self.square_layer2 = SquareLayer()
        self.linear4 = nn.Linear(64, 16)
        self.activation4 = nn.SiLU()
        self.square_layer3 = SquareLayer()
        self.linear5 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.square_layer1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.square_layer2(x)
        x = self.linear4(x)
        x = self.activation4(x)
        x = self.square_layer3(x)
        x = self.linear5(x)
        return x
