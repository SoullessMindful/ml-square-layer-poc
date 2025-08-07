import torch.nn as nn

from layers import SquareLayer

class SquareModel(nn.Module):
    
    def __init__(self, variable_count: int):
        super(SquareModel, self).__init__()
        
        self.activation = nn.Tanh
        
        self.model = nn.Sequential(
            nn.Linear(variable_count, 32),
            self.activation(),
            SquareLayer(),
            nn.Linear(64, 32),
            self.activation(),
            nn.Linear(32, 32),
            self.activation(),
            SquareLayer(),
            nn.Linear(64, 32),
            self.activation(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        return self.model(x)
