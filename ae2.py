import torch
import torch.nn as nn

# AE that takes linear input and encode it linearly
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder2  =  nn.Sequential(
            nn.Linear(2902*3, 500),
            nn.Tanh(),
            nn.Linear(500, 20),
            nn.Tanh(),
            nn.Linear(20, 3)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(3,20),
            nn.Tanh(),
            nn.Linear(20, 500),
            nn.Tanh(),
            nn.Linear(500, 2902*3),
        )

    def forward(self, x):
        
        encoded = self.encoder2(x)
        decoded = self.decoder2(encoded)
        return encoded,decoded