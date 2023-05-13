import torch
import torch.nn as nn

# AE that horizontally encode tensor
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size = (4, 4))
        # self.pool1 = nn.MaxPool2d((3, 1))

        # self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size = (4, 4))
        # self.pool2 = nn.MaxPool2d((3, 1))
        self.encoder1  =  nn.Sequential(
            nn.Linear(180, 45),
            nn.ReLU(),
            nn.Linear(45, 3),
        )
        self.decoder1  =  nn.Sequential(
            nn.Linear(3, 45),
            nn.ReLU(),
            nn.Linear(45, 180),
            nn.ReLU(),
        )

    def forward(self, x):
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        encoded = self.encoder1(x)

        decoded = self.decoder1(encoded)
        return encoded,decoded
