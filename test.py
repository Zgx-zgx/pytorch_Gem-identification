from torch import nn
import  torch

import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=87):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(8192 , 1024)
        self.fc2 = nn.Linear(1024, 87)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        print(x.shape)
        x=x.flatten(1)
        x = self.fc(x)
        x=self.fc2(x)
        return x


if __name__ == '__main__':
    model=SimpleCNN()
    x= torch.randn((1,3, 128, 128))
    out=model(x)
    print(out.shape)