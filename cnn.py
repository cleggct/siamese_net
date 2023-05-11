import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# this class defines a basic CNN for CIFAR10 images
# it takes a 32 by 32 3-channel image as input and outputs
# a 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0, dilation=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, dilation=5)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
