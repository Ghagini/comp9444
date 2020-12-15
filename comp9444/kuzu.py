# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        # print("net lin")
        self.fc1 = nn.Linear(784, 10)


    def forward(self, x):
        x = x.view(x.shape[0], -1)  # make sure inputs are flattened
        #x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc1(x), dim=1)
        return x # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()

        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
        # INSERT CODE HERE

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x=torch.tanh(self.fc1(x))
        x=torch.tanh(self.fc2(x))
        x=F.log_softmax(x,dim=1)
        return x # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(50 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(x.shape)
        wi = x.size(2)
        if wi != 28:
            print("No 28")
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x# CHANGE CODE HERE
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

