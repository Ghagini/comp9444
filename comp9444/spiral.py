# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.fc1=nn.Sequential(nn.Linear(2,num_hid))
        self.fc2=nn.Tanh()
        self.fc3=nn.Linear(num_hid,1)
        self.fc4=nn.Sigmoid()
        self.temp1=torch.zeros([97,1])
        # INSERT CODE HERE

    def forward(self, input):
        r=torch.sqrt(input[:,0]*input[:,0]+(input[:,1])*(input[:,1]))
        a=torch.atan2(input[:,1],input[:,0])
        # print("output",output)
        output=torch.stack([r,a],dim=1)
        output=self.fc1(output)
        self.temp1=self.fc2(output)
        output=self.fc3(self.temp1)
        output=self.fc4(output)
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.fc1=nn.Linear(2,num_hid)
        self.fc2=nn.Linear(num_hid,num_hid)
        self.fc3=nn.Linear(num_hid,1)
        self.temp1=torch.zeros([97,1])
        self.temp2=torch.zeros([97,1])
    def forward(self, input):
        output = 0*input[:,0] # CHANGE CODE HERE
        self.temp1=torch.tanh(self.fc1(input))
        self.temp2=torch.tanh(self.fc2(self.temp1))
        output=torch.sigmoid(self.fc3(self.temp2))
        return output

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        # INSERT CODE HERE
        self.fc1=nn.Linear(2,num_hid)
        self.fc2=nn.Linear(num_hid+2,num_hid)
        self.fc3=nn.Linear(2*num_hid+2,1)
        self.temp1=torch.zeros([97,1])
        self.temp2=torch.zeros([97,1])

    def forward(self, input):
        output = 0*input[:,0] # CHANGE CODE HERE
        self.temp1=torch.tanh(self.fc1(input))
        self.temp2=torch.tanh(self.fc2(torch.cat((self.temp1,input),dim=1)))
        output=torch.sigmoid(self.fc3(torch.cat((input,self.temp1,self.temp2),dim=1)))
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients

        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        net.train() # toggle batch norm, dropout back again
        # pred = (output >= 0.5).float()
        if layer==1:
            pred=(net.temp1[:,node]>=0.5).float()
        if layer==2:
            pred=(net.temp2[:,node]>=0.5).float()
        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')

    # INSERT CODE HERE
