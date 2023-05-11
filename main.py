import torch
import torchvision
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg
from torch.utils.data import DataLoader
from cnn import Net
from loader import TripletData

# set up the data loader
dataset = TripletData(train=True)
trainloader = DataLoader(dataset = dataset, batch_size=1)

# set up the gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loss function
criterion = nn.TripletMarginLoss(margin=0.5)

# set up the model
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# put the model on the gpu
net.to(device)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    i = 0
    
    # each datum consists of an anchor image, a positive image, and a negative image
    # the positive image is correlated with the anchor image
    # the negative image is not correlated with the anchor
    for a, p, n in trainloader:
        
        # put the data on the gpu
        a = a.to(device)
        p = p.to(device)
        n = n.to(device)
    

        # zero the parameter gradients
        optimizer.zero_grad()
        
        #compute the network outputs on the images
        out = net.forward(a)
        pos = net.forward(p)
        neg = net.forward(n)
        
        #compute the loss and backpropagate
        loss = criterion(out, pos, neg)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
        
        i += 1

print('Finished Training')

PATH = './siamese_net.pth'
torch.save(net.state_dict(), PATH)