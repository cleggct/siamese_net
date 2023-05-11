import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

# a custom data loader for our siamese network
# this data loader will provide us with three images at each step
# these will be an anchor image, a positive example (an image correlated
#   with the anchor image), and a negative example (an uncorrelated image)
# the output at each step is of the form (image, image, image)
class TripletData(Dataset):

    # train - specifies whether to use the training set or the test set
    # download - specifies whether to try downloading the data set if it
    #    is not available locally

    def __init__(self, train, download=False):
        super().__init__()

        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.data = torchvision.datasets.CIFAR10(root='./data', train=train,
                                        download=download, transform=transform)
                                        
        self.dropout = nn.Dropout(p=0.4)
    
    def __len__(self):
        return len(self.data)
    
    # returns a tuple of the form (image, image, image)
    def __getitem__(self, index):
    
        # get the anchor image using the current index
        a, lab = self.data[index]
        a = a.float()
            
        # the positive example will be constructed by
        #    randomly setting intensity values in the first image
        #    to zero.
        # to do this, we will simply pass the anchor through a dropout layer
        p = self.dropout(a)
        
        # the negative example will be constructed by grabbing a random image
        #    from the dataset
        rand_idx = random.randint(0, self.__len__()-1)
        n, lab = self.data[rand_idx]
        n = n.float()
    
        return (a, p, n)
