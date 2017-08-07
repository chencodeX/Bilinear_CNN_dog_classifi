#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Fc_Net(nn.Module):
    def __init__(self,input_fetures,num_classes=100):
        super(Fc_Net, self).__init__()
        self.fc1 = nn.Linear(input_fetures, 2048)
        self.rl1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(2048, 2048)
        self.rl2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x