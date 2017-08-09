#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Fc_Net(nn.Module):
    def __init__(self,input_fetures,num_classes=100):
        super(Fc_Net, self).__init__()
        self.input_fetures = input_fetures
        self.training = True
        self.fc1 = nn.Linear(input_fetures, 512)
        self.rl1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(512, 256)
        self.rl2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(256, num_classes)
        # self.model = nn.Sequential(
        #     nn.Linear(self.input_fetures, num_classes)
        # )
    def forward(self,x):
        x = self.fc1(x)
        x = self.rl1(x)
        if self.training:
            x = self.dropout1(x)
        x = self.fc2(x)
        x = self.rl2(x)
        if self.training:
            x = self.dropout2(x)
        x = self.fc3(x)
        return x
        # x = x.view(x.size(0), self.input_fetures)
        # out = self.model(x)
        #
        # return out