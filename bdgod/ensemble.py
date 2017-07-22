#!/usr/bin/evn python
# -*- coding: utf-8 -*-


import torch
from torch.autograd import Variable

from torch.utils import model_zoo
from dog_config import *
model = torch.load('models/better1.pkl')
print model
for p in model.parameters():
    print p.size()

state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', Model_Root)
for name, param in state_dict.items():
    print name
    print param.size()