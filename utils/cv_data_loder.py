#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import sys
sys.path.append("/mnt/git/Bilinear_CNN_dog_classifi-/")
import pickle
import random
import cv2
import os
import numpy as np
from bdgod.dog_config import *
from bdgod.data_augmentation import data_augmentation_img

class data_loader_(object):
    def __init__(self, batch_size, proportion=0.8, shuffle=True, data_add=4, onehot=True, data_size=224,
                 nb_classes=100):
