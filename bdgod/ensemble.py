#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import sys

sys.path.append("/mnt/git/Bilinear_CNN_dog_classifi/")
import torch
import numpy as np
import cv2
from torch.autograd import Variable
from utils.cv_data_loder import data_loader_
import random
from dog_config import *

def predict(model, x_val):
    x = Variable(x_val.cuda(), requires_grad=False)
    output = model.forward(x)
    if type(output) == tuple:
        print output[1].size()
        return output[1].cpu().data.numpy()
    return output.cpu().data.numpy().argmax(axis=1)

def preprocess_input(x):
    x /= 255.
    # x -= 0.5
    # x *= 2.
    return x


def main():
    data_l = data_loader_(batch_size=64, band_num=1, tag_id=0, shuffle=False, data_add=4, onehot=False,
                 data_size=224,nb_classes=100)

    model = torch.load('models/resnet101_model_pretrained_SGD_16_498_1.pkl')
    model.training = False
    num_batches = data_l.test_length / data_l.batch_szie
    all_data = np.zeros((0,2048)).astype(np.float)
    all_lable = np.zeros((0))
    for j in range(num_batches + 1):
        teX, teY = data_l.get_test_data()
        # print teY.shape
        print all_lable.shape
        teX = teX.transpose(0, 3, 1, 2)
        teX[:, 0, ...] -= MEAN_VALUE[0]
        teX[:, 1, ...] -= MEAN_VALUE[1]
        teX[:, 2, ...] -= MEAN_VALUE[2]
        # teX = preprocess_input(teX)
        # teX = torch.from_numpy(teX).float()
        # futures = predict(model, teX)
        # print futures.shape
        # all_data = np.concatenate((all_data,futures),axis=0)
        all_lable = np.concatenate((all_lable,teY),axis=0)

    # print all_data.shape
    print all_lable.shape

    # all_data = all_data[:data_l.test_length]
    all_lable = all_lable[:data_l.test_length]
    print all_data.shape
    print all_lable.shape
    # np.save('future_densenet161.npy',all_data)
    np.save('lable_resnet101.npy', all_lable)

if __name__ == '__main__':
    main()