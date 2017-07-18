#!/usr/bin/evn python
# -*- coding: utf-8 -*-
"""
author:  zihao.chen
date:  2017/07/06
descrption:
"""
import cv2
import os
import numpy as np
import torch
from torch.autograd import Variable
from dog_config import *


def predict(model, x_val):
    x = Variable(x_val.cuda(), requires_grad=False)
    output = model.forward(x)
    return output.cpu().data.numpy().argmax(axis=1)


def main():
    image_files = os.listdir(Test_Image_Path)
    model = torch.load('models/resnet_model_pretrained_adam_46_9_3.pkl')
    X_data = []
    Y_Data = []
    dog_key = os.listdir(Image_Path)
    key_map = {dog_key[x]: x for x in range(100)}
    count = 1
    for file_name in image_files:
        image_path = os.path.join(Test_Image_Path, file_name)
        print image_path
        if os.path.exists(image_path):
            Y_Data.append(file_name)
            img = cv2.imread(image_path) * 1.0
            img = cv2.resize(img, (224, 224))
            img = img.transpose(2, 0, 1)
            X_data.append(img[None, ...])
            if count % 128 == 0:
                X_data_NP = np.concatenate(X_data, axis=0)
                print X_data_NP.shape
                X_data_NP[:, 0, ...] -= MEAN_VALUE[0]
                X_data_NP[:, 1, ...] -= MEAN_VALUE[1]
                X_data_NP[:, 2, ...] -= MEAN_VALUE[2]
                teX = torch.from_numpy(X_data_NP).float()
                predY = predict(model, teX)
                print predY.shape
                print predY
                assert len(predY) == len(Y_Data)
                for i in range(len(predY)):
                    for key, value in key_map.iteritems():
                        if value == predY[i]:
                            with open('predict_dog.txt', 'a') as f:
                                f.write('%s\t%s\n' % (key, Y_Data[i][:-4]))
                X_data = []
                Y_Data = []
            count += 1

    X_data_NP = np.concatenate(X_data, axis=0)
    print X_data_NP.shape
    X_data_NP[:, 0, ...] -= MEAN_VALUE[0]
    X_data_NP[:, 1, ...] -= MEAN_VALUE[1]
    X_data_NP[:, 2, ...] -= MEAN_VALUE[2]
    teX = torch.from_numpy(X_data_NP).float()
    predY = predict(model, teX)
    print predY.shape
    print predY
    assert len(predY) == len(Y_Data)
    for i in range(len(predY)):
        for key, value in key_map.iteritems():
            if value == predY[i]:
                with open('predict_dog.txt', 'a') as f:
                    f.write('%s\t%s\n' % (key, Y_Data[i][:-4]))


if __name__ == '__main__':
    main()
