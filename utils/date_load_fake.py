#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import sys
sys.path.append("/mnt/git/Bilinear_CNN_dog_classifi/")
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
        self.batch_szie = batch_size
        self.shuffle = shuffle
        # print self.all_pic_inf
        self.all_data_length = 20000
        self.proportion = proportion
        self.train_length = int(self.all_data_length * self.proportion)
        self.test_length = self.all_data_length - self.train_length
        self.data_add = data_add
        self.data_size = data_size
        self.nb_classes = nb_classes
        self.onehot = onehot


    def get_train_data(self):
        result = np.ones((0, self.data_size, self.data_size, 3))
        pop_num = self.batch_szie / self.data_add
        all_labels = []
        for i in range(pop_num):
            label = np.random.randint(100)
            all_imgs = np.ones((1,self.data_size, self.data_size, 3))
            all_imgs = (all_imgs*label*2)+15
            all_labels.append(label)
            result = np.concatenate((result, all_imgs), axis=0)
        if self.onehot:
            targets = np.array(all_labels).reshape(-1)
            all_labels = np.eye(self.nb_classes)[targets]
        else:
            all_labels = np.array(all_labels)

        assert result.shape[0] == all_labels.shape[0]
        return result, all_labels

    def get_test_data(self):
        result = np.ones((0, self.data_size, self.data_size, 3))
        all_labels = []
        for i in range(self.batch_szie):

            label = np.random.randint(100)
            all_imgs = np.ones((1,self.data_size, self.data_size, 3))
            all_imgs = (all_imgs*label*2)+15
            all_labels.append(label)
            result = np.concatenate((result, all_imgs), axis=0)

        if self.onehot:
            targets = np.array(all_labels).reshape(-1)
            all_labels = np.eye(self.nb_classes)[targets]
        else:
            all_labels = np.array(all_labels)

        assert result.shape[0] == all_labels.shape[0]
        return result, all_labels


if __name__ == '__main__':
    dl = data_loader_(batch_size=64, proportion=0.85, shuffle=True, data_add=4, onehot=True, data_size=448,
                      nb_classes=100)
    for i in range(3):
        # X_data,Y_data = dl.get_train_data()
        # for x in range(len(X_data)):
        #     cv2.imwrite('train_%s_%s.jpg'%(i,x),X_data[x])
        # print X_data.shape
        # print Y_data.shape
        X_data, Y_data = dl.get_test_data()
        # for x in range(len(X_data)):
        #     cv2.imwrite('test_%s_%s.jpg'%(i,x),X_data[x])
        print X_data.shape
        print Y_data.shape
        #
        # X_data,Y_data = dl.get_train_data()
        # print X_data[0]
        # print Y_data[0]
