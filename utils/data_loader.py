#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import pickle
import random
import cv2
import os
import numpy as np
from bdgod.dog_config import *
from bdgod.data_augmentation import data_augmentation_img

class data_loader_(object):
    def __init__(self, batch_size, proportion=0.8, shuffle=True, data_add=4, onehot=True,data_size =224,nb_classes = 100):
        self.batch_szie = batch_size
        self.shuffle = shuffle
        self.all_pic_inf = self.load_keys()
        # print self.all_pic_inf
        self.all_data_length = len(self.all_pic_inf)
        self.proportion = proportion
        self.train_length = int(self.all_data_length * self.proportion)
        self.test_length = self.all_data_length - self.train_length
        self.train_data = self.all_pic_inf[:self.train_length]
        self.test_data = self.all_pic_inf[self.train_length:]
        self.train_index = 0
        self.test_index = 0
        self.data_add = data_add
        self.data_size = data_size
        self.nb_classes = nb_classes
        self.onehot = onehot
    def load_keys(self):
        file_d = open('../all_pic_infs.pkl', 'rb')
        pic_data = pickle.load(file_d)
        # print pic_data
        pic_data = pic_data.items()
        random.shuffle(pic_data)

        dog_key = os.listdir(Image_Path)
        self.key_map = {dog_key[x]: x for x in range(100)}
        file_d.close()
        # print pic_data
        return pic_data

    def add_train_index(self):
        self.train_index += 1
        if self.train_index == self.train_length:
            self.train_data = random.shuffle(self.train_data)
            self.train_index = 0

    def add_test_index(self):
        self.test_index += 1
        if self.test_index == self.test_length:
            self.test_data = random.shuffle(self.test_data)
            self.test_index =0

    def data_pop(self, train=True):
        if train:
            data_temp = self.train_data[self.train_index]
            self.add_train_index()
            return data_temp
        else:
            data_temp = self.test_data[self.test_index]
            self.add_test_index()
            return data_temp

    def get_train_data(self):
        result = np.ones((0,self.data_size,self.data_size,3))
        pop_num = self.batch_szie / self.data_add
        all_labels = []
        for i in range(pop_num):
            data_temp = self.data_pop(train=True)
            image_path = data_temp[0]
            label = image_path.split('/')[-2]
            label = self.key_map[label]
            labels = [int(label) for x in range(self.data_add)]
            all_labels +=labels
            image_points = data_temp[1]
            point_index = random.randint(1, len(image_points)) - 1
            point_value = image_points[point_index]
            img_temp = cv2.imread(image_path)
            img_temp_arr = np.array(img_temp)
            rad_width = (point_value[2] - point_value[0]) / 5
            rad_hight = (point_value[3] - point_value[1]) / 5
            point_value[0] = point_value[0] + random.randint(0, rad_width) - (rad_width / 2)
            if point_value[0]<0:
                point_value[0]=0
            point_value[2] = point_value[2] + random.randint(0, rad_width) - (rad_width / 2)
            point_value[1] = point_value[1] + random.randint(0, rad_hight) - (rad_hight / 2)
            if point_value[1]<0:
                point_value[1]=0
            point_value[3] = point_value[3] + random.randint(0, rad_hight) - (rad_hight / 2)
            corp_img = img_temp_arr[point_value[1]:point_value[3],point_value[0]:point_value[2]]
            all_imgs = data_augmentation_img(corp_img,data_size=self.data_size)
            all_imgs = all_imgs[:self.data_add]
            assert len(all_imgs) == self.data_add
            result = np.concatenate((result,all_imgs),axis=0)
        if self.onehot:
            targets = np.array(all_labels).reshape(-1)
            all_labels = np.eye(self.nb_classes)[targets]
        else:
            all_labels = np.array(all_labels)

        assert result.shape[0] == all_labels.shape[0]
        return result,all_labels

    def get_test_data(self):
        result = np.ones((0, self.data_size, self.data_size, 3))
        all_labels = []
        for i in range(self.batch_szie):
            data_temp = self.data_pop(train=False)
            image_path = data_temp[0]
            label = image_path.split('/')[-2]
            label = self.key_map[label]
            labels = [int(label)]
            all_labels += labels
            img_temp = cv2.imread(image_path)
            img_temp = cv2.resize(img_temp,(self.data_size,self.data_size))
            img_temp_arr = np.array([img_temp])

            result = np.concatenate((result, img_temp_arr), axis=0)
        if self.onehot:
            targets = np.array(all_labels).reshape(-1)
            all_labels = np.eye(self.nb_classes)[targets]
        else:
            all_labels = np.array(all_labels)

        assert result.shape[0] == all_labels.shape[0]
        return result, all_labels


dl = data_loader(batch_size=64,proportion=0.85,shuffle=True,data_add=4,onehot=True,data_size=448,nb_classes=100)
# for i in range(100):
#     X_data,Y_data = dl.get_train_data()
#     print X_data.shape
#     print Y_data.shape
#     X_data,Y_data = dl.get_test_data()
#     print X_data.shape
#     print Y_data.shape

X_data,Y_data = dl.get_train_data()
print X_data[0]
print Y_data[0]