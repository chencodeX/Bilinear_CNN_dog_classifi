#!/usr/bin/evn python
# -*- coding: utf-8 -*-

"""
author:  zihao.chen
date:  2017/06/30
descrption:
"""

import os
import numpy as np
import glob
import cv2
import random

from bdgod.dog_config import *


def load_dog_flist():
    train_iamge_list = []
    test_image_list = []
    dog_key = os.listdir(Image_Path)
    key_map = {dog_key[x]: x for x in range(100)}
    for key in dog_key:
        key_image_dir = os.path.join(Image_Path, key)
        key_all_file = os.listdir(key_image_dir)
        key_all_file = [[os.path.join(key_image_dir, fn), key_map[key]] for fn in key_all_file]
        train_num = int(len(key_all_file) * 0.85)
        train_iamge_list += key_all_file[0:train_num]
        test_image_list += key_all_file[train_num:]

    random.shuffle(train_iamge_list)
    random.shuffle(test_image_list)

    return train_iamge_list, test_image_list


def load_dog_flist_addval():
    train_iamge_list = []
    test_image_list = []
    val_iamg_list = []
    dog_key = os.listdir(Image_Path)
    key_map = {dog_key[x]: x for x in range(100)}
    for key in dog_key:
        key_image_dir = os.path.join(Image_Path, key)
        key_all_file = os.listdir(key_image_dir)
        key_all_file = [[os.path.join(key_image_dir, fn), key_map[key]] for fn in key_all_file]
        train_num = int(len(key_all_file) * 0.7)
        test_num = int(len(key_all_file) * 0.9)
        train_iamge_list += key_all_file[0:train_num]
        test_image_list += key_all_file[train_num:test_num]
        val_iamg_list += key_all_file[test_num:]

    # random.shuffle(train_iamge_list)
    # random.shuffle(test_image_list)

    return train_iamge_list, test_image_list,val_iamg_list

def write_txt(file_name,image_list):
    for index in range(len(image_list)):
        with open(file_name+'_%d.txt'%(index//2048),'a') as f:
            f.write('%s %d\n'%(image_list[index][0],image_list[index][1]))


def load_image(file_list):
    X_data = []
    Y_data = []
    for item in file_list:
        try:
            image = cv2.imread(item[0])
            image = cv2.resize(image, (224, 224))
            image = image.transpose(2, 0, 1)
            X_data.append(image[None, ...])
            Y_data.append(int(item[1]))
        except Exception as e:
            print '==========error=========='
            print e
            print item
    X_data = np.concatenate(X_data, axis=0)
    Y_data = np.asarray(Y_data)
    return X_data, Y_data


def load_data():
    train_iamge_list, test_image_list = load_dog_flist()
    # train_data_X, train_data_Y = load_image(train_iamge_list)
    test_data_X,test_data_Y = load_image(test_image_list)
    # train_data_X=np.zeros((2,2))
    # train_data_Y=np.zeros((2,2))
    # np.save()
    # train_data_X = np.load('bddog/trX.npy')
    # train_data_Y = np.load('bddog/trY.npy')
    print test_data_X.shape
    print test_data_Y.shape
    print test_data_X.dtype
    print test_data_Y.dtype
    np.save('bddog/teX.npy',test_data_X)
    np.save('bddog/teY.npy',test_data_Y)
    # print train_data_X.mean(axis=3).mean(axis=2).mean(axis=0)
    # train_data_X[:, 0, ...] -= MEAN_VALUE[0]
    # train_data_X[:, 1, ...] -= MEAN_VALUE[1]
    # train_data_X[:, 2, ...] -= MEAN_VALUE[2]

    # test_data_X[:, 0, ...] -= MEAN_VALUE[0]
    # test_data_X[:, 1, ...] -= MEAN_VALUE[1]
    # test_data_X[:, 2, ...] -= MEAN_VALUE[2]
    # assert len(train_data_X) == len(train_data_Y)
    # count_10 = len(train_data_X) // 10
    # for i in range(10):
    #     np.save('bddog/trX%d.npy' % i, train_data_X[(i * count_10):((i + 1) * count_10)])
    #     np.save('bddog/trY%d.npy' % i, train_data_Y[(i * count_10):((i + 1) * count_10)])
        # print train_data_X.shape
        # print train_data_Y.shape
        # print train_data_X.dtype
        # print train_data_Y.dtype
        # print train_data_X.mean(axis=3).mean(axis=2).mean(axis=0)
        # print test_data_X.shape
        # print test_data_Y.shape
        # return train_data_X,test_data_X,train_data_Y,test_data_Y


if __name__ == '__main__':
    # train_image_list, test_image_list,val_image_list = load_data()
    # random.shuffle(train_image_list)
    # write_txt('train_data',train_image_list)
    # write_txt('test_data.txt',test_image_list)
    # write_txt('validation_data.txt',val_image_list)

    load_data()
    # trX,teX,trY,teY = load_data()
    # np.save('bddog/trX.npy',trX)
    # np.save('bddog/teX.npy',teX )
    # np.save('bddog/trY.npy',trY)
    # np.save('bddog/teY.npy',teY)

    # [ 110.34760864  126.37054721  134.38067511]
    # 3781
    #     [110.53614567  127.03544618  135.23870879]
    # 7490
    #     [110.8218385   127.67041653  135.41058521]
    # 7490
