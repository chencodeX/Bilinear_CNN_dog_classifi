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
from utils.cv_data_loder import data_loader_
from dog_config import *

all_models = ['b1_0_3.pkl','b1_1_2.pkl','b1_2_0.pkl','b1_3_1.pkl','b1_4_0.pkl','b2_0_1.pkl',
              'b2_1_0.pkl','b2_2_1.pkl','b2_3_2.pkl','b2_4_0.pkl']

all_img_lab = {}
all_img_cv_lab = {}
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

def do_all(index):
    image_files = os.listdir(Test_Image_Path)
    model = torch.load('models/'+all_models[index])
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
                            if not all_img_lab.has_key(Y_Data[i][:-4]):
                                all_img_lab[Y_Data[i][:-4]]=[]
                            all_img_lab[Y_Data[i][:-4]].append(int(key))
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
                if not all_img_lab.has_key(Y_Data[i][:-4]):
                    all_img_lab[Y_Data[i][:-4]] = []
                all_img_lab[Y_Data[i][:-4]].append(int(key))

def do_all_cv(index):
    img_index = 0
    batch_size = 128
    data_l = data_loader_(batch_size=batch_size, band_num=1, tag_id=1, shuffle=True, data_add=2, onehot=False,
                          data_size=299, nb_classes=100)
    num_batches_test = data_l.test_length / batch_size
    dog_key = os.listdir(Image_Path)
    key_map = {dog_key[x]: x for x in range(100)}
    model = torch.load('models/' + all_models[index])
    for j in range(num_batches_test):
        teX, teY = data_l.get_test_data()
        teX = teX.transpose(0, 3, 1, 2)
        teX[:, 0, ...] -= MEAN_VALUE[0]
        teX[:, 1, ...] -= MEAN_VALUE[1]
        teX[:, 2, ...] -= MEAN_VALUE[2]
        teX = torch.from_numpy(teX).float()
        # teY = torch.from_numpy(teY).long()
        predY = predict(model, teX)
        for i in range(len(predY)):
            for key, value in key_map.iteritems():
                if value == predY[i]:
                    if not all_img_lab.has_key(img_index):
                        all_img_lab[img_index] = []
                    all_img_lab[img_index].append(int(key))
                    if not all_img_cv_lab.has_key(img_index):
                        all_img_cv_lab[img_index] = teY[i]
                    all_img_cv_lab[img_index]=teY[i]




if __name__ == '__main__':
    # main()
    models = []
    for i in range(len(all_models)):
        model = torch.load('models/' + all_models[i])
        models.append(model)
    raw_input("raw_input: ")
    # for key,value in all_img_lab.iteritems():
    #     print key,value