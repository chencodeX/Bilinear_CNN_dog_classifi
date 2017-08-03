#!/usr/bin/evn python
# -*- coding: utf-8 -*-

# from bdgod.dog_config import *
import cv2
import pickle
import numpy as np

Image_Path = '/mnt/git/data/image_cp'
def read_txt():
    all_pic_infs = {}
    inf = open('detres_sense001_test.txt', 'rb')
    lines = inf.readlines()  # 读取全部内容
    for line in lines:
        line = line.strip('\r\n')
        key_value = line.split(' ')
        raw_path = key_value[0]
        new_path = raw_path.replace('/home/meteo/xibin.yue/data/dog_cls_data/test/test1',Image_Path)
        # print new_path
        temp_list = []
        key_value = line.split('.jpg ')
        data_str = key_value[1]
        data_str = data_str.lstrip('[')
        data_str = data_str.rstrip(']')
        if len(data_str) <4:
            img = cv2.imread(new_path)
            (h,w,x) = img.shape
            temp = [0,0,w,h]
            temp_list.append(temp)
        else:
            all_locas = data_str.split('], [')
            for locas in all_locas:
                a_points = locas.split(',')
                a_points = map(int,a_points)
                img_temp = cv2.imread(new_path)
                img_temp_arr = np.array(img_temp)
                if a_points[0] < 0:
                    a_points[0] = 0
                if a_points[1] < 0:
                    a_points[1] = 0
                corp_img = img_temp_arr[a_points[1]:a_points[3], a_points[0]:a_points[2]]
                cv2.imwrite(new_path,corp_img)
                temp_list.append(a_points)
                break

        if not all_pic_infs.has_key(new_path):
            all_pic_infs[new_path]=temp_list
        all_pic_infs[new_path] = temp_list
        print new_path ,' ',temp_list
    # file_d = open('all_test_pic_infs.pkl','wb')
    #
    # pickle.dump(all_pic_infs,file_d)
    # file_d.close()



read_txt()
