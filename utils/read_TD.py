#!/usr/bin/evn python
# -*- coding: utf-8 -*-

# from bdgod.dog_config import *
import cv2
import pickle

Image_Path = '/mnt/git/data/all_img'
def read_txt():
    all_pic_infs = {}
    inf = open('all_det.txt', 'rb')
    lines = inf.readlines()  # 读取全部内容
    for line in lines:
        line = line.strip('\r\n')
        key_value = line.split(' ')
        raw_path = key_value[0]
        new_path = raw_path.replace('/home/meteo/zihao.chen/filter_ext/img_data',Image_Path)
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
                temp_list.append(a_points)
        if not all_pic_infs.has_key(new_path):
            all_pic_infs[new_path]=temp_list
        all_pic_infs[new_path] = temp_list
        print new_path ,' ',temp_list
    file_d = open('all_pic_infs.pkl','wb')

    pickle.dump(all_pic_infs,file_d)
    file_d.close()



read_txt()
