#!/usr/bin/evn python
# -*- coding: utf-8 -*-

"""
author:  zihao.chen
date:  2017/07/08
descrption:
"""

from torchvision import transforms
from PIL import Image
from PIL import ImageChops
import numpy as np
import random
import os

# from dog_config import *

#
# def data_augmentation(raw_path):
#     new_path = raw_path.replace('img_data', 'img_augmen_data')
#
#     raw_img = Image.open(raw_path)
#     width = raw_img.size[0]
#     height = raw_img.size[1]
#     max_l = max(raw_img.size)
#     min_l = min(raw_img.size)
#
#     # 平移
#     py_1 = ImageChops.offset(raw_img, int(width * 0.2), int(height * 0.2))
#     py_2 = ImageChops.offset(raw_img, -int(width * 0.2), -int(height * 0.2))
#
#     # 旋转
#     xz_1 = raw_img.transpose(Image.ROTATE_90)
#     xz_2 = raw_img.transpose(Image.ROTATE_180)
#     xz_3 = raw_img.transpose(Image.ROTATE_270)
#
#     # 镜像
#     jx_1 = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
#     jx_2 = raw_img.transpose(Image.FLIP_TOP_BOTTOM)
#
#     # 仿射变换
#     params = (1 + random.random() * 0.2 - 0.1,
#               random.random() * 0.2 - 0.1,
#               random.random(),
#               random.random() * 0.3 - 0.15,
#               1 + random.random() * 0.2 - 0.1,
#               random.randint(0, 2) - 1)
#     fs_1 = raw_img.transform(raw_img.size, Image.AFFINE, params, Image.BILINEAR)
#     params = (1 + random.random() * 0.2 - 0.1,
#               random.random() * 0.2 - 0.1,
#               random.random(),
#               random.random() * 0.3 - 0.15,
#               1 + random.random() * 0.2 - 0.1,
#               random.randint(0, 2) - 1)
#     fs_2 = raw_img.transform(raw_img.size, Image.AFFINE, params, Image.BILINEAR)
#
#     # 随机切割
#     cc = transforms.CenterCrop(int(min_l * 0.8))
#     sq_1 = cc(raw_img)
#
#     rc = transforms.RandomCrop(int(min_l * 0.8))
#     sq_2 = rc(raw_img)
#     sq_3 = rc(raw_img)
#     sq_4 = rc(raw_img)
#
#     last_d = new_path.rfind('/')
#     file_dir = new_path[:last_d]
#     touch_dir(file_dir)
#     file_name = new_path[last_d + 1:-4]
#     all_imgs = [py_1, py_2, xz_1, xz_2, xz_3, jx_1, jx_2, fs_1, fs_2, sq_1, sq_2, sq_3, sq_4]
#     raw_img.save(new_path)
#     for i in range(len(all_imgs)):
#         img = all_imgs[i]
#         img.save(os.path.join(file_dir, file_name + '_%d.jpg' % i))
def data_augmentation_img_tag(raw_img, data_size=224,tag=0):
    # new_path = raw_path.replace('img_data', 'img_augmen_data')
    #
    # raw_img = Image.open(raw_path)
    raw_img = Image.fromarray(np.uint8(raw_img))
    width = raw_img.size[0]
    height = raw_img.size[1]
    max_l = max(raw_img.size)
    min_l = min(raw_img.size)

    # 平移
    py_1 = ImageChops.offset(raw_img, int(width * random.random() * 0.2), int(height * random.random() * 0.2))
    py_1 = py_1.resize((data_size, data_size))
    py_1 = np.asarray(py_1)
    py_2 = ImageChops.offset(raw_img, -int(width * random.random() * 0.2), -int(height * random.random() * 0.2))
    py_2 = py_2.resize((data_size, data_size))
    py_2 = np.asarray(py_2)
    if bool(random.getrandbits(1)):
        py = py_1
    else:
        py = py_2
    # 旋转
    xz_1 = raw_img.transpose(Image.ROTATE_90)
    xz_1 = xz_1.resize((data_size, data_size))
    xz_1 = np.asarray(xz_1)
    # xz_2 = raw_img.transpose(Image.ROTATE_180)
    xz_2 = raw_img.transpose(Image.ROTATE_270)
    xz_2 = xz_2.resize((data_size, data_size))
    xz_2 = np.asarray(xz_2)
    if bool(random.getrandbits(1)):
        xz = xz_1
    else:
        xz = xz_2
    # 镜像
    jx_1 = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
    jx_1 = jx_1.resize((data_size, data_size))
    jx_1 = np.asarray(jx_1)
    jx_2 = raw_img.transpose(Image.FLIP_TOP_BOTTOM)
    jx_2 = jx_2.resize((data_size, data_size))
    jx_2 = np.asarray(jx_2)
    if bool(random.getrandbits(1)):
        jx = jx_1
    else:
        jx = jx_2
    # 仿射变换
    params = (1 + random.random() * 0.2 - 0.1,
              random.random() * 0.2 - 0.1,
              random.random(),
              random.random() * 0.3 - 0.15,
              1 + random.random() * 0.2 - 0.1,
              random.randint(0, 2) - 1)
    fs_1 = raw_img.transform(raw_img.size, Image.AFFINE, params, Image.BILINEAR)
    fs_1 = fs_1.resize((data_size, data_size))
    fs_1 = np.asarray(fs_1)
    params = (1 + random.random() * 0.2 - 0.1,
              random.random() * 0.2 - 0.1,
              random.random(),
              random.random() * 0.3 - 0.15,
              1 + random.random() * 0.2 - 0.1,
              random.randint(0, 2) - 1)
    fs_2 = raw_img.transform(raw_img.size, Image.AFFINE, params, Image.BILINEAR)
    fs_2 = fs_2.resize((data_size, data_size))
    fs_2 = np.asarray(fs_2)
    if bool(random.getrandbits(1)):
        fs = fs_1
    else:
        fs = fs_2
    # 随机切割
    cc = transforms.CenterCrop(int(min_l * 0.9))
    sq_1 = cc(raw_img)
    sq_1 = sq_1.resize((data_size, data_size))
    sq_1 = np.asarray(sq_1)
    rc = transforms.RandomCrop(int(min_l * 0.9))
    sq_2 = rc(raw_img)
    sq_2 = sq_2.resize((data_size, data_size))
    sq_2 = np.asarray(sq_2)
    sq_3 = rc(raw_img)
    sq_3 = sq_3.resize((data_size, data_size))
    sq_3 = np.asarray(sq_3)
    sq_4 = rc(raw_img)
    sq_4 = sq_4.resize((data_size, data_size))
    sq_4 = np.asarray(sq_4)
    raw_img = raw_img.resize((data_size, data_size))
    raw_img = np.asarray(raw_img)
    # all_imgs = [raw_img, py, xz, jx, fs, sq_1, sq_2, sq_3]
    if tag ==0:
        all_imgs = [raw_img, py_1, py_2,xz_1,xz_2]
    elif tag ==1:
        all_imgs = [raw_img, xz_1, xz_2,jx_1,jx_2]
    elif tag == 2:
        all_imgs = [raw_img, jx_1,jx_2, fs_1,fs_2]
    elif tag ==3 :
        all_imgs = [raw_img, fs_1, fs_2 ,sq_1,sq_2]
    elif tag ==4 :
        all_imgs = [raw_img,sq_3, sq_4 ,sq_1,py]
    random.shuffle(all_imgs)
    all_imgs = np.array(all_imgs)
    # all_imgs = all_imgs.transpose(0, 3, 1, 2)

    return all_imgs


def data_augmentation_img(raw_img, data_size=224):
    # new_path = raw_path.replace('img_data', 'img_augmen_data')
    #
    # raw_img = Image.open(raw_path)
    raw_img = Image.fromarray(np.uint8(raw_img))
    width = raw_img.size[0]
    height = raw_img.size[1]
    max_l = max(raw_img.size)
    min_l = min(raw_img.size)

    # 平移
    py_1 = ImageChops.offset(raw_img, int(width * random.random() * 0.2), int(height * random.random() * 0.2))
    py_1 = py_1.resize((data_size, data_size))
    py_1 = np.asarray(py_1)
    py_2 = ImageChops.offset(raw_img, -int(width * random.random() * 0.2), -int(height * random.random() * 0.2))
    py_2 = py_2.resize((data_size, data_size))
    py_2 = np.asarray(py_2)
    if bool(random.getrandbits(1)):
        py = py_1
    else:
        py = py_2
    # 旋转
    xz_1 = raw_img.transpose(Image.ROTATE_90)
    xz_1 = xz_1.resize((data_size, data_size))
    xz_1 = np.asarray(xz_1)
    # xz_2 = raw_img.transpose(Image.ROTATE_180)
    xz_2 = raw_img.transpose(Image.ROTATE_270)
    xz_2 = xz_2.resize((data_size, data_size))
    xz_2 = np.asarray(xz_2)
    if bool(random.getrandbits(1)):
        xz = xz_1
    else:
        xz = xz_2
    # 镜像
    jx_1 = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
    jx_1 = jx_1.resize((data_size, data_size))
    jx_1 = np.asarray(jx_1)
    jx_2 = raw_img.transpose(Image.FLIP_TOP_BOTTOM)
    jx_2 = jx_2.resize((data_size, data_size))
    jx_2 = np.asarray(jx_2)
    if bool(random.getrandbits(1)):
        jx = jx_1
    else:
        jx = jx_2
    # 仿射变换
    params = (1 + random.random() * 0.2 - 0.1,
              random.random() * 0.2 - 0.1,
              random.random(),
              random.random() * 0.3 - 0.15,
              1 + random.random() * 0.2 - 0.1,
              random.randint(0, 2) - 1)
    fs_1 = raw_img.transform(raw_img.size, Image.AFFINE, params, Image.BILINEAR)
    fs_1 = fs_1.resize((data_size, data_size))
    fs_1 = np.asarray(fs_1)
    params = (1 + random.random() * 0.2 - 0.1,
              random.random() * 0.2 - 0.1,
              random.random(),
              random.random() * 0.3 - 0.15,
              1 + random.random() * 0.2 - 0.1,
              random.randint(0, 2) - 1)
    fs_2 = raw_img.transform(raw_img.size, Image.AFFINE, params, Image.BILINEAR)
    fs_2 = fs_2.resize((data_size, data_size))
    fs_2 = np.asarray(fs_2)
    if bool(random.getrandbits(1)):
        fs = fs_1
    else:
        fs = fs_2
    # 随机切割
    cc = transforms.CenterCrop(int(min_l * 0.9))
    sq_1 = cc(raw_img)
    sq_1 = sq_1.resize((data_size, data_size))
    sq_1 = np.asarray(sq_1)
    rc = transforms.RandomCrop(int(min_l * 0.9))
    sq_2 = rc(raw_img)
    sq_2 = sq_2.resize((data_size, data_size))
    sq_2 = np.asarray(sq_2)
    sq_3 = rc(raw_img)
    sq_3 = sq_3.resize((data_size, data_size))
    sq_3 = np.asarray(sq_3)
    sq_4 = rc(raw_img)
    sq_4 = sq_4.resize((data_size, data_size))
    sq_4 = np.asarray(sq_4)
    raw_img = raw_img.resize((data_size, data_size))
    raw_img = np.asarray(raw_img)
    all_imgs = [raw_img, py, xz, jx, fs, sq_1, sq_2, sq_3]
    random.shuffle(all_imgs)
    all_imgs = np.array(all_imgs)
    # all_imgs = all_imgs.transpose(0, 3, 1, 2)

    return all_imgs


def touch_dir(path):
    result = False
    try:
        path = path.strip().rstrip("\\")
        if not os.path.exists(path):
            os.makedirs(path)
            result = True
        else:
            result = True
    except:
        result = False
    return result



# data_augmentation('/home/meteo/zihao.chen/filter_ext/img_data/1/9.jpg')
# load_dog_flist()
