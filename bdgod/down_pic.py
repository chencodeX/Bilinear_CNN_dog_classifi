#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
读取文件，下载图片
author:     zihao.chen
date:       2017/6/28
descrption: 此模块详细作用描述
"""
# import xlrd
# import urllib
import os
# import datetime
# import urllib2
import shutil
# import urllib
# import numpy as np
class DownloadPic(object):

    def __init__(self):
        self.document_path=''
        self.save_path = '/mnt/git/data/img_data'
        self.table_name = ''
        self.start_index = 0
        self.root_path = os.getcwd()


    def down_save_pic(self,task):
        print task
        pic_url = task[1]
        raw_image_path = os.path.join(self.save_path,pic_url+'.jpg')
        img_save_path = task[0]
        try:
            print raw_image_path,img_save_path
            shutil.copy(raw_image_path,img_save_path)
        except Exception as e:
            print 'error'
            print e.message




    def load_urls(self):
        root_dir = '/mnt/git/data/all_img'
        dog_keys = {}
        down_tasks = []
        inf = open('val.txt','rb')
        lines = inf.readlines()  # 读取全部内容
        for line in lines:
            line = line.strip('\r\n')
            key_value = line.split(' ')
            print key_value
            if not dog_keys.has_key(key_value[1]):
                dog_keys[key_value[1]]=0
            save_dir = key_value[1]
            save_dir = os.path.join(root_dir,save_dir)
            touch_dir(save_dir)
            save_path = os.path.join(save_dir,str(dog_keys[key_value[1]])+'.jpg')
            task = [save_path,key_value[0]]
            dog_keys[key_value[1]] +=1

            down_tasks.append(task)

        return down_tasks

    def main(self):
        tasks = self.load_urls()
        print '待下载任务数:',len(tasks)
        for i in range(len(tasks)):
            self.down_save_pic(tasks[i])

dp = DownloadPic()
dp.main()

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