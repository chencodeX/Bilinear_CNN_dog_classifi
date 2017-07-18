#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import os
import pickle
Test_Image_Path = '/home/meteo/zihao.chen/filter_ext/data/bddata/image'

def mian():
    file_names = os.listdir(Test_Image_Path)
    all_keys=[]
    for index in range(len(file_names)):
        file_name = file_names[index]
        key = file_name.split('.')[0]
        file_path = os.path.join(Test_Image_Path,file_name)
        all_keys.append(key)
        with open('TEST_FIlE/Test_%d.txt'%(index//1024), 'a') as f:
            f.write('%s 0\n'%file_path)
    f = open('all_keys.pkl','wb')
    pickle.dump(all_keys,f)
    f.close()

mian()