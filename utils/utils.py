#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import os


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