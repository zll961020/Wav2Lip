#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/03/05 14:46:19
@Author  :   zhanglingling 
@Version :   1.0
@Contact :   None
@License :   None
@Desc    :   None
'''

# here put the import lib
import logging
import os.path
import traceback
from logging import handlers
from functools import wraps
import time
from moviepy.editor import *
import glob 
import cv2 
import platform 
import subprocess
import re
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper