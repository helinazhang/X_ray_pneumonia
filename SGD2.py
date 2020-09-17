# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:24:12 2020

@author: Jinlong_Dell
"""

import os
import numpy as np
from tqdm import tqdm
import pandas as pd 

import cv2
import matplotlib.pyplot as plt
import skimage

import numpy as np


input_path = 'C:/Users/Jinlong_Dell/Documents/SMU/Sem2/Applied Machine Learning/Project/chest-xray-pneumonia/chest_xray/chest_xray/'
os.chdir(input_path)
train_dir = "../train/"
test_dir =  "../test/"
