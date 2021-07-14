#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:10:16 2021

@author: newuser
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
import skvideo.io

cap = skvideo.io.vread('2021-03-26 09-42-03.mkv')

from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage.color import rgb2gray

image = cap[0]



gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray_img.shape

white_padding = np.zeros((50, width, 3))
white_padding[:, :] = [255, 255, 255]
rgb_img = np.row_stack((white_padding, image))

gray_img = 255 - gray_img
gray_img[gray_img > 100] = 255
gray_img[gray_img <= 100] = 0
black_padding = np.zeros((50, width))
gray_img = np.row_stack((black_padding, gray_img))

kernel = np.ones((30, 30), np.uint8)
closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
closing_copy = np.uint8(closing)
edges = cv2.Canny(closing_copy, 100, 200)

















