# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:45:37 2025

@author: Vinoth
"""


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class Neutrosophic_set:
    """___ Change Log __
    In this version modifications are done to the following:
    * In-built resize option
    * Zero padding and one padding in a single class function
    * Membership function values are modified manually for zero and one images
    """
    
    def __init__(self, img_data, h, sz=None, z_pd=True):
        self.img_data = img_data
        self.h = h
        self.sz = sz
        self.z_pd = z_pd
    
    def resizeImg(self):
        """ Resize image to specified size (int or tuple) """
        if self.sz is not None:
            if isinstance(self.sz, int):
                self.img_data = cv.resize(self.img_data, (self.sz, self.sz))
            elif isinstance(self.sz, tuple) and len(self.sz) == 2:
                self.img_data = cv.resize(self.img_data, (self.sz[1], self.sz[0]))
            else:
                raise AttributeError("Size should be an integer or 2D tuple.")
        return self.img_data

    def pading_img(self):
        """ Perform padding with zeros or ones """
        resized_img = self.resizeImg()
        if self.h >= resized_img.shape[0] or self.h >= resized_img.shape[1]:
            raise ValueError("The block size should be less than the image size")
        
        if self.z_pd:
            return np.pad(resized_img, self.h // 2, mode='constant', constant_values=0)
        else:
            return np.pad(resized_img, self.h // 2, mode='constant', constant_values=1)

    def g_bar(self):
        """ Compute the mean of each neighborhood block """
        resized_img = self.resizeImg()
        return np.array([[np.mean(self.pading_img()[i:i + self.h, j:j + self.h])
                          for j in range(resized_img.shape[1])]
                         for i in range(resized_img.shape[0])], dtype=np.float64)

    def delta(self):
        """ Calculate the delta (absolute difference from g_bar) """
        return np.abs(self.resizeImg() - self.g_bar())

    def truth_mem(self):
        """ Compute the truth membership function """
        resized_img = self.resizeImg()
        if np.sum(resized_img) == 0:
            return np.zeros(resized_img.shape)
        if np.sum(resized_img) == np.prod(resized_img.shape):
            return np.ones(resized_img.shape)
        mn, mx = self.g_bar().min(), self.g_bar().max()
        tru_val = (self.img_data - mn) / (mx - mn)
        return np.clip(tru_val, 0, 1)

    def indeter_mem(self):
        """ Compute the indeterminate membership function """
        resized_img = self.resizeImg()
        if np.sum(resized_img) == 0:
            return self.truth_mem()
        elif np.sum(resized_img) == np.prod(resized_img.shape):
            return self.false_mem()
        mn, mx = self.delta().min(), self.delta().max()
        ind_val = (self.delta() - mn) / (mx - mn)
        return np.clip(ind_val, 0, 1)

    def false_mem(self):
        """ Compute the false membership function """
        return 1 - self.truth_mem()





class Image_stat:
    def __init__(self, img_data, h, sz=None, z_pd=True):
        self.img_data = img_data
        self.h = h
        self.sz = sz
        self.z_pd = z_pd

    def resizeImg(self):
        """ Resize image to specified size (int or tuple) """
        if self.sz is not None:
            if isinstance(self.sz, int):
                self.img_data = cv.resize(self.img_data, (self.sz, self.sz))
            elif isinstance(self.sz, tuple) and len(self.sz) == 2:
                self.img_data = cv.resize(self.img_data, (self.sz[1], self.sz[0]))
            else:
                raise AttributeError("Size should be an integer or 2D tuple.")
        return self.img_data

    def pading_img(self):
        """ Perform padding with zeros or ones """
        resized_img = self.resizeImg()
        if self.h >= resized_img.shape[0] or self.h >= resized_img.shape[1]:
            raise ValueError("The block size should be less than the image size")
        
        if self.z_pd:
            return np.pad(resized_img, self.h // 2, mode='constant', constant_values=0)
        else:
            return np.pad(resized_img, self.h // 2, mode='constant', constant_values=1)

    def img_mean(self):
        """ Calculate the mean of each neighborhood block """
        return self._calculate_stat(np.mean)

    def img_std(self):
        """ Calculate the standard deviation of each neighborhood block """
        return self._calculate_stat(np.std)

    def img_max(self):
        """ Calculate the max value of each neighborhood block """
        return self._calculate_stat(np.max)

    def img_min(self):
        """ Calculate the min value of each neighborhood block """
        return self._calculate_stat(np.min)

    def _calculate_stat(self, stat_func):
        """ Helper function to calculate statistics like mean, std, max, min """
        resized_img = self.resizeImg()
        stat_img = np.zeros(resized_img.shape)
        for i in range(resized_img.shape[0]):
            for j in range(resized_img.shape[1]):
                stat_img[i, j] = stat_func(self.pading_img()[i:i + self.h, j:j + self.h])
        return stat_img


if __name__ == "__main__":
    h = 3
    sz = 53
    x = cv.imread("./samples/lena.png", 0)
    kk = Neutrosophic_set(x, h, sz).false_mem()
    plt.imshow(kk, cmap="gray")
