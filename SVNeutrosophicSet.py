# -*- coding: utf-8 -*-
"""
Created on Sun May  4 10:34:42 2025

@author: Vinoth
"""
from NeutrosophicSet import *
class SV_Neutrosophic_set:
    def __init__(self, img_data, h, sz=None, z_pd=True):
        self.img_data = img_data
        self.h = h
        self.sz = sz
        self.z_pd = z_pd
        self.tru = Neutrosophic_set(self.img_data, self.h, self.sz, self.z_pd).truth_mem()
        self.ind = Neutrosophic_set(self.img_data, self.h, self.sz, self.z_pd).indeter_mem()
        self.fal = Neutrosophic_set(self.img_data, self.h, self.sz, self.z_pd).false_mem()
        self.r, self.c = self.tru.shape

    def truth_mem(self):
        """ Compute the truth membership function with clipping """
        return np.clip(self.tru, 0, 1)

    def indeter_mem(self):
        """ Compute the indeterminate membership function with clipping """
        return np.clip(self.ind, 0, 1)

    def false_mem(self):
        """ Compute the false membership function with clipping """
        return np.clip(self.fal, 0, 1)