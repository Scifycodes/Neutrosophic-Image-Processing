# -*- coding: utf-8 -*-
"""
Created on Sun May  4 07:00:00 2025

@author: Vinoth
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from NeutrosophicSet import *

class NS_plots(Neutrosophic_set):
    def __init__(self, img_data, h, sz=None, z_pd=True):
        super().__init__(img_data, h, sz)
        self.tru= self.truth_mem()
        self.ind = self.indeter_mem()
        self.fal = self.false_mem()
        pass
    def tru_viz(self, f_name):
        plt.imshow(self.tru, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        pass
    def ind_viz(self, f_name):
        plt.imshow(self.ind, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        pass
    def fal_viz(self, f_name):
        plt.imshow(self.fal, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        pass
    def ns_kde(self, f_name):
        """
        Input: float array
                t_mem: 
        """
        f_pixels = self.fal.flatten()  
        i_pixels = self.ind.flatten()  
        t_pixels = self.tru.flatten()  
        plt.figure(figsize=(10, 6))
        sns.kdeplot(t_pixels, bw_method=0.1, fill=True, color='green', label='Truth membership')
        sns.kdeplot(i_pixels, bw_method=0.1, fill=True, color='blue', label='Indeterminacy membership')
        sns.kdeplot(f_pixels, bw_method=0.1, fill=True, color='red', label='Falsity membership')
        plt.title('KDE plot for Neutrosophic set')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.show()
        pass
    def tru_intensity(self, f_name):
        x = np.arange(self.tru.shape[1])  
        y = np.arange(self.tru.shape[0])  
        X, Y = np.meshgrid(x, y)      
        Z = self.tru                      
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='Greens')  
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.show()
    def ind_intensity(self, f_name):
        x = np.arange(self.ind.shape[1])  
        y = np.arange(self.ind.shape[0])  
        X, Y = np.meshgrid(x, y)      
        Z = self.ind                      
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='Blues')  
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.show()
    def fal_intensity(self, f_name):
        x = np.arange(self.fal.shape[1])  
        y = np.arange(self.fal.shape[0])  
        X, Y = np.meshgrid(x, y)      
        Z = self.fal                      
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='Reds')  
        plt.savefig(f_name, dpi=300, bbox_inches='tight')
        plt.show()


# h = 3
# sz = 25
# x = cv.imread("./samples/lena.png", 0)
# ns = NS_plots(x, h, sz)
# ns.fal_intensity("cjdid")