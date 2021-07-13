#! /usr/bin/python
# -*- coding: utf8 -*-
from __future__ import print_function
# Description: 
"""
 plot or save MRI images / masks in a screenshot 
-------
Types of sequences in the data = ['Sag_T2' 'Sag_T1' 'Sag_Stir']
"""
# Imports

import os, time
import glob
import random
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_learning_curve(results,path, save=True):

    fig, ax = plt.subplots(1,2, figsize=(16, 8)) # figsize=(8, 8)
    ax1, ax2 = ax.ravel()
    ax1.set_title("Learning curve\n Training and Validation - loss")
    ax1.plot(results.history["loss"][2:], label="loss")
    ax1.plot(results.history["val_loss"][2:], label="val_loss")
    ax1.plot( np.argmin(results.history["val_loss"][2:]),
             np.min(results.history["val_loss"][2:]), 
             marker="x", color="r", label="Best loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("loss")
    ax1.legend();
    ax2.set_title("Learning curve\n Training and Validation - m-IoU")
    ax2.plot(results.history["mean_iou"][2:], label="m-IoU")
    ax2.plot(results.history["val_mean_iou"][2:], label="val_m-IoU")
    ax2.plot( np.argmax(results.history["val_mean_iou"][2:]), 
             np.max(results.history["val_mean_iou"][2:]), 
             marker="x", color="r", label="Best m-IoU")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("m-IoU")
    ax2.legend(); 
    
    if save:
        plt.savefig(path+"Learning_curve.png", bbox_inches='tight', format='png')#, dpi=600
        plt.close(fig)
    else: plt.show()

def plot_screenshot(images, mask = True, cols=4, smooth=True, ch=1, path_name = './', label_name='screenshot', save = True):
    """
    plot or save images / masks in a screenshot
    Parameters
    ----------
    images = images or mask, np.ndarray image data [slice][width][height][channel]
    mask =   if True, returns mask's screenshot
    smooth = if True, interpolation = 'spline16', if False = interpolation = 'nearest'
    path_name = save path, str
    label_name = filename, str
    save = if True, save images in path_ + label_name + .jpg
    Returns
    -------
    plot or save images / masks in a screenshot
    """

    n = np.shape(images)[0]
    rows = np.rint(n/np.float(cols)).astype(int) 
    
    if (rows*cols)>n:
        cover = np.zeros((np.shape(images)[1],np.shape(images)[2]))    
        for i in range(rows*cols-n):
            images=np.dstack((np.array(images), np.array(cover)))
            

    # Create figure with sub-plots.
    fig, axes = plt.subplots(rows, cols, figsize=(12,10))

    # Adjust vertical spacing if we need to print ensemble and best-net.
    hspace = 0.03
    fig.subplots_adjust(hspace=hspace, wspace=0.03)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        if mask :    
            # Plot mask.
            ax.imshow(rotate((images[i,...,ch]/np.max(images[i,...,ch])),90).squeeze(), cmap='jet',
                      interpolation=interpolation)
            xlabel = "Mask: {0}".format(i+1)
            
        else:
            # Plot image.
            ax.imshow(rotate(images[i,...,ch],90).squeeze(), cmap='bone',
                  interpolation=interpolation)
            xlabel = "MRI_Scan: {0}".format(i+1)
            
            
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    
    if save:
        fig.savefig(path_name+label_name+'.jpg', format='jpg', bbox_inches='tight')#, dpi=600
        plt.close(fig)
    else: plt.show()
    
