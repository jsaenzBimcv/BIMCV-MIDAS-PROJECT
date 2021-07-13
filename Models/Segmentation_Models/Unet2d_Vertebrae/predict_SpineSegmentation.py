#! /usr/bin/python
# -*- coding: utf8 -*-
###############################################################################
# Description
###############################################################################
__author__ = "Jhon Jairo Sáenz Gamboa"
__copyright__ = ["MIDAS (Massive Image Data Anatomy Spine)", "PhD (Machine Learning in Magnetic Resonance. Describing the Pathophysiology of Lumbar Pain)"]
__credits__ = ["Julio Domenech; Antonio Alonso-Manjarrez","Jose Manuel Saborit-Torres","Joaquim Montell Serrano","Jon Ander Gómez-Adrian","Maria de la Iglesia-Vayá"]
__license__ = " "
__version__ = "0.0.1"
__email__ = "jsaenz@cipf.es"
__status__ = ["PhD Candidate, UPV Valencia-Spain, Computer Science","Development, Ceib, Valencia-Spain"]

# creation_date: 24/05/2018
#
# Last_modification: 25/07/2018
#
# Description:
"""
Spinal Image Segmentation Using the MIDAS Dataset
Use case: Vertebral semantic segmentation

The following code, semantic segment an MRI image (T2 weighted image) in two class: Background and Vertebrae

Args:
    nifty_path (str): the path to the T2 weighted image nifti file, .nii.gz extension
    model_path (str): the path to the model file, JSON-format data 
    weights-path (str): the path to the model weights file, .h5 extension
    save_path(str): the path to the save screenshots of the MRI image and segmentation predicted, if save-path is empty them screenshots will not be saved
Returns:
    segmentation (np.ndarray) :  segmented image, the shape [slice][width][height][class]

Model:
    The Model-Unet2d.json model is a modified version from U-Net, 
    see:    https://arxiv.org/pdf/1505.04597.pdf  
            https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    
Compatibility across versions
    tensorflow version: 1.14.0
    keras version: 2.2.4
    skimage version: 0.14.5

data augmentation used in training:
    rotation_range=25,
    zoom_range=[0.5 , 1.5],
    width_shift_range=0.15,
    height_shift_range=0.15, 
    shear_range=0.15, 
    horizontal_flip=True
"""

###############################################################################
# Imports
###############################################################################
import nibabel as nb
import numpy as np
from scipy import stats
from skimage.transform import resize, rotate
from keras.models import model_from_json
import matplotlib.pyplot as plt
import argparse

###############################################################################
# Variables
###############################################################################
prm = {'IMAGE_SIZE':(256, 256),
       'NUM_CLASS': 2, # background +  shapes
       'STORE_PATH':'./models/',
       'MODEL_TYPE':'Unet2d',
       #'BATCH_SIZE':26,
       #'MODE':'train',
       #'MAX_EPOCH':200,
       #'LR_DECAY': 0.0001, 
       #'PATIENCE':10,
       #'LOSS': 'SparseCategoricalCrossentropy', 
       #'METRICS':['accuracy'],
       #'LR': 33e-5,
       #'OPTIMIZER':keras.optimizers.RMSprop(lr = 33e-5),
       #'ACTIVATION' : 'softmax'
}

###############################################################################
# Function
###############################################################################
def parse_inputs():
    """"""
    parser = argparse.ArgumentParser(description='Label Spinal Region -MRI- ')
    #parser.add_argument('-r', '--root-path', dest='root_path', default='/')
    parser.add_argument(
        '-np',
        '--nifty-path',
        dest='nifty_path',
        default='',
        help = 'Image path'
    )
    parser.add_argument(
        '-mp',
        '--model-path',
        dest='model_path',
        default='./models/Model-Unet2d-32.json',
        help = 'Model path'
    )
    parser.add_argument(
        '-wp',
        '--weights-path',
        dest='weights_path',
        default='./models/Weights-Unet2d-32_RMSprop_lr-33e5_acc98.h5',
        help = 'Model weights'
    )
    parser.add_argument(
        '-sp',
        '--save-path',
        dest='save_path',
        default='',
        help='the path to the save screenshots of the MRI image and segmentation predicted, if save-path is empty them screenshots will not be saved'
    )
    parser.add_argument('-v', '--verbose', dest='verbose', type=int, default=0)

    return vars(parser.parse_args())


def norm(image):
    """
    z-score with mean 0 and standard deviation 1
    """
    image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()

def get_subject_images(subject_path):
    """
    load image from NIFTI file
    """
    img = nb.load(subject_path)
    return norm(np.array(img.get_data()))
    
def plot_screenshot(images, mask = True, gt=False, cols=4, smooth=True, ch=1, path_name = './', label_name='screenshot', save = True):
    """
    plot or save images / masks in a screenshot
    Parameters
    ----------
    images = images or mask, np.ndarray image data [slice][width][height][channel]
    mask =   if True, returns mask's screenshot
    smooth = if True, interpolation = 'spline16', if False = interpolation = 'nearest'
    path_name = save path, str
    label_name = filename, str
    save = if True, save images in path_ + label_name + .png
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
            if not(gt):
                img = np.argmax(images[i], axis=-1)
            else: 
                img = images[i]

                
                
            ax.imshow(rotate(img,90).squeeze(), cmap='jet')
            #ax.imshow(rotate((images[i,...,ch]/np.argmax(images[i,...,ch])),90).squeeze(), cmap='jet',
            #          interpolation=interpolation)
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
        fig.savefig(path_name+label_name+'.png', format='png', bbox_inches='tight')#, dpi=600
        plt.close(fig)
    else: plt.show()
    


def get_segmentation(nifty_path, model_path, weights_path, save_path='', niftyGT_path=''):
    """
    inputs:
    nifty_path (str): the path to the T2 weighted image nifti file, .nii.gz extension
    model_path (str): the path to the model file, JSON-format data 
    weights-path (str): the path to the model weights file, .h5 extension
    
    return np. array [slice][width][height][class]
    """
    # load image from NIFTI file
    img = get_subject_images(nifty_path)
    size = prm['IMAGE_SIZE']+(np.shape(img)[2],)
    #print(size)
    img_resized = np.array([resize(
        img,size,
        preserve_range=True,
        mode='constant',
        anti_aliasing=True
    )][0])
    data_input=np.array(img_resized)[..., np.newaxis].transpose(2,0,1,3)
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # load models
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    #loaded_model.summary() 
    # make predictions
    segmentation = loaded_model.predict(data_input, verbose=0)
    
    if save_path:
        plot_screenshot(images=segmentation, cols=4,
                            mask = True, smooth=True,
                            path_name = save_path,
                            label_name=nifty_path.split('/')[-1].split('.')[0] + '_Screenshot-Vert',
                            save = True)
        plot_screenshot(images=data_input,cols=4, 
                        mask = False, smooth=False, 
                        path_name = save_path, 
                        label_name =nifty_path.split('/')[-1].split('.')[0] + '_Screenshot-T2w',
                        ch=0,
                        save = True)
        if niftyGT_path:
             
            label = np.array(nb.load(niftyGT_path).get_data()).astype(int) # read labels
            
            label_resized = np.array([resize(label,size,preserve_range=True, 
                                             order=0, mode='constant',
                                             anti_aliasing=False)][0])

            plot_screenshot(images=label_resized.transpose(2,0,1), cols=4,
                            mask = True, smooth=True, gt=True,
                            path_name = save_path,
                            label_name=nifty_path.split('/')[-1].split('.')[0] + '_Screenshot-Vert_manual',
                            save = True)
            
    
    
    return segmentation

def main():
    args = parse_inputs()
    segmentation = get_segmentation(args['nifty_path'],args['model_path'],args['weights_path'],args['save_path'])
    print(segmentation.shape)

###############################################################################
# Main
###############################################################################
if __name__ == '__main__':
    main()
