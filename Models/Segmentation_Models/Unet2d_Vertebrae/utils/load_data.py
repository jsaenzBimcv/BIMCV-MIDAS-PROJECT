import os
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pathlib
import numpy as np
import pandas as pd
import nibabel as nb

import random
from random import shuffle 
from skimage.transform import resize
from sklearn.model_selection import train_test_split

print('nibabel ' + nb.__version__)

### Loading Data and Creating Datasets
def norm(image):
    # z-score with mean 0 and standard deviation 1
    image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()

def load_nifti(filename, with_affine=False):
    """
    load image from NIFTI file
    Parameters
    ----------
    filename : str
        filename of NIFTI file
    with_affine : bool
        if True, returns affine parameters

    Returns
    -------
    data : np.ndarray image data 
           e.g. (512, 512, 12)
           [width][height][slice]
    """
    img = nb.load(filename)
    data = img.get_fdata()
    data = np.copy(data, order="C")
    
    if with_affine:
        return data, img.affine, img.header
    return data

def load_nifti_label(filename, with_affine=False):
    """
    load image from NIFTI file
    Parameters
    ----------
    filename : str
        filename of NIFTI file
    with_affine : bool
        if True, returns affine parameters

    Returns
    -------
    data : np.ndarray image data 
           e.g. (12, 512, 512, 1)
           [slice][width][height][channel]
    """
    #print(filename)
    img = nb.load(filename).get_fdata()
    data=[np.resize(np.array(img[:,:,i]).astype('float32'),(img.shape[0],img.shape[1],1)) for i in range(0,img.shape[2])]

    if with_affine:
        return data, img.affine
    return data 

def get_subject_images(subject_path, verbose=False):
    img_data, img_affine, img_header = load_nifti(subject_path, with_affine=True)
    img = np.array(img_data)
    size = np.shape(img)
    print(size)
    return img

def path_image_generator(images_path, labels_path, bs, mode="train", aug=None, shuffle_=False, image_size=(256,256)):
    x_list = [i for i in range(len(images_path))]
    if shuffle_:
        shuffle(x_list)
    idx = 0

    images_saved = np.zeros(image_size + (0,))
    labels_saved = np.zeros(image_size + (0,))
    while True:
        # initialize our batches of images and labels
        images = []
        labels = []
        # keep looping until we reach our batch size
        while len(images) < bs:            
            if idx == len(x_list): 
                
                # reset the file pointer to the beginning of the paths list
                # and re-read the line
                idx = 0
                #if we are evaluating we should now break from our
                # loop to ensure we don't continue to fill up the
                # batch from samples at the beginning of the file
                #if mode == "eval":
                #    break
            image_file = images_path[x_list[idx]]    
            label_file = labels_path[x_list[idx]]

            # attempt to read the next line of the CSV file
            image = np.array(norm(load_nifti(image_file, with_affine=False))).astype(np.float32) # read imag
            label = np.array(load_nifti(label_file , with_affine=False)).astype(int)  # read labels
            
            size = list(image_size +(np.shape(image)[2],))          
            image_resized = np.array([resize(image,size,preserve_range=True, mode='constant',anti_aliasing=True)][0])
            label_resized = np.array([resize(label,size,preserve_range=True, order=0, mode='constant',anti_aliasing=False)][0])
            
            # verify that no images of previous patients exist
            # if they exist, add them to the current patient's images
            if images_saved.shape[2] > 0:
                image_resized = np.concatenate((image_resized,images_saved),axis=2)
                label_resized = np.concatenate((label_resized,labels_saved),axis=2)
                images_saved = np.zeros(image_size + (0,))
                labels_saved = np.zeros(image_size + (0,))
                
                
            # if the number of images is greater than the batch size, separate the surplus for the next iteration l  
            if image_resized.shape[2] + len(images) >= bs:
                n_img = bs - len(images)
                images_saved = image_resized[...,n_img:]
                labels_saved = label_resized[...,n_img:]
                image_resized = image_resized[...,0:n_img] # [width][height][slice]
                label_resized = label_resized[...,0:n_img]
            # ----------task- labels----------------
            label = np.zeros_like(label_resized)
            label[label_resized >0] = 1
            
            # --------------------------            
            for i in range(image_resized.shape[2]):
                images.append(image_resized[:,:,i])
                labels.append(label[:,:,i])
            idx += 1
       
        labels = np.array(labels)[:,:,:,np.newaxis].astype(np.int)
        images = np.array(images)[:,:,:,np.newaxis].astype(np.float32)
        labels = to_categorical(labels)
        
        if aug is not None:
            seed = random.randint(1, 1000)
            image_datagen = ImageDataGenerator(**aug)
            mask_datagen = ImageDataGenerator(**aug)
            
            image_datagen.fit(np.array(images), augment=True, seed=seed)
            mask_datagen.fit(np.array(labels), augment=True, seed=seed)

            images = next(image_datagen.flow(np.array(images), batch_size=bs, seed=seed))
            labels = next(mask_datagen.flow(np.array(labels), batch_size=bs, seed=seed))
        #print(np.shape(images))
        #print(np.shape(labels))
        yield (np.array(images), np.array(labels))

def get_data_paths(prm):
    from collections import defaultdict         


    # Load the path list with the subjects to be used for training and testing.
    # The list should have the following columns: 
    # |subject|session|filepath|aut_seg|manual_seg|test-20|test-25|test-30|

    data = pd.read_csv(prm['PARTITION_PATH'], sep='\t', index_col=None)

    # split the dataset in train, test and validation
    train_data = data[data['test-'+str(int(prm['TEST_SIZE']*100))] == True]
    test_data = data[data['test-'+str(int(prm['TEST_SIZE']*100))] == False]

    # get the RMI paths
    root = pathlib.Path(prm['DATA_ROOT_PATH'])
    X_train = [root.joinpath(*[row['filepath']]) for index, row in train_data.iterrows()]
    y_train = [root.joinpath(*[row['manual_seg']]) for index, row in train_data.iterrows()]

    X_test = [root.joinpath(*[row['filepath']]) for index, row in test_data.iterrows()]
    y_test = [root.joinpath(*[row['manual_seg']]) for index, row in test_data.iterrows()]

    # The validation partition is derived from the training partition.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size=prm['VALIDATION_SIZE'], 
                                                      random_state=2021)

    # Number of images per partition 
    sizes = defaultdict(lambda:defaultdict(lambda: 0 ))
    label = ['X_train', 'X_test', 'X_val']
    slices_ = []
    for n, data in enumerate([X_train, X_test, X_val]):
        n_slices=0
        for image_file in data:
            # read imag, read labels
            image = np.array(norm(load_nifti(image_file, with_affine=False))).astype(np.float32)
            n_slices += np.shape(image)[2]
            sizes[label[n]][str(np.shape(image))] += np.shape(image)[2]
        slices_.append(n_slices)
    n_slices = [np.sum(list(n.values())) for n in sizes.values()]

    print('     Number of patients\t\t2D-Images')
    print('Train:\t\t\t%d\t%d  \nTest:\t\t\t%d\t%d  \nValidation:\t\t%d\t%d'
          %(len(X_train),n_slices[0],len(X_val),n_slices[1],len(X_test),n_slices[2])) 

    paths ={'X_train' : X_train,        'y_train' : y_train,
            'X_test' : X_test,        'y_test' : y_test,
            'X_val' : X_val,        'y_val' : y_val,
            'train_size': n_slices[0],        
            'test_size': n_slices[1],
            'val_size': n_slices[2],
               }
    return paths