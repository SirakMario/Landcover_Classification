"""

The landcover dataset from here: https://landcover.ai/
Version 1 dataset

Tasks 
1. Read large images and corresponding masks, divide them into smaller patches.
And write the patches as images to the local drive.  

2. Save only images and masks where masks have some decent amount of labels other than 0. 
Using blank images with label=0 is a waste of time and may bias the model towards 
unlabeled pixels. 

3. Divide the sorted dataset from above into train, test and validation datasets. 

4. You have to manually create and move some folders and rename appropriately and use 
ImageDataGenerator from keras. 

"""

import os
import cv2
import numpy as np
import glob

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
import random

#Understanding of the dataset
temp_img = cv2.imread("data/images/M-34-51-C-d-4-1.tif") #3 channels / spectral bands
plt.imshow(temp_img[:,:,2]) #View each channel...
temp_mask = cv2.imread("data/masks/M-34-51-C-d-4-1.tif") #3 channels but all same. 
labels, count = np.unique(temp_mask[:,:,0], return_counts=True) #Check for each channel. All chanels are identical
print("Labels are: ", labels, " and the counts are: ", count)

#Crop each large image into patches of 256x256. Save them into a directory 
root_directory = 'data/'

patch_size = 256

#Read images from 'images' subdirectory
#All images are of different size we have 2 options, resize or crop
#Some images are too large and some small. Resizing will change the size of real objects.
#Therefore, we will crop them to a nearest size divisible by 256 and then 
#divide all images into patches of 256x256x3. 
img_dir=root_directory+"images/"
for path, subdirs, files in os.walk(img_dir):
    dirname = path.split(os.path.sep)[-1]
    images = os.listdir(path)  #List of all image names in this subdirectory
    for i, image_name in enumerate(images):  
        if image_name.endswith(".tif"):
            image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
            SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
            SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
            image = Image.fromarray(image)
            image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
            image = np.array(image)             
   
            #Extract patches from each image
            print("Now patchifying image:", path+"/"+image_name)
            patches_img = patchify(image, (256, 256, 3), step=256)
    
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    
                    single_patch_img = patches_img[i,j,:,:]
                    single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                    
                    cv2.imwrite(root_directory+"256_patches/images/"+
                               image_name+"patch_"+str(i)+str(j)+".tif", single_patch_img)
            
  
 #Do the same as above for masks
mask_dir=root_directory+"masks/"
for path, subdirs, files in os.walk(mask_dir):
    dirname = path.split(os.path.sep)[-1]

    masks = os.listdir(path)  #List of all image names in this subdirectory
    for i, mask_name in enumerate(masks):  
        if mask_name.endswith(".tif"):           
            mask = cv2.imread(path+"/"+mask_name, 0)  #Read each image as Grey
            SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
            SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
            mask = Image.fromarray(mask)
            mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
            mask = np.array(mask)             
   
            #Extract patches from each image
            print("Now patchifying mask:", path+"/"+mask_name)
            patches_mask = patchify(mask, (256, 256), step=256)    
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    
                    single_patch_mask = patches_mask[i,j,:,:]
                    cv2.imwrite(root_directory+"256_patches/masks/"+
                               mask_name+"patch_"+str(i)+str(j)+".tif", single_patch_mask)


# Display the patched images with its corresponding masks
train_img_dir = "data/256_patches/images/"
train_mask_dir = "data/256_patches/masks/"

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(os.listdir(train_img_dir))


img_num = random.randint(0, num_images-1)

img_for_plot = cv2.imread(train_img_dir+img_list[img_num], 1)
img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)

mask_for_plot =cv2.imread(train_mask_dir+msk_list[img_num], 0)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_for_plot)
plt.title('Image')
plt.subplot(122)
plt.imshow(mask_for_plot, cmap='gray')
plt.title('Mask')
plt.show()

###########################################################################
# real information = if mask has decent amount of labels other than 0. 

useless=0
for img in range(len(img_list)): 
    img_name=img_list[img]
    mask_name = msk_list[img]
    print("Now preparing image and masks number: ", img)
      
    temp_image=cv2.imread(train_img_dir+img_list[img], 1)
   
    temp_mask=cv2.imread(train_mask_dir+msk_list[img], 0)
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if (1 - (counts[0]/counts.sum())) > 0.05:  #At least 5% useful area with labels that are not 0
        cv2.imwrite('data/256_patches/images_with_useful_info/images/'+img_name, temp_image)
        cv2.imwrite('data/256_patches/images_with_useful_info/masks/'+mask_name, temp_mask)
        
    else:
        useless +=1

print("Total useful images are: ", len(img_list)-useless)
print("Total useless images are: ", useless)
###############################################################
#Split the data into training, validation and testing. 

"""
Code for splitting folder into train, test, and val.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators. 
"""
import splitfolders

input_folder = 'data/256_patches/images_with_useful_info/'
output_folder = 'data/data_for_training_testing_val/'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.80, 0.10,0.10), group_prefix=None)
########################################
#Manually move folders around to bring them to the following structure.
"""
Your current directory structure:
Data/
    data_for_training_testing_val/train/
        images/
            img1, img2, ...
        masks/
            msk1, msk2, ....
            
    data_for_training_testing_val/test/
        images/
            img1, img2, ...
        masks/
            msk1, msk2, ....
            
    data_for_training_testing_val/val/
        images/
            img1, img2, ...
        masks/
            msk1, msk2, ....
        
Copy the folders around to the following structure... 


Data/
    data_for_keras_aug/train_images/
                train/
                    img1, img2, img3, ......
    
    data_for_keras_aug/train_masks/
                train/
                    msk1, msk, msk3, ......
    
    data_for_keras_aug/test_images/
                test/
                    img1, img2, img3, ......                

    data_for_keras_aug/test_masks/
                test/
                    msk1, msk, msk3, ......
    
    data_for_keras_aug/val_images/
                val/
                    img1, img2, img3, ......                

    data_for_keras_aug/val_masks/
                val/
                    msk1, msk, msk3, ......
      
                    
"""