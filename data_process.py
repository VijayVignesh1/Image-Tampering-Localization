from PIL import Image
import os
import h5py
from tqdm import tqdm
from PIL import Image
import glob
import numpy as np
import random
import cv2
from dodge import img_dodge

img_path_train='train/train/cat*'
img_path_val='train/val/cat*'
all_files_train=glob.glob(img_path_train)
all_files_val=glob.glob(img_path_val)
count=0
# Iterate through images and tamper random portions of images
# and create the corresponding label as well for training.

with h5py.File(os.path.join('CAT_IMAGES.h5py'),'a') as h:
    images=h.create_dataset('images',(2*len(all_files_train),3,256,256),dtype='uint8')
    values=h.create_dataset('values',(2*len(all_files_train),2),dtype='uint8')
    for file_name in tqdm(all_files_train):
        #   print('Enter for..')
        if count<=200:
            side='left_top'
        elif count>200 and count<=400:
            side='right_bottom'
        elif count>400 and count<=600:
            side='right_top'
        elif count>600 and count<=800:
            side='left_bottom'
        else:
            side='center'
        img=Image.open(file_name)
        img=img.resize((256,256))
        img=np.array(img)[:,:,::-1] 
        img=img.transpose(2,0,1)
        assert img.shape==(3,256,256)
        assert np.max(img)<=255
        images[count]=img
        values[count]=[1,0]
        count+=1
        img=img.transpose(1,2,0)
        img_tamp,mask=img_dodge(img,side)
        img_tamp=img_tamp.transpose(2,0,1)
        img=img.transpose(2,0,1)
        assert img.shape==(3,256,256)
        images[count]=img_tamp
        values[count]=[0,1]
        count+=1

count=0
# Iterate through images and tamper random portions of images
# and create the corresponding label as well for validation.
with h5py.File(os.path.join('CAT_IMAGES_VAL.h5py'),'a') as h:
    images=h.create_dataset('images',(2*len(all_files_val),3,256,256),dtype='uint8')
    values=h.create_dataset('values',(2*len(all_files_val),2),dtype='uint8')
    for file_name in tqdm(all_files_val):
        #   print('Enter for..')
        if count<=200:
            side='left_top'
        elif count>200 and count<=400:
            side='right_bottom'
        elif count>400 and count<=600:
            side='right_top'
        elif count>600 and count<=800:
            side='left_bottom'
        else:
            side='center'
        img=Image.open(file_name)
        img=img.resize((256,256))
        img=np.array(img)[:,:,::-1] 
        img=img.transpose(2,0,1)
        assert img.shape==(3,256,256)
        assert np.max(img)<=255
        images[count]=img
        values[count]=[1,0]
        count+=1
        img=img.transpose(1,2,0)
        img_tamp,mask=img_dodge(img,side)
        img_tamp=img_tamp.transpose(2,0,1)
        img=img.transpose(2,0,1)
        assert img.shape==(3,256,256)
        images[count]=img_tamp
        values[count]=[0,1]
        count+=1
