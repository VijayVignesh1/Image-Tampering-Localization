from PIL import Image
import os
import h5py
from tqdm import tqdm
from scipy.misc import imread,imresize
import glob
import numpy as np
import random
import cv2
from dodge import img_dodge
img_path_train='train/train/cat*'
img_path_val='train/val/cat*'
all_files_train=glob.glob(img_path_train)
all_files_val=glob.glob(img_path_val)
j=0
with h5py.File(os.path.join('UNET_TRAIN.h5py'),'a') as h:
    imgs=h.create_dataset('images',(2500,3,256,256),dtype='uint8')
    masks=h.create_dataset('masks',(2500,256,256),dtype='uint8')
    for file_name in tqdm(all_files_train[:2500]):
        if j<=500:
            side='left_top'
        elif j>500 and j<=1000:
            side='right_bottom'
        elif j>1000 and j<=1500:
            side='right_top'
        elif j>1500 and j<=2000:
            side='left_bottom'
        else:
            side='center'
        img=cv2.imread(file_name)
        img_tamp,mask=img_dodge(img,side)
        img_tamp=img_tamp.transpose(2,0,1)
        assert img_tamp.shape==(3,256,256)
        assert np.max(img_tamp)<=255
        #print(mask.shape)
        assert mask.shape==(256,256)
        assert np.max(mask)<=255
        imgs[j]=img_tamp
        masks[j]=mask
        j+=1

j=0
with h5py.File(os.path.join('UNET_VAL.h5py'),'a') as h:
    imgs=h.create_dataset('images',(1000,3,256,256),dtype='uint8')
    masks=h.create_dataset('masks',(1000,256,256),dtype='uint8')
    for file_name in tqdm(all_files_val[:1000]):
        if j<=100:
            side='left_top'
        elif j>100 and j<=200:
            side='right_bottom'
        elif j>200 and j<=300:
            side='right_top'
        elif j>300 and j<=400:
            side='left_bottom'
        else:
            side='center'
        img=cv2.imread(file_name)
        img_tamp,mask=img_dodge(img)
        img_tamp=img_tamp.transpose(2,0,1)
        assert img_tamp.shape==(3,256,256)
        assert np.max(img_tamp)<=255
        assert mask.shape==(256,256)
        assert np.max(mask)<=255
        imgs[j]=img_tamp
        masks[j]=mask
        j+=1
