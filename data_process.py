from PIL import Image
import os
import h5py
from tqdm import tqdm
from scipy.misc import imread,imresize
import glob
import numpy as np
import random
import cv2
from dodge import image_dodge
img_path_train='train/train/cat*'
img_path_val='train/val/cat*'
all_files_train=glob.glob(img_path_train)
all_files_val=glob.glob(img_path_val)
i=0
with h5py.File(os.path.join('CAT_IMAGES_VAL.h5py'),'a') as h:
    images=h.create_dataset('images',(2*len(all_files_val),3,256,256),dtype='uint8')
    values=h.create_dataset('values',(2*len(all_files_val),2),dtype='uint8')
    for file_name in tqdm(all_files_val):
        #   print('Enter for..')
        if i<=200:
            side='left_top'
        elif i>200 and i<=400:
            side='right_bottom'
        elif i>400 and i<=600:
            side='right_top'
        elif i>600 and i<=800:
            side='left_bottom'
        else:
            side='center'
        img=imread(file_name)
        img=imresize(img,(256,256))
        img=img.transpose(2,0,1)
        assert img.shape==(3,256,256)
        assert np.max(img)<=255
        images[i]=img
        values[i]=[1,0]
        i+=1
        img=img.transpose(1,2,0)
        img_tamp,mask=img_dodge(img,side)
        img_tamp=img_tamp.transpose(2,0,1)
        img=img.transpose(2,0,1)
        assert img.shape==(3,256,256)
        images[i]=img_tamp
        values[i]=[0,1]
        i+=1
