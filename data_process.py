from PIL import Image
import os
import h5py
from tqdm import tqdm
from scipy.misc import imread,imresize
import glob
import numpy as np
import random
import cv2
img_path_train='train/train/cat*'
img_path_val='train/val/cat*'
all_files_train=glob.glob(img_path_train)
all_files_val=glob.glob(img_path_val)
i=0

with h5py.File(os.path.join('CAT_IMAGES.h5py'),'a') as h:
    images=h.create_dataset('images',(2*len(all_files_train),3,256,256),dtype='uint8')
    values=h.create_dataset('values',(2*len(all_files_train),2),dtype='uint8')
    for file_name in tqdm(all_files_train):
        #   print('Enter for..')
        img=imread(file_name)
        img=imresize(img,(256,256))
        img=img.transpose(2,0,1)
        assert img.shape==(3,256,256)
        assert np.max(img)<=255
        images[i]=img
        values[i]=[1,0]
        i+=1
        #print('printing normal image')
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindow()

        img=img.transpose(1,2,0)
        x1=random.randint(0,50)
        x2=random.randint(170,250)
        y1=random.randint(0,50)
        y2=random.randint(170,250)
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0))
        img=img.transpose(2,0,1)
        assert img.shape==(3,256,256)
        images[i]=img
        values[i]=[0,1]
        i+=1
        #print("printing manipulated image..")
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindow()
        #print('breaking')

i=0
with h5py.File(os.path.join('CAT_IMAGES_VAL.h5py'),'a') as h:
    images=h.create_dataset('images',(2*len(all_files_val),3,256,256),dtype='uint8')
    values=h.create_dataset('values',(2*len(all_files_val),2),dtype='uint8')
    for file_name in tqdm(all_files_val):
        #   print('Enter for..')
        img=imread(file_name)
        img=imresize(img,(256,256))
        img=img.transpose(2,0,1)
        assert img.shape==(3,256,256)
        assert np.max(img)<=255
        images[i]=img
        values[i]=[1,0]
        i+=1
        #print('printing normal image')
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindow()

        img=img.transpose(1,2,0)
        x1=random.randint(0,50)
        x2=random.randint(170,250)
        y1=random.randint(0,50)
        y2=random.randint(170,250)
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0))
        img=img.transpose(2,0,1)
        assert img.shape==(3,256,256)
        images[i]=img
        values[i]=[0,1]
        i+=1
        #print("printing manipulated image..")
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindow()
        #print('breaking')
