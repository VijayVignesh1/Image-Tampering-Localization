import torch
from torch.utils.data import Dataset
import h5py
import os
from torchvision.transforms import transforms

class DatasetLoad(Dataset):
    def __init__(self,type=None):
        if type=='train':
            self.h=h5py.File('UNET_TRAIN.h5py','r')
        elif type=='val':
            self.h=h5py.File('UNET_VAL.h5py','r')
        else:
            print('Type not specified..')
            exit(0)            
        self.imgs=self.h['images']
        self.masks=self.h['masks']
        self.dataset_size=len(self.masks)
    def __getitem__(self,i):
        img=torch.FloatTensor(self.imgs[i]//255.)
        mask=torch.FloatTensor(self.masks[i]//255.)
        return img,mask
    def __len__(self):
        return self.dataset_size