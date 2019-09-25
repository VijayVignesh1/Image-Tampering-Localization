import torch
from torch.utils.data import Dataset
import h5py
import os
from torchvision.transforms import transforms
class DatasetLoad(Dataset):
    def __init__(self,type=None):
        if type=='train':
            self.h=h5py.File('CAT_IMAGES.h5py','r')
        elif type=='val':
            self.h=h5py.File('CAT_IMAGES_VAL.h5py','r')
        else:
            print('Type not specified..')
            exit(0)
        self.imgs=self.h['images']
        self.values=self.h['values']
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        self.dataset_size=len(self.values)
    def __getitem__(self,i):
        #print('In train')
        img=torch.FloatTensor(self.imgs[i]//255.)
        img=self.normalize(img)
        value=torch.LongTensor(self.values[i])
        return img,value
    def __len__(self):
        return self.dataset_size
