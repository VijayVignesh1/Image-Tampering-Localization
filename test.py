from torch import nn
import torchvision
import torch
import torch.optim as optim
from load_dataset import *
from encoder import *
from PIL import Image
import numpy as np
import cv2
import random
from dice_loss import dice_coeff
from torch.autograd import Variable
from dodge import img_dodge
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_class',default='checkpoint_class.pth.tar', help='Checkpoint of the classifier')
parser.add_argument('--checkpoint',default='checkpoint.pth.tar', help='Checkpoint of the localizer')
parser.add_argument('--image',default='train/val/cat.12025.jpg', help='Image name with the path')
parser.add_argument('--tamper',default=False, type=bool,help='Tamper image? (True/False)')
parser.add_argument('--side',default='shuffle', help='Which side to tamper? (Options: left_bottom,left_top, right_bottom, right_top, center)')
args = parser.parse_args()
checkpoint_class=args.checkpoint_class
checkpoint=args.checkpoint
image=args.image
tamper=args.tamper
side=args.side
classes=('Normal','Manipulated')
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function for testing on an image given the two checkpoints (one for tampering detection and another
# for tampering localization) 
def test(checkpoint_class,checkpoint,image_name,tamper=False,side='center'):
    sides=['left_top','left_bottom','right_top','right_bottom','center']

    # if side==shuffle, pick random part of image to tamper
    if side=='shuffle':
        side=sides[random.randint(0,4)]
    
    # Load checkpoints
    checkpoint=torch.load(checkpoint)
    encoder=checkpoint['encoder']
    checkpoint_class=torch.load(checkpoint_class)
    encoder_class=checkpoint_class['encoder']

    img=Image.open(image_name)
    img=img.resize((256,256))
    img=np.array(img)[:,:,::-1] 
    if tamper==True:   
        img_tamp,mask=img_dodge(img,side)
    else:
        img_tamp=img
    cv2.imshow('',img_tamp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_tamp=img_tamp.transpose(2,0,1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    assert img_tamp.shape==(3,256,256)
    assert np.max(img_tamp)<=255
    img_tamp=torch.FloatTensor(img_tamp//255.0)
    img_tamp= normalize(img_tamp)
    img_tamp=img_tamp.unsqueeze(0)    
    img_tamp=img_tamp.to(device)
    output=encoder_class(img_tamp)

    if classes[torch.max(output,1)[1]]=='Manipulated':
        score=encoder(img_tamp)
    else:
        print("Image is not Manipulated")
        exit(0)
    t = Variable(torch.Tensor([0.9])).to(device)
    predicted_mask_binarized=((score>t).float()*1).squeeze(0).squeeze(0).detach().cpu().numpy()
    predicted_mask=score.squeeze(0).squeeze(0).detach().cpu().numpy()
    predicted_mask*=255.
    print("Manipulated Image")
    cv2.imshow('',predicted_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test(checkpoint_class,checkpoint,image,tamper,side)