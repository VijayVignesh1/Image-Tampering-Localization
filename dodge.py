from scipy.misc import imread, imresize
import numpy as np
import cv2
import random
def dodgeV2(image, mask):
    return cv2.divide(image, 255-mask, scale=256)
def img_dodge(img,side='center'):
    if side=='left_top':
        x1=random.randint(0,50)
        x2=random.randint(100,140)
        y1=random.randint(0,50)
        y2=random.randint(100,140)
    elif side=='right_bottom':
        x1=random.randint(150,190)
        x2=random.randint(200,250)
        y1=random.randint(150,190)
        y2=random.randint(200,250)
    elif side=='right_top':
        x1=random.randint(0,50)
        x2=random.randint(100,150)
        y1=random.randint(150,170)
        y2=random.randint(200,250)   
    elif side=='left_bottom':
        x1=random.randint(150,190)
        x2=random.randint(200,250)
        y1=random.randint(0,50)
        y2=random.randint(100,140)
    else:
        x1=random.randint(100,110)
        x2=random.randint(120,140)
        y1=random.randint(100,110)
        y2=random.randint(120,140)   
    img=cv2.resize(img,(256,256))

    img_crop=img[x1:x2,y1:y2]
    img_blur = cv2.GaussianBlur(img_crop, ksize=(7, 7),sigmaX=0, sigmaY=0)
    img_blend=dodgeV2(img_crop,img_blur)
    img[x1:x2,y1:y2]=img_blend
    img_new=np.zeros((img.shape[0],img.shape[1]),np.uint8)
    rect=np.ones((x2-x1,y2-y1))
    rect[:]=255
    img_new[x1:x2,y1:y2]=rect
    return img,img_new
