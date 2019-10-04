from torch import nn
import torchvision
import torch
import torch.optim as optim
from load_dataset_unet import *
from encoder import Encoder, AverageMeter
from encoder_unet import UNet
from scipy.misc import imread, imresize
import numpy as np
import cv2
import random
from dice_loss import dice_coeff
global classes
def save_checkpoint(file_name,epochs,encoder):
    state={'epochs':epochs,'encoder':encoder}
    torch.save(state,file_name)
learning_rate=1e-4
epochs=12
batch_size=1
start_epoch=0
#checkpoint=None
checkpoint='checkpoint.pth.tar'
acc_avg = AverageMeter()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = torch.utils.data.DataLoader(
                                            DatasetLoad(type='train'),
                                            batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
                                            )
val_loader = torch.utils.data.DataLoader(
                                             DatasetLoad(type='val'),
                                             batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
                                             )

print('Data Loaded Succesfully..')

if checkpoint==None:
    encoder=UNet(3,1)
else:
    check=torch.load(checkpoint)
    encoder=check['encoder']
    start_epoch=check['epochs'] + 1
    print(start_epoch)
classes=('Normal','Manipulated')
criterion = nn.BCELoss().to(device)
optimizer=optim.Adam(encoder.parameters(),lr=learning_rate)
encoder=encoder.to(device)

for epoch in range(start_epoch,epochs):
    running_loss = 0.0
    print("Epoch %d"%(epoch))
    for i,data in enumerate(train_loader):
        encoder.train()
        images,masks=data
        images=images.to(device)
        masks=masks.to(device)
        optimizer.zero_grad()
        outputs=encoder(images)
        outputs=outputs.view(-1)
        masks=masks.view(-1)
        loss=criterion(outputs,masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%100==0:
            print("Loss: %.3f"%(running_loss/100))
            running_loss=0.0
print("Saving Checkpoint..")
save_checkpoint('checkpoint.pth.tar',epochs,encoder)
running_loss=0.0
for i,data in enumerate(val_loader):
    encoder.eval()
    images,masks=data
    images=images.to(device)
    masks=masks.to(device)
    scores=encoder(images)
    scores=scores.view(-1)
    masks=masks.view(-1)
    loss=criterion(scores,masks)
    running_loss+=loss.item()
    if i%100==0:
        print("Loss: %.3f"%(running_loss/100))
        running_loss=0.0

