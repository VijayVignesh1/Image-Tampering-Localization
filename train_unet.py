from torch import nn
import torchvision
import torch
import torch.optim as optim
from load_dataset_unet import *
from encoder import Encoder, AverageMeter
from encoder_unet import UNet
import numpy as np
import cv2
import random
from dice_loss import dice_coeff
global classes
# Save the states as checkpoint
def save_checkpoint(file_name,epochs,encoder):
    state={'epochs':epochs,'encoder':encoder}
    torch.save(state,file_name)
learning_rate=1e-4
epochs=12
batch_size=1
start_epoch=0
checkpoint=None #To train from scratch uncomment line
# checkpoint='checkpoint.pth.tar' # To train from checkpoint uncomment line
acc_avg = AverageMeter()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training and validation dataset
train_len=len(DatasetLoad(type='train'))

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

# Start Training
for epoch in range(start_epoch,epochs):
    running_loss = 0.0
    print("-------------------")
    print("Epoch %d"%(epoch))
    print("-------------------")
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
            print("Loss %d / %d : %.3f"%(i, train_len/batch_size, running_loss/100))
            running_loss=0.0
print("Saving Checkpoint..")
save_checkpoint('checkpoint.pth.tar',epochs,encoder)
running_loss=0.0

# Start Validation
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

