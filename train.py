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
from dodge import img_dodge
global classes

# Function to save the states as checkpoint 
def save_checkpoint(file_name,epochs,encoder):
    state={'epochs':epochs,'encoder':encoder}
    torch.save(state,file_name)
learning_rate=1e-4
epochs=4
batch_size=10
start_epoch=0
# checkpoint='checkpoint_class.pth.tar'  # To train from checkpoint uncomment.
checkpoint=None # To train from scratch, uncomment
acc_avg = AverageMeter()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training and validation data
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
    encoder=Encoder()
else:
    check=torch.load(checkpoint)
    encoder=check['encoder']
    start_epoch=check['epochs'] + 1

classes=('Normal','Manipulated')
criterion=nn.CrossEntropyLoss().to(device)
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
        images,values=data
        images=images.to(device)
        values=values.to(device)
        optimizer.zero_grad()
        outputs=encoder(images)
        loss=criterion(outputs,torch.max(values, 1)[1])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%100==0:
            print("Loss %d / %d : %.3f"%(i, train_len/batch_size, running_loss/100))
            running_loss=0.0
            
# Run Validation
print("Entering Validation..")
for i, data in enumerate(val_loader):
    encoder.eval()  
    images,values=data
    images=images.to(device)
    values=values.to(device)
    scores=encoder(images)
    values=torch.max(values,1)[1]
    _,scores=torch.max(scores, dim=1)
    result=torch.sum(values==scores)
    acc = (result * 100.0 / len(values))
    acc_avg.update(acc)
    if i%100==0:
        print("scores:",scores)
        print("values:",values)    
        print("Accuracy: %f"%(acc))
print("Final Accuracy: ",acc_avg.avg)
print("Saving Checkpoint..")
save_checkpoint('checkpoint_class.pth.tar',epochs,encoder)

# Test on image function. Seems redundant (see test.py)
def test(checkpoint,image_name,tamper=False,side='center'):
    checkpoint=torch.load(checkpoint)
    encoder=checkpoint['encoder']
    img=Image.open(image_name)
    img=img.resize((256,256))
    img=np.array(img)[:,:,::-1]
    if tamper==True:    
        img,mask=img_dodge(img,side)
    imgs=img
    img=img.transpose(2,0,1)
    assert img.shape==(3,256,256)
    assert np.max(img)<=255
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    img=torch.FloatTensor(img//255.)
    img= normalize(img)
    img=img.unsqueeze(0)
    img=img.to(device)
    score=encoder(img)
    print("The Image is: ",classes[torch.max(score,1)[1]])
    cv2.imshow('',imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
test('checkpoint_class.pth.tar','train/val/cat.12197.jpg',True,'left_bottom')
        