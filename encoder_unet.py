import torch
from torchvision.models import vgg16
from torch import nn
import torch.nn.functional as F
from unet_parts import *
"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vgg=vgg16(pretrained=False)
        modules=list(self.vgg.children())[:-1]
        modules.append(nn.Linear(in_features=25088, out_features=4096, bias=True))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout(p=0.5, inplace=False))
        modules.append(nn.Linear(in_features=4096, out_features=2048, bias=True))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Linear(4096,2))
        modules.append(nn.Dropout(p=0.5, inplace=False))
        modules.append(nn.Linear(in_features=4096, out_features=2, bias=True))
        self.vgg=nn.Sequential(*modules)
    def forward(self,x):
        out=self.vgg(x)
        return out
"""
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)