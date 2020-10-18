import torch
from torchvision.models import vgg16
from torch import nn
import torch.nn.functional as F
# Encoder using existing architecture
# Uncomment and comment the next Enoder function to VGGNet architecture
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

# Custom architecture using conv, maxpooling and linear layers

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(59536, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 59536)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

