import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import model.resnet_256 as resnet
from torch.nn import Parameter
import numpy as np
import math

class background_resnet(nn.Module):
    def __init__(self, num_classes, backbone='resnet34'):
        super(background_resnet, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=False)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False)
        elif backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.fc1 = Parameter(torch.Tensor(256, 256))
        nn.init.xavier_uniform_(self.fc1)

        self.weight = Parameter(torch.Tensor(num_classes, 256))
        nn.init.xavier_uniform_(self.weight)

        self.relu = nn.ReLU()

    def forward(self, x):
        # input x: minibatch x 1 x 40 x 40
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)

        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x) #[batch, 256, *, *]

        # Global Average Pooling
        x = x.flatten(start_dim=2)
        x = x.mean(dim=2)
        x = self.relu(self.pretrained.avg_bn(x))

        spk_embedding = F.linear(x, self.fc1)

        return spk_embedding
