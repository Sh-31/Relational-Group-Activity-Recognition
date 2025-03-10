"""
PersonActivityClassifier Description:
--------------------------------
pretrained Resnet50 network is fined tuned and a person is represented with 4096-d features
"""

import torch
import argparse
import torch.nn as nn
import albumentations as A
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchinfo import summary

class PersonActivityClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PersonActivityClassifier, self).__init__()
        
        self.resnet50 = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape      # x.shape => batch, channals , hight, width
        x = self.resnet50(x)      # (batch, 2048, 1 , 1)
        x = x.view(b, -1)         # (batch, 2048)
        x = self.fc(x)            # (batch, num_class)          
        return x