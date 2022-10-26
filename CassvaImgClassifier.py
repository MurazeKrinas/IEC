import torch
from torch import nn
import timm

class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained, num_classes = n_class)

        #resnet50 initilization
        n_features = self.model.num_features
        self.num_classes = n_class
        self.fc = nn.Linear(n_features, n_class)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
