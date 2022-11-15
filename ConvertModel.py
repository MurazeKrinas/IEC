import torch
import torch.onnx
from torch import nn
import timm
import os
import time

CFG = {
    #'model_arch': 'tf_efficientnet_b4', #OK (Just ONNX, Opset = 11)
    #'model_arch': 'convit_tiny', #OK (Just ONNX, Opset = 11)

    #'model_arch': 'cait_s24_224', #OK (Opset = 11)
    #'model_arch': 'coat_tiny', #OK (Opset = 10)
    #'model_arch': 'gmlp_s16_224', #OK (Opset = 11)
    #'model_arch': 'inception_v4', #OK (Opset = 11)
    #'model_arch': 'resnet50', #OK (Opset = 11)
    
    #'model_arch': 'mixer_b16_224_in21k', #Not enough memory
    #'model_arch': 'deit_base_patch16_224', #Not enough memory 
    #'model_arch': 'vit_base_patch16_224', #Not enough memory 
    #'model_arch': 'resmlp_12_224', #ERROR: Operator addcmul
    'device': 'cuda:0'
}

class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)

        #1 efficientNet initilization
        #n_features = self.model.classifier.in_features
        #self.model.classifier = nn.Linear(n_features, n_class)
        
        #2 inceptionv4 initilization        
        #n_features = self.model.num_features
        #self.last_linear = nn.Linear(n_features, n_class)
        
        #3 resnet50 initilization        
        #n_features = self.model.num_features
        #self.fc = nn.Linear(n_features, n_class)
        
        #4 MLP-mixer, gmlp_s16_224, ResMLP initilization        
        #n_features = self.model.num_features
        #self.head = nn.Linear(n_features, n_class)  
        
        #ViT, Deit, CaiT, Coat, ConViT initilization
        #self.model.head = nn.Linear(self.model.head.in_features, n_class)
        
    def forward(self, x):
        x = self.model(x)
        return x


print('Start loading model...')
device = torch.device(CFG['device'])
model = CassvaImgClassifier(CFG['model_arch'], 4, pretrained=True).to(device)

PATH = f'./trained_model/{CFG["model_arch"]}'
model.load_state_dict(torch.load(PATH))
model.eval()
print('Load model successfull!')

BATCH_SIZE = 1
dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224).to(device)

Output = f'./ONNXModels/{CFG["model_arch"]}.onnx'
#For Coat_tiny
#torch.onnx.export(model, dummy_input, Output, opset_version=10, verbose=False)

#For ConViT, Efficientnet, Resnet50, Inception, Gmlp, CaiT
#torch.onnx.export(model, dummy_input, Output, opset_version=11, verbose=False)
print('Convert model to ONNX successfully!')

print('Convert to TRT...')
COMMAND = f'trtexec --onnx={Output} --saveEngine=./TRTModels/{CFG["model_arch"]}.trt'
print(COMMAND)

