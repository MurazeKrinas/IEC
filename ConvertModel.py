import torch
import torch.onnx
from torch import nn
import timm

CFG = {
    #'model_arch': 'tf_efficientnet_b4', #OK (Just ONNX, Opset = 11)
    #'model_arch': 'convit_tiny', #OK (Just ONNX, Opset = 11)

    #'model_arch': 'cait_s24_224', #OK (Opset = 11)
    #'model_arch': 'coat_tiny', #OK (Opset = 10)
    #'model_arch': 'gmlp_s16_224', #OK (Opset = 11)
    #'model_arch': 'inception_v4', #OK (Opset = 11)
    'model_arch': 'resnet50', #OK (Opset = 11)
    #'model_arch': 'mixer_b16_224_in21k', #OK (Opset = 11)
    #'model_arch': 'deit_base_patch16_224', #OK (Opset = 11)
    #'model_arch': 'vit_base_patch16_224', #OK (Opset = 11)
    
    #'model_arch': 'resmlp_12_224', #ERROR: Operator addcmul
    'device': 'cuda:0'
}

class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
      
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    PATH = f'./PTHModels/{CFG["model_arch"]}.pth'
    print(f'Start loading model {CFG["model_arch"]}')
    model = torch.load(PATH)
    print('Loading model successfull!')

    BATCH_SIZE = 1
    device = torch.device(CFG['device'])
    dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224).to(device)

    Output = f'./ONNXModels/{CFG["model_arch"]}.onnx'
    #For Coat_tiny
    #torch.onnx.export(model, dummy_input, Output, opset_version=10, verbose=True)

    torch.onnx.export(model, dummy_input, Output, opset_version=11, verbose=True)
    print('Convert model to ONNX successfully!')

    print('Convert to TRT...')
    COMMAND = f'trtexec --onnx={Output} --saveEngine=./TRTModels/{CFG["model_arch"]}.trt --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16'
    print(COMMAND)