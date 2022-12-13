from Resnet import *

Model = {
    #'arch': 'tf_efficientnet_b4', #OK (Just ONNX, Opset = 11)
    #'arch': 'convit_tiny', #OK (Just ONNX, Opset = 11)
    #'arch': 'cait_s24_224', #OK (Opset = 11)
    #'arch': 'coat_tiny', #OK (Opset = 10)
    #'arch': 'gmlp_s16_224', #OK (Opset = 11)
    #'arch': 'inception_v4', #OK (Opset = 11)
    #'arch': 'resnet50', #OK (Opset = 11)
    #'arch': 'mixer_b16_224_in21k', #OK (Opset = 11)
    #'arch': 'deit_base_patch16_224', #OK (Opset = 11)
    #'arch': 'vit_base_patch16_224', #OK (Opset = 11)
    #'arch': 'resmlp_12_224', #ERROR: Operator addcmul
    
    #'arch': 'Resnet50',
    #'arch': 'Resnet18',
    #'arch': 'Resnet10',
    #'arch': 'Resnet8_V1',
    #'arch': 'Resnet8_V2',
    #'arch': 'Resnet8_V3',
    'arch': 'Resnet8_V4',
    
    'device': 'cuda:0'
}

if __name__ == '__main__':
    PATH = f'./PTHModels/{Model["arch"]}.pth'
    print(f'Start loading model {Model["arch"]}')
    model = torch.load(PATH)
    print(model)
    print('Loading model successfull!')
    
    BATCH_SIZE = 1
    device = torch.device(Model['device'])
    dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224).to(device)

    Output = f'./ONNXModels/{Model["arch"]}.onnx'
    #For Coat_tiny
    #torch.onnx.export(model, dummy_input, Output, opset_version=10, verbose=True)

    torch.onnx.export(model, dummy_input, Output, opset_version=11, verbose=True)
    print('Convert model to ONNX successfully!')

    print('Convert to TRT...')
    COMMAND = f'trtexec --onnx={Output} --saveEngine=./TRTModels/{Model["arch"]}.trt --explicitBatch --inputIOFormats=int8:chw --outputIOFormats=int8:chw --int8'
    print(COMMAND)
    
