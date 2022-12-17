from Resnet import *

if __name__ == '__main__':
    PATH = f'./PTHModels/{CFG["model_arch"]}.pth'
    print(f'Start loading model {CFG["model_arch"]}')
    model = torch.load(PATH)
    print(model)
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
    COMMAND = f'trtexec --onnx={Output} --saveEngine=./TRTModels/{CFG["model_arch"]}.trt --explicitBatch --inputIOFormats=int8:chw --outputIOFormats=int8:chw --best'
    print(COMMAND)