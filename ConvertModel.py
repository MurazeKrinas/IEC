from Resnet import *

if __name__ == '__main__':
    device = torch.device(CFG['device'])
    PATH = f'./PTHModels/{CFG["model_arch"]}.pth'
    print(f'Start loading model {CFG["model_arch"]}')
    model = torch.load(PATH).eval().to(device)
    print('Loading model successfull!')
    
    BATCH_SIZE = 8
    dummy_input=torch.randn(BATCH_SIZE, 3, CFG['img_size'], CFG['img_size']).to(device)

    Output = f'./ONNXModels/{CFG["model_arch"]}.onnx'
    #For Coat_tiny
    #torch.onnx.export(model, dummy_input, Output, opset_version=10, verbose=True)

    torch.onnx.export(model, dummy_input, Output, verbose=False)
    print('Convert model to ONNX successfully!')

    print('Convert to TRT...')
    COMMAND = f'trtexec --onnx={Output} --saveEngine=./TRTModels/{CFG["model_arch"]}.trt --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16'
    print(COMMAND)
