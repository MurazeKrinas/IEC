from Resnet import *

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
    COMMAND = f'trtexec --onnx={Output} --saveEngine=./TRTModels/{CFG["model_arch"]}.trt --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16'
    print(COMMAND)
