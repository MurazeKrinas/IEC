import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision import datasets
from PIL import Image
import timm
import timeit

CFG = {
    #'model_arch': 'tf_efficientnet_b4', #OK (Just ONNX, Opset = 11)
    #'model_arch': 'convit_tiny', #OK (Just ONNX, Opset = 11)

    #'model_arch': 'cait_s24_224', #OK (Opset = 11)
    #'model_arch': 'coat_tiny', #OK (Opset = 10)
    #'model_arch': 'gmlp_s16_224', #OK (Opset = 11)
    #'model_arch': 'inception_v4', #OK (Opset = 11)
    #'model_arch': 'resnet50', #OK (Opset = 11)
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

PATH = f'./PTHModels/{CFG["model_arch"]}.pth'
model = torch.load(PATH)

print('\nStart load dataset...')
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]
transform_norm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean, std)])

dataset = datasets.ImageFolder('./Dataset/test_images/', transform=transform_norm)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
print('Load dataset successfull!')

device = torch.device(CFG['device'])
print('\nStart validation...')
cnt = Avg = 0
for images in dataloader:    
    images, labels = next(iter(dataloader))
    images = images[0].float()
    images = images.unsqueeze_(0)
    images = images.to(device)
    
    with torch.no_grad():        
        start = timeit.default_timer()
        output = model(images)
        stop = timeit.default_timer()
        print('Time for image',cnt,':', stop - start)
        cnt += 1
        Avg += (stop - start) / 8
print(f'\n=> Time of {CFG["model_arch"]}: {Avg}')

f = open("Benmark.txt", "a")
s = f'\nTime of {CFG["model_arch"]}: {str(Avg)} (second)\n'
f.write(s)
f.close()