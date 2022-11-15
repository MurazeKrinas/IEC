import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision import datasets
from PIL import Image
import timm
import timeit
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

CFG = {
    'fold_num': 5,
    'seed': 719,
    #'model_arch': 'tf_efficientnet_b4', #OK (Just ONNX, Opset = 11)
    #'model_arch': 'convit_tiny', #OK (Just ONNX, Opset = 11)

    #'model_arch': 'cait_s24_224', #OK (Opset = 11)
    #'model_arch': 'coat_tiny', #OK (Opset = 10)
    #'model_arch': 'gmlp_s16_224', #OK (Opset = 11)
    #'model_arch': 'inception_v4', #OK (Opset = 11)
    'model_arch': 'resnet50', #OK (Opset = 11)
    
    #'model_arch': 'mixer_b16_224_in21k', #Not enough memory
    #'model_arch': 'deit_base_patch16_224', #Not enough memory 
    #'model_arch': 'vit_base_patch16_224', #Not enough memory 
    #'model_arch': 'resmlp_12_224', #ERROR: Operator addcmul
    'device': 'cuda:0'
}

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger()) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

print('Start loading model...')
PATH = f'./TRTModels/{CFG["model_arch"]}.trt'
model = load_engine(PATH)
print('Load model successfull!')

'''
print('\nStart load dataset...')
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]
transform_norm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean, std)])

dataset = datasets.ImageFolder('./Dataset/test_images/', transform=transform_norm)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
print('Load dataset successfull!')

print('\nStart validation...')
cnt = Avg = 0
for images in dataloader:    
    images, labels = next(iter(dataloader))
    images = images[0].float()
    images = images.unsqueeze_(0)
    images = images.to(device)
    
    with torch.no_grad():        
        model.eval()
        start = timeit.default_timer()
        output = model(images)
        stop = timeit.default_timer()
        print('Time for image',cnt,':', stop - start)
        cnt += 1
        Avg += (stop - start) / 8
print(f'\n=> Time of {CFG['model_arch']}: {Avg})

f = open("Benmark.txt", "a")
s = f'\nTime of {CFG['model_arch']}: {str(Avg)} (second)\n'
f.write(s)
f.close()
'''