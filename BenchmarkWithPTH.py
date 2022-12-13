from Resnet import *

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

    #'model_arch': 'Resnet50',
    #'model_arch': 'Resnet18',
    #'model_arch': 'Resnet10',
    #'model_arch': 'Resnet8_V1',
    #'model_arch': 'Resnet8_V2',
    #'model_arch': 'Resnet8_V3',
    #'model_arch': 'Resnet8_V4',
    'device': 'cuda:0'
}

PATH = f'./PTHModels/{CFG["model_arch"]}.pth'
model = torch.load(PATH)

print('\nStart load dataset...')
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]
transform_norm = transforms.Compose([transforms.Resize((CFG['img_size'],CFG['img_size'])), transforms.ToTensor(), transforms.Normalize(mean, std)])

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
        cnt += 1
        print('Time for image',cnt,':', stop - start)
        Avg += (stop - start) / 8
print(f'\n=> Time of {CFG["model_arch"]}: {Avg}')

f = open("Benchmark.txt", "a")
s = f'\nTime of {CFG["model_arch"]}: {str(Avg)} (second)\n'
f.write(s)
f.close()
