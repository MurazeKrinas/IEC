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

device = torch.device(Model['device'])
PATH = f'./PTHModels/{Model["arch"]}.pth'
model = torch.load(PATH)
model.to(device)

print('\nStart load dataset...')
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]
transform_norm = transforms.Compose([transforms.Resize((112,112)), transforms.ToTensor(), transforms.Normalize(mean, std)])

dataset = datasets.ImageFolder('./Dataset/Images/', transform=transform_norm)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
print('Load dataset successfull!')

print('\nStart validation...')
cnt = Avg = 0
for images in dataloader:    
    images, labels = next(iter(dataloader))
    images = images[0].float().unsqueeze_(0).to(device)
    labels = labels.int().to(device)

    with torch.no_grad():        
        start = timeit.default_timer()
        output = model(images)
        stop = timeit.default_timer()
        cnt += 1
        print('Time for image',cnt,':', stop - start)
        Avg += (stop - start) / 100
print(f'\n=> Time of {Model["arch"]}: {Avg}')

f = open("Benchmark.txt", "a")
s = f'\nTime of {Model["arch"]}: {str(Avg)} (second)\n'
f.write(s)
f.close()
