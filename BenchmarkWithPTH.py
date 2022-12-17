from Resnet import *

print('Loading model...')
device = torch.device(CFG['device'])
PATH = f'./PTHModels/{CFG["model_arch"]}.pth'
Model = torch.load(PATH)
Model.to(device)
print('Load model successfully!')

print('\nLoading dataset...')
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]
transform_norm = transforms.Compose([transforms.Resize((CFG['img_size'],CFG['img_size'])), transforms.ToTensor(), transforms.Normalize(mean, std)])

dataset = datasets.ImageFolder('./Dataset/Images/', transform=transform_norm)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
print('Load dataset successfully!')

print('\nWarming up...')
dummy_input = torch.rand((1, 3, CFG['img_size'], CFG['img_size'])).float().to(device)
for i in range(5):
    Model(dummy_input)
print('Warm up done!')

print('\nStart validating...')
cnt = Avg = 0
for images in dataloader:    
    images, labels = next(iter(dataloader))
    images = images.float().to(device)
    labels = labels.int().to(device)
    
    with torch.no_grad():
        start = timeit.default_timer()
        output = Model(images)
        stop = timeit.default_timer()
        cnt += 1
        print('Time for image',cnt,':', stop - start)
        Avg += (stop - start) / 100
print(f'\n=> Time of {CFG["model_arch"]}: {Avg}')

f = open("Benchmark.txt", "a")
s = f'\nTime of {CFG["model_arch"]}: {str(Avg)} (second)\n'
f.write(s)
f.close()
