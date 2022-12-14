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
        n_features = self.model.num_features
        self.head = nn.Linear(n_features, n_class)  
        
        #ViT, Deit, CaiT, Coat, ConViT initilization
        #self.model.head = nn.Linear(self.model.head.in_features, n_class)
        
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    device = torch.device(CFG['device'])
    start = timeit.default_timer()
    print('Loading model...')
    PATH = f'./PTHModels/{CFG["model_arch"]}.pth'
    Model = torch.load(PATH).eval().to(device)
    print('Load model successfully!')

    print('\nLoading dataset...')
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.Resize((CFG['img_size'],CFG['img_size'])), transforms.ToTensor(), transforms.Normalize(mean, std)])
    dataset = datasets.ImageFolder('./Dataset/Images/', transform=transform_norm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=True)
    print('Load dataset successfully!')

    print('\nWarming up...')
    for i in range(5):
        with torch.no_grad():
            dummy_input = torch.rand((1, 3, CFG['img_size'], CFG['img_size'])).to(device)
            torch.cuda.synchronize(device)
            Model(dummy_input)
            torch.cuda.synchronize(device)
    print('Warm up done!') 
    stop = timeit.default_timer()
    Preprocess = stop - start
    print(f'Preprocess: {Preprocess}')

    print('\nStart validating...')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cnt = Avg = 0
    Iteration = len(dataloader)
    print(f"Batchsize: {CFG['batch_size']}")
    print(f'Number of iteration: {Iteration}')

    for images in dataloader:    
        images,_ = next(iter(dataloader))
        images = images.to(device).float()
        with torch.no_grad():
            start = timeit.default_timer()
            torch.cuda.synchronize(device) 
            output = Model(images)
            torch.cuda.synchronize(device)
            stop = timeit.default_timer()
            print(output)
            cnt += 1
            print('Time for iteration',cnt,':', stop - start)
            Avg += (stop - start)

    Avg /= Iteration
    print(f'Average validating time per image of {CFG["model_arch"]}.pth: {Avg} second')

    f = open("Benchmark.txt", "a")
    s = f'Average validating time per image of {CFG["model_arch"]}.pth: {Avg} second\nPreprocess: {Preprocess}\n'
    f.write(s)
    f.close()