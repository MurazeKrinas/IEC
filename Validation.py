import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision import datasets
from PIL import Image
import timm
import timeit

CFG = {
    'fold_num': 5,
    'seed': 719,
    #'model_arch': 'tf_efficientnet_b4',
    #'model_arch': 'vit_base_patch16_224',
    #'model_arch': 'deit_base_patch16_224', #ERROR
    #'model_arch': 'cait_s24_224', #ERROR
    #'model_arch': 'convit_tiny', #ERROR
    #'model_arch': 'inception_v4',
    'model_arch': 'resnet50',
    #'model_arch': 'coat_tiny',
    #'model_arch': 'resmlp_12_224', #ERROR
    #'model_arch': 'gmlp_s16_224',
    #'model_arch': 'mixer_b16_224_in21k',
    'img_size': 224,
    'epochs': 1,
    'train_bs': 32,
    'valid_bs': 32,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 4,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0'
}

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
        n_features = self.model.num_features
        self.fc = nn.Linear(n_features, n_class)
        
        #4 MLP-mixer, gmlp_s16_224, ResMLP initilization        
        #n_features = self.model.num_features
        #self.head = nn.Linear(n_features, n_class)  
        
        #ViT, Deit, CaiT, Coat, ConViT initilization
        #self.model.head = nn.Linear(self.model.head.in_features, n_class)
        
    def forward(self, x):
        x = self.model(x)
        return x

print('Start loading model...')
PATH = f'./trained_model/{CFG["model_arch"]}'
device = torch.device(CFG['device'])

model = CassvaImgClassifier(CFG['model_arch'], 4, pretrained=True).to(device)
model.load_state_dict(torch.load(PATH))
print('Load model successfull!')

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
print('\n=> Time of', CFG['model_arch'],':', Avg)

f = open("Benmark.txt", "a")
s = '\nTime of ' + CFG['model_arch'] + ': ' + str(Avg) + ' (second)\n'
f.write(s)
f.close()