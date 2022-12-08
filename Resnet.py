import torch
import torch.nn as nn
from IEC import IEC
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=IEC.CFG['numclass'], groups=1, width_per_group=64,norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        if (layers[1] != 0):
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if (layers[2] != 0):
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if (layers[3] != 0):
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def Resnet(NumLayer, **kwargs):
    if (NumLayer == 18):
        Layer = [2,2,2,2]
    elif (NumLayer == 10):
        Layer = [1,1,1,1]
    return ResNet(BasicBlock, Layer, **kwargs)

def SplitData():
    Train = pd.read_csv('./Dataset/Train.csv')
    TrainDataframe = {
        'image_id': [],
        'label': [],
    }
    MinimalDataset = pd.DataFrame(TrainDataframe)

    DatasetLen = len(Train)
    ImgPerLable = 25
    L = R = 0
    for R in range(0, DatasetLen):
        if (R == DatasetLen - 1 or Train.iloc[R,1] != Train.iloc[R+1,1]):
            ID = random.sample(range(L,R), ImgPerLable)
            #print(L, R)
            #print(ID)
            for Obj in ID:
                Obj_Name = Train.iloc[Obj,0]
                Obj_Label = Train.iloc[Obj,1]
                #print(Obj_Name, Obj_Lable),
                MinimalDataset.loc[len(MinimalDataset)] = [Obj_Name, Obj_Label]
            L = R + 1

    MinimalDataset.to_csv('./Dataset/MinimalTrainDataset.csv', index = False)

if __name__ == '__main__':
    print('Start building Model...')
    Model = Resnet(10)
    Device = torch.device(IEC.CFG['device'])
    Model.to(Device)
    #print(Model)
    print('Build Model successfully!')
    
    print('Start rebuild and read CSV file...')
    SplitData()
    train = pd.read_csv('./Dataset/MinimalTrainDataset.csv')
    #print(train.head())
    IEC.seed_everything(IEC.CFG['seed'])
    IEC.folds = StratifiedKFold(n_splits=IEC.CFG['fold_num'], shuffle=True, random_state=IEC.CFG['seed']).split(np.arange(train.shape[0]), train.label.values)

    for fold, (trn_idx, val_idx) in enumerate(IEC.folds):
        print(f'Start training with fold {fold}...')
        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = IEC.prepare_dataloader(train, trn_idx, val_idx, data_root='./Dataset/train_images')

        #scaler = GradScaler()   
        optimizer = torch.optim.Adam(Model.parameters(), lr=IEC.CFG['lr'], weight_decay=IEC.CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=IEC.CFG['T_0'], T_mult=1, eta_min=IEC.CFG['min_lr'], last_epoch=-1)
        
        loss_tr = nn.CrossEntropyLoss().to(Device)
        
        for epoch in range(IEC.CFG['epochs']):
            IEC.train_one_epoch(fold, epoch, Model, loss_tr, optimizer, train_loader, Device, scheduler=scheduler, schd_batch_update=False)
        
        ExportPATH = './PTHModels/Resnet10.pth'
        torch.save(Model, ExportPATH)
        print(f'Save pretrained model Resnet10 successfull!')