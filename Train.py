
from IEC import IEC
from CassvaImgClassifier import CassvaImgClassifier
import sklearn
from sklearn.model_selection import GroupKFold, StratifiedKFold
import torch.quantization
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch import nn

import pandas as pd
import random

def SplitData():
    Train = pd.read_csv('./Dataset/Train.csv')
    TrainDataframe = {
        'image_id': [],
        'label': [],
    }
    MinimalDataset = pd.DataFrame(TrainDataframe)

    DatasetLen = len(Train)
    ImgPerLable = 4
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
    SplitData()
 
    train = pd.read_csv('./Dataset/MinimalTrainDataset.csv')
    IEC.seed_everything(IEC.CFG['seed'])
    print(train.head())
    
    IEC.folds = StratifiedKFold(n_splits=IEC.CFG['fold_num'], shuffle=True, random_state=IEC.CFG['seed']).split(np.arange(train.shape[0]), train.label.values)
    
    print('Start loading model...')
    PATH = f'./trained_model/{IEC.CFG["model_arch"]}'
    device = torch.device(IEC.CFG['device'])

    model = CassvaImgClassifier(IEC.CFG['model_arch'], 4, pretrained=True).to(device)
    model.load_state_dict(torch.load(PATH))
    print('Load model successfull!')
    
    for fold, (trn_idx, val_idx) in enumerate(IEC.folds):
        # we'll train fold 0 first
        if fold > 0:
          break

        print('Training with {} started'.format(fold))

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = IEC.prepare_dataloader(train, trn_idx, val_idx, data_root='./Dataset/train_images')

        #scaler = GradScaler()   
        optimizer = torch.optim.Adam(model.parameters(), lr=IEC.CFG['lr'], weight_decay=IEC.CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=IEC.CFG['T_0'], T_mult=1, eta_min=IEC.CFG['min_lr'], last_epoch=-1)
        
        loss_tr = nn.CrossEntropyLoss().to(device)
        
        for epoch in range(IEC.CFG['epochs']):
            IEC.train_one_epoch(fold, epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=False)
            torch.save(model.state_dict(),'./trained_model/Resnet50_CornDataset') 
            print("Saved model!")