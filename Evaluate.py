from IEC import IEC
from CassvaImgClassifier import CassvaImgClassifier
import sklearn
from sklearn.model_selection import GroupKFold, StratifiedKFold
import torch.quantization
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch import nn
import timm
import random
import pandas as pd


def GetValidateData():
    Valid = pd.read_csv('./Dataset/Train.csv')
    ValidDataframe = {
        'image_id': [],
        'label': [],
    }
    MinimalDataset = pd.DataFrame(ValidDataframe)

    DatasetLen = len(Valid)
    NumValidateImg = 20
    ID = random.sample(range(0,DatasetLen), NumValidateImg)
    for Obj in ID:
        Obj_Name = Valid.iloc[Obj,0]
        Obj_Label = Valid.iloc[Obj,1]
        #print(Obj_Name, Obj_Lable),
        MinimalDataset.loc[len(MinimalDataset)] = [Obj_Name, Obj_Label]

    MinimalDataset.to_csv('./Dataset/MinimalValidateDataset.csv', index = False)

if __name__ == '__main__':
    GetValidateData()

    valid = pd.read_csv('./Dataset/MinimalValidateDataset.csv')
    print(valid.head())
    
    IEC.seed_everything(IEC.CFG['seed'])
    IEC.folds = StratifiedKFold(n_splits=IEC.CFG['fold_num'], shuffle=True, random_state=IEC.CFG['seed']).split(np.arange(valid.shape[0]), valid.label.values)
    device = torch.device(IEC.CFG['device'])
    print(device)
    
    print('Load model')
    model = CassvaImgClassifier(IEC.CFG['model_arch'], valid.label.nunique(), pretrained=True).to(device)
    model.load_state_dict(torch.load('./trained_model/Resnet50_CornDataset'))
    print('Load model successfull!')	
    
    '''
    for fold, (trn_idx, val_idx) in enumerate(IEC.folds):
        if fold > 0:
            break
        print('Evaluate with {} started'.format(fold))

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = IEC.prepare_dataloader(valid, trn_idx, val_idx)

        loss_fn = nn.CrossEntropyLoss().to(device)
        
        for epoch in range(IEC.CFG['epochs']):
            model.eval()
            with torch.no_grad():
                IEC.valid_one_epoch(0, fold, epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
'''