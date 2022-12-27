import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
#import pycuda.driver as cuda
#import pycuda.autoinit
import fnmatch
#import tensorrt as trt
#import timm
import torch
from torchvision import transforms
from torchvision import datasets
from torch import nn
import torch.nn as nn
import torch.onnx
from Tools import EarlyStopping
import pandas as pd
import random
import numpy as np
import cv2
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
import timeit
from albumentations.pytorch import ToTensorV2
from albumentations import (
          HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
          Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
          IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
          IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
      )
CFG = {
    #'model_arch': 'Convit_tiny', #OK but just ONNX :(
    #'model_arch': 'Coat_tiny', #OK but just ONNX :(
    #'model_arch': 'Tf_efficientnet', #OK but just ONNX :(
    
    'model_arch': 'Resnet8_V4', #OK :)
    #'model_arch': 'Resnet8_V3', #OK :)
    #'model_arch': 'Resnet8_V2', #OK :)
    #'model_arch': 'Resnet8_V1', #OK :)
    #'model_arch': 'Resnet10', #OK :)
    #'model_arch': 'Resnet18', #OK :)
    #'model_arch': 'Resnet50', #OK :)
    #'model_arch': 'Inception_v4', #OK :)
    #'model_arch': 'Gmlp', #OK :)
    #'model_arch': 'Mixer', #OK :)
    #'model_arch': 'Cait', #OK :)
    
    #'model_arch': 'Deit', #Killed
    #'model_arch': 'Vit', #Killed
    #'model_arch': 'Resmlp', #Operator not supported
    
    'type': np.float16,
    'fold_num': 4,
    'seed': 719,
    'numclass': 4,
    'img_size': 224,
    'epochs': 1000,
    'batch_size': 2,
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

class Dataset():
    def __init__(self, df, data_root,
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 do_fmix=False, 
                 fmix_params={
                     'alpha': 1., 
                     'decay_power': 3., 
                     'shape': (CFG['img_size'], CFG['img_size']),
                     'max_soft': True, 
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 },
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        self.output_label = output_label
        self.one_hot_label = one_hot_label
    
        if output_label == True:
            self.labels = self.df['label'].values
            
            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]

    def __len__(self):
        return self.df.shape[0]

    def get_img(path = './Dataset/CornDataset'):
        im_bgr = cv2.imread(path)
        im_rgb = im_bgr[:, :, [2, 1, 0]]
        return im_rgb

    def rand_bbox(size, lam):
        W = size[0]
        H = size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    
    def __getitem__(self, index: int):
        if self.output_label:
           target = self.labels[index]

        img  = Dataset.get_img("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))

        if self.transforms:
            
            img = self.transforms(image=img)['image']

        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = Dataset.get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                print(cmix_img)
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']
                    
                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = Dataset.rand_bbox((CFG['img_size'], CFG['img_size']), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (CFG['img_size'] * CFG['img_size']))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]

        if self.output_label == True:
            return img, target
        else:
            return img

def get_valid_transforms():
        return Compose([
                Resize(CFG['img_size'], CFG['img_size']),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)

def get_train_transforms():
        return Compose([
                RandomResizedCrop(CFG['img_size'], CFG['img_size']),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                CoarseDropout(p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)

def prepare_dataloader(df, trn_idx, val_idx, data_root='./Dataset/CornDataset'):
    
        train_ = df.loc[trn_idx,:].reset_index(drop=True)
        valid_ = df.loc[val_idx,:].reset_index(drop=True)
            
        train_ds = Dataset(train_, data_root, transforms=get_train_transforms(), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False)
        valid_ds = Dataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True)
        
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=CFG['train_bs'],
            pin_memory=False,
            drop_last=False,
            shuffle=True,        
            num_workers=CFG['num_workers'],
        )
        val_loader = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )
        return train_loader, val_loader

def TrainModel(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
        model.train()

        running_loss = None

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (imgs, image_labels) in pbar:
            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device).long()
            #print(imgs,image_labels)
            scaler = GradScaler()
            with autocast():
          
                image_preds = model(imgs)   #output = model(input)

                loss = loss_fn(image_preds, image_labels)
                
                scaler.scale(loss).backward()

                if running_loss is None:
                    running_loss = loss.item()
                else:
                    running_loss = running_loss * .99 + loss.item() * .01

                if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                    # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() 
                    
                    if scheduler is not None and schd_batch_update:
                        scheduler.step()

                if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                    description = f'Epoch {epoch} loss: {running_loss:.4f}'
                    
                    pbar.set_description(description)
                  
        if scheduler is not None and not schd_batch_update:
           scheduler.step()
    
def EvalModel(isTrain, fold, epoch, model, loss_fn, val_loader, device, StopHere, scheduler=None, schd_loss_update=False, early_stopping = EarlyStopping()):
        model.eval()

        loss_sum = 0
        sample_num = 0
        image_preds_all = []
        image_targets_all = []
        
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for step, (imgs, image_labels) in pbar:
            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device).long()
            
            image_preds = model(imgs)   #output = model(input)
            image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
            image_targets_all += [image_labels.detach().cpu().numpy()]
            
            loss = loss_fn(image_preds, image_labels)
            
            loss_sum += loss.item()*image_labels.shape[0]
            sample_num += image_labels.shape[0]  

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
                description = f'Epoch {epoch} loss: {loss_sum/sample_num:.4f}'
                pbar.set_description(description)
        
        image_preds_all = np.concatenate(image_preds_all)
        image_targets_all = np.concatenate(image_targets_all)
        
        print('Validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))
        print ("Classification report: ", (classification_report(image_targets_all, image_preds_all)))
        print ("F1 micro averaging:",(f1_score(image_targets_all, image_preds_all, average='micro')))
        
        if isTrain == True:
            print('Training loss', loss_sum/sample_num, epoch + fold*33)
            print('Training accuracy', (image_preds_all==image_targets_all).mean(), epoch + fold*33)
        else:
            print('Validation loss', loss_sum/sample_num, epoch + fold*33)
            print('Validation accuracy', (image_preds_all==image_targets_all).mean(), epoch + fold*33)
            
            early_stopping(loss_sum/sample_num, model)
            if early_stopping.early_stop:
                print('EARLY STOP!')
                StopHere = True
          
        if scheduler is not None:
            if schd_loss_update:
                scheduler.step(loss_sum/sample_num)
            else:
                scheduler.step()
        return (image_preds_all==image_targets_all).mean()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def SplitData():
    Train = pd.read_csv('./Dataset/Train.csv')
    TrainDataframe = {
        'image_id': [],
        'label': [],
    }
    MinimalDataset = pd.DataFrame(TrainDataframe)

    DatasetLen = len(Train)
    ImgPerLable = 250
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
