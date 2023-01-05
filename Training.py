from Resnet import *

if __name__ == '__main__':
    print('Start building Model...')
    Model = Resnet(CFG['model_arch'])
    Device = torch.device(CFG['device'])
    Model.to(Device)
    #print(Model)
    print('Build Model successfully!')

    print('\nStart reading CSV file...')
    train = pd.read_csv('./Dataset/Train.csv')
    # SplitData()
    # train = pd.read_csv('./Dataset/MinimalTrainDataset.csv')
    #print(valid.head())
    seed_everything(CFG['seed'])
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)
    print('Read file CSV successfully!')
    print('\n=================================================\n\n')
    
    f = open("TrainingResult.txt","a")
    for fold, (trn_idx, val_idx) in enumerate(folds):
        print(f'\n* Start training with fold {fold}...')
        print(f'* Length train and valid index: {len(trn_idx)} - {len(val_idx)}')
        print(f'* Batch size: {CFG["train_bs"]} \t Epochs: {CFG["epochs"]}')
        train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx)
        #scaler = GradScaler()   
        optimizer = torch.optim.Adam(Model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)
        
        loss_tr = loss_fn = nn.CrossEntropyLoss().to(Device)
        TrainingAccuracy = []
        ValidAccuracy = []
        TrainLoss = []
        ValidLoss = []
        early_stopping = EarlyStopping()
        for epoch in range(CFG['epochs']):
            print('=================================================')
            print(f'\n[ TRAINING EPOCH {epoch} ]')
            TrainModel(epoch, Model, loss_tr, optimizer, train_loader, Device, scheduler=scheduler, schd_batch_update=False)
            with torch.no_grad():
                print('\n[ EVALUATING TRAINING ACCURACY ]')
                Train = EvalModel(True, fold, epoch, Model, loss_fn, train_loader, Device, early_stopping)
                print('\n[ EVALUATING VALIDATION ACCURACY ]')
                Valid = EvalModel(False, fold, epoch, Model, loss_fn, val_loader, Device, early_stopping)
                print('\n-------------------------------------------------\n')
                
                TrainingAccuracy.append(Train[0])
                TrainLoss.append(Train[1])
                ValidAccuracy.append(Valid[0])
                ValidLoss.append(Valid[1])
                    
                if early_stopping.early_stop:
                    print('=> EARLY STOP! SAVED MODEL SUCCESSFULLY...')
                    EpochSaved = epoch - early_stopping.patience
                    print(f'=> Fold {fold} - Epochs {epoch} - Stop at {EpochSaved}')
                    print(f'=> Training accuracy: {TrainingAccuracy}')
                    print(f'=> Validating accuracy: {ValidAccuracy}')
                    print(f'=> Training loss: {TrainLoss}')
                    print(f'=> Validating loss: {ValidLoss}')
                    print('\n=================================================\n\n')

                    f.write('=> EARLY STOP! SAVED MODEL SUCCESSFULLY...\n')
                    f.write(f'=> Fold {fold} - Epochs {epoch} - Stop at {EpochSaved}\n')
                    f.write(f'=> Training accuracy: {TrainingAccuracy}\n')
                    f.write(f'=> Validating accuracy: {ValidAccuracy}\n')
                    f.write(f'=> Training loss: {TrainLoss}\n')
                    f.write(f'=> Validating loss: {ValidLoss}\n')
                    f.write('\n\n--------------------------------------------\n\n')
                    break