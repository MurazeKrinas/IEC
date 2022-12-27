from Resnet import *
#from pthflops import count_ops 

if __name__ == '__main__':
    print('Start building Model...')
    Model = Resnet(CFG['model_arch'])
    Device = torch.device(CFG['device'])
    Model.to(Device)
    print(Model)
    print('Build Model successfully!')
    
    print('\nStart reading CSV file...')
    train = pd.read_csv('./Dataset/Train.csv')
    
    # SplitData()
    # train = pd.read_csv('./Dataset/MinimalTrainDataset.csv')
    #print(valid.head())
    seed_everything(CFG['seed'])
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)
    print('Read file CSV successfully!')

    
    for fold, (trn_idx, val_idx) in enumerate(folds):
        # if fold > 0:
        #     break
        print(f'\nStart training with fold {fold}...')
        print(f'Length train and valid index: {len(trn_idx)} - {len(val_idx)}')
        print(f'Batch size: {CFG["train_bs"]} \t Epochs: {CFG["epochs"]}')
        train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx)

        #scaler = GradScaler()   
        optimizer = torch.optim.Adam(Model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)
        
        loss_tr = loss_fn = nn.CrossEntropyLoss().to(Device)
        
        for epoch in range(CFG['epochs']):
            TrainModel(epoch, Model, loss_tr, optimizer, train_loader, Device, scheduler=scheduler, schd_batch_update=False)
            with torch.no_grad():
                print('\nEVALUATING TRAINING ACCURACY...')
                EvalModel(True, fold, epoch, Model, loss_fn, train_loader, Device)
                print('\nEVALUATING VALIDATION ACCURACY...')
                EvalModel(False, fold, epoch, Model, loss_fn, val_loader, Device)
                print('\n--------------------------------------------\n')

