pretrain_config = {
    'pretrain': False,
    'model_id': 'VICReg_pretrained_1682959065.pth',
    'patch_size': 8,
    'embed_dim': 64,
    'expander_out': 800, # should be greater than 160*240/patch_size^2
    'batch_size': 8,
    'num_epochs': 20,
    'optimizer': 'AdamW', # choices: 'Adam', 'SGD', 'AdamW'
    'lr': 0.001
}

finetune_config = {
    'kernel_size': 3, 
    'padding': 1,
    'stride': 1,
    'batch_size': 2,
    'num_epochs': 10,
    'lr': 0.005 
}
