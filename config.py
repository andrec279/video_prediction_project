pretrain_config = {
    'pretrain': True,
    'model_id': None,
    'patch_size': 8,
    'embed_dim': 256,
    'expander_out': 600,
    'batch_size': 16,
    'num_epochs': 5,
    'optimizer': 'Adam', # choices: 'Adam', 'SGD', 'AdamW'
    'lr': 0.001
}

finetune_config = {
    'kernel_size': 3,
    'padding': 1,
    'stride': 2,
    'batch_size': 16,
    'num_epochs': 10,
    'lr': 0.001
}