import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from segmentation_models_pytorch.losses import JaccardLoss

import models as m
import utils as u
import config
from pretrain import PretrainVICReg
from finetune import FinetuneVideoPredictor


def main():
    if config.pretrain_config['pretrain'] == True:
        
        VICReg_training = PretrainVICReg(patch_size=config.pretrain_config['patch_size'],
                                         embed_dim=config.pretrain_config['embed_dim'],
                                         expander_out=config.pretrain_config['expander_out'],
                                         batch_size=config.pretrain_config['batch_size'],
                                         num_epochs=config.pretrain_config['num_epochs'],
                                         optimizer=config.pretrain_config['optimizer'],
                                         lr=config.pretrain_config['lr'])
        VICReg_training.pretrain()
        VICReg_model_path = VICReg_training.model_name
    
    else:
        if config.pretrain_config['model_id'] == None:
            raise ValueError('Must provide model_id if configuration "pretrain" is set to False')
        if not os.path.exists(config.pretrain_config['model_id']):
            raise ValueError(f'{config.pretrain_config["model_id"]} not found in directory, please check your spelling.')
        VICReg_model_path = config.pretrain_config['model_id']

    video_predictor_finetuning = FinetuneVideoPredictor(VICReg_model_path, 
                                                        patch_size=config.pretrain_config['patch_size'],
                                                        embed_dim=config.pretrain_config['embed_dim'],
                                                        expander_out=config.pretrain_config['expander_out'],
                                                        kernel_size=config.finetune_config['kernel_size'], 
                                                        padding=config.finetune_config['padding'], 
                                                        stride=config.finetune_config['stride'],
                                                        batch_size=config.finetune_config['batch_size'],
                                                        num_epochs=config.finetune_config['num_epochs'],
                                                        lr=config.finetune_config['lr'])
    video_predictor_finetuning.finetune()

if __name__ == '__main__':
    main()