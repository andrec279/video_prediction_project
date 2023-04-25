import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import models as m
import utils as u

class PretrainVICReg:
    def __init__(self,
                 image_size=(160,240),
                 patch_size=8,
                 embed_dim=256,
                 expander_out=600,
                 batch_size=32,
                 num_epochs=10,
                 lr=0.001,
                 unlabeled_folder='Dataset_Student/unlabeled/'):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set parameters for patch embeddings
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.expander_out = expander_out
        self.batch_size = batch_size
        self.unlabeled_folder = unlabeled_folder
        self.num_epochs = num_epochs
    
        self.dataloader = u.create_pretrain_dataloader(unlabeled_folder, image_size, batch_size)
        self.VICReg_model = m.VICReg(image_size, patch_size, embed_dim, expander_out=expander_out).to(self.device)
        self.optimizer = torch.optim.Adam(self.VICReg_model.parameters(), lr=lr)
    
    
    def pretrain(self):
        for epoch in range(self.num_epochs):
            self.VICReg_model.train()
            epoch_loss = []
            for data in tqdm(self.dataloader, desc=f'epoch {epoch}/{self.num_epochs}'):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.VICReg_model(x, y)
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            print(f'Epoch {epoch} loss: {np.mean(epoch_loss)}')

        self.model_name = f'VICReg_pretrained_{round(time.time())}.pth'
        torch.save(self.VICReg_model.state_dict(), self.model_name)
        print('Successfully pretrained and saved VICReg model')