import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from segmentation_models_pytorch.losses import JaccardLoss
import torchmetrics

import models as m
import utils as u
import gc

torch.cuda.empty_cache()

class FinetuneVideoPredictor:
    def __init__(self,
                 VICReg_model_path,
                 image_size=(160,240),
                 patch_size=8,
                 embed_dim=256,
                 expander_out=600,
                 kernel_size=3,
                 padding=1,
                 stride=2,
                 batch_size=32,
                 num_epochs=10,
                 lr=0.001,
                 train_folder="Dataset_Student/train/",
                 val_folder="Dataset_Student/val/"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.video_prediction_trainloader = u.create_finetune_dataloader(train_folder, image_size, batch_size, train_or_val='train')
        self.video_prediction_valloader = u.create_finetune_dataloader(val_folder, image_size, batch_size, train_or_val='val')

        VICReg_model = m.VICReg(image_size, patch_size, embed_dim, expander_out=expander_out)
        VICReg_model.load_state_dict(torch.load(VICReg_model_path))
        VICReg_model.eval()

        self.video_prediction_model = m.VideoPredictor(VICReg_model, kernel_size, padding, stride)
        self.video_prediction_model = self.video_prediction_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.video_prediction_model.parameters(), lr=lr)
        # self.criterion = JaccardLoss(mode='multiclass', classes=49).to(self.device)
        self.criterion = nn.CrossEntropyLoss() 
    
    def finetune(self):
        train_losses = []
        val_losses = []
        best_val_loss = 9999
        time_best_val = round(time.time())
        for epoch in range(self.num_epochs):
            self.video_prediction_model.train()
            epoch_train_loss = []
            epoch_val_loss = []

            for data in tqdm(self.video_prediction_trainloader, desc=f'epoch {epoch}'):
                x_train, y_train = data
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)
                self.optimizer.zero_grad()
                y_train_pred = self.video_prediction_model(x_train)
                loss = self.criterion(y_train_pred, y_train)
                
                epoch_train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
                del x_train, y_train 
                gc.collect()

            epoch_train_loss_mean = np.mean(epoch_train_loss)
            print(f'Epoch {epoch} train loss: {np.mean(epoch_train_loss)}')
            train_losses.append(epoch_train_loss_mean)

            with torch.no_grad():
                for data in self.video_prediction_valloader:
                    x_val, y_val = data  # batch_size, num_channels = 3, image_size
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    y_val_pred = self.video_prediction_model(x_val)

                    #input to the model should be patch_embeddings learned during pretraining
                    loss = self.criterion(y_val_pred, y_val)
                    epoch_val_loss.append(loss.item())
                    
                    del x_val, y_val 
                    gc.collect()

                epoch_val_loss_mean = np.mean(epoch_val_loss)
                print(f'Epoch {epoch} val loss: {np.mean(epoch_val_loss)}')
                val_losses.append(epoch_val_loss_mean)

                if epoch_val_loss_mean < best_val_loss:
                    best_val_loss = epoch_val_loss_mean
                    self.best_model_name = f"video_predictor_finetuned_best_val_{time_best_val}.pth"
                    torch.save(self.video_prediction_model.state_dict(), self.best_model_name)

        self.model_name = f"video_predictor_finetuned_{round(time.time())}.pth"
        torch.save(self.video_prediction_model.state_dict(), self.model_name)
        print('Successfully fine-tuned and saved video frame predictor model')
        