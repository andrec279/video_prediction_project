import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

import utils as u

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
      super().__init__()
      self.proj = nn.Conv2d(in_channels, embed_dim, 
                            kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
      x = self.proj(x).flatten(2).transpose(1,2)
      return x
    
class VICReg(nn.Module):
    def __init__(self, patch_size, embed_dim, mlp_layers='256-256-256', sim_coeff=1/3, std_coeff=1/3, cov_coeff=1/3):
      super().__init__()
      self.encoder = PatchEmbedding(patch_size, 3, embed_dim)
      self.num_features = int(mlp_layers.split("-")[-1])
      self.sim_coeff = sim_coeff
      self.std_coeff = std_coeff
      self.cov_coeff = cov_coeff
      self.expander = expander_head(embed_dim, mlp_layers)

    def forward(self, x, y):
      # x and y shape (batch_size, in channels, image width, image height)
      x_orig = self.encoder(x)
      y_orig = self.encoder(y)

      # # pass the embeddings through the expander heads
      # x = self.expander(x_orig)
      # y = self.expander(y_orig)

      # BUILD THE PREDICTOR
      repr_loss = 0

      # normalize the data (WHY DO THIS??)
      x = x_orig - x_orig.mean(dim=0)
      y = y_orig - y_orig.mean(dim=0)

      # get the std and std_loss
      std_x = torch.sqrt(x.var(dim=0) + 0.0001)
      std_y = torch.sqrt(y.var(dim=0) + 0.0001)
      std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

      # get the covariance and cov loss
      x_T = torch.transpose(x, 1, 2)
      y_T = torch.transpose(y, 1, 2)
      cov_x = torch.bmm(x_T, x) / (x.size(1) - 1)
      cov_y = torch.bmm(y_T, y) / (y.size(1) - 1)
      # THIS LOSS IS CURRENTLY NOT WORKING CORRECTLY!!
      cov_loss = u.off_diagonal(cov_x).pow_(2).sum().div(self.num_features
        ) + u.off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
      
      loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss)
      
      return x_orig, y_orig, loss

def expander_head(embed_dim, mlp_layers):
   mlp_spec = f'{embed_dim}-{mlp_layers}'
   f = list(map(int, mlp_spec.split("-")))
   layers = []
   for i in range(len(f)-2):
      layers.append(nn.Linear(f[i], f[i + 1]))
      layers.append(nn.BatchNorm1d(f[i + 1]))
      layers.append(nn.ReLU(True))
      layers.append(nn.Linear(f[-2], f[-1], bias=False))
   return nn.Sequential(*layers)