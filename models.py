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
    
class EmbedPosEncoding(nn.Module):
   def __init__(self, patch_embed, num_patches, embed_dim):
      super().__init__()
      self.patch_embed = patch_embed
      self.pos_emb = nn.Parameter(torch.randn(1, num_patches, embed_dim).normal_(std=0.02))
   
   def forward(self, x):
      patches = self.patch_embed(x)

      # Concatenate positional embeddings to patches
      positional_patches = patches + self.pos_emb

      return positional_patches
   
    
class VICReg(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, hidden_sizes=[1024,2048], expander_out=8192, sim_coeff=1/3, std_coeff=1/3, cov_coeff=1/3):
      super().__init__()
      self.patch_embed_x = PatchEmbedding(patch_size, 3*11, embed_dim)
      self.patch_embed_y = PatchEmbedding(patch_size, 3, embed_dim)
      
      self.sim_coeff = sim_coeff
      self.std_coeff = std_coeff
      self.cov_coeff = cov_coeff

      self.expander_out = expander_out

      self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
      self.encoder_x = EmbedPosEncoding(self.patch_embed_x, self.num_patches, embed_dim)
      self.encoder_y = EmbedPosEncoding(self.patch_embed_y, self.num_patches, embed_dim)

      self.expander = Expander(self.num_patches, expander_out)
      self.predictor = Predictor(self.num_patches*embed_dim, hidden_sizes, self.num_patches*embed_dim)

    def forward(self, x, y):
      # x and y shape (batch_size, in channels, image width, image height)
      s_x = self.encoder_x(x) 
      s_y = self.encoder_y(y)

      # pass the embeddings through the expander heads to increase channels
      v_x = self.expander(s_x)
      v_y = self.expander(s_y)

      # pass s_x through predictor to get s_yhat
      s_yhat = self.predictor(s_x) # s_yhat represents the predicted patch representations of image y given first 11 images
      repr_loss = F.mse_loss(s_yhat, s_y)

      # normalize the data across each embedding (is this necessary??)
      v_x = v_x - v_x.mean(dim=2).unsqueeze(2).repeat(1,1,v_x.size(2))
      v_y = v_y - v_y.mean(dim=2).unsqueeze(2).repeat(1,1,v_y.size(2))

      # get the std and std_loss
      std_x = torch.sqrt(v_x.var(dim=2) + 0.0001)
      std_y = torch.sqrt(v_y.var(dim=2) + 0.0001)
      std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

      # get the covariance and cov loss
      cov_x = u.batch_cov(v_x)/((v_x.size(0)-1)*v_x.size(1))
      cov_y = u.batch_cov(v_y)/((v_y.size(0)-1)*v_y.size(1))
      cov_loss = u.off_diagonal(cov_x).pow_(2).sum().div(v_x.size(2)
        ) + u.off_diagonal(cov_y).pow_(2).sum().div(v_y.size(2))
      
      loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss)
      
      return loss

class Expander(nn.Module):
   def __init__(self, in_channels, out_channels):
      super(Expander, self).__init__()
      # note trying conv transpose as suggested by ChatGPT, but could also try a MLP instead to expand channels
      self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
      self.bn = nn.BatchNorm1d(out_channels)
      self.relu = nn.ReLU(inplace=True)

   def forward(self, x):
      x = self.conv_transpose(x)
      x = self.bn(x)
      x = self.relu(x)
      return x
   
class Predictor(nn.Module):
   def __init__(self, input_size, hidden_sizes, output_size):
      super(Predictor, self).__init__()
      self.input_size = input_size
      self.hidden_sizes = hidden_sizes
      self.output_size = output_size

      self.hidden_layers = nn.ModuleList()
      in_size = input_size
      for h in hidden_sizes:
         self.hidden_layers.append(nn.Linear(in_size, h))
         self.hidden_layers.append(nn.BatchNorm1d(h))
         in_size = h

      self.output_layer = nn.Linear(in_size, output_size)

   def forward(self, x):
      # flatten input along last 2 dimensions
      i = x.reshape(x.size(0), -1)
      
      for idx, layer in enumerate(self.hidden_layers):
         i = layer(i)
         if idx%2 != 0:
            i = F.relu(i)

      output = self.output_layer(i)
      return output.reshape(x.size())