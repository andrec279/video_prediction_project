import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image
import numpy as np

import models as m
import utils as u

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# load in data from first 11 frames for each video

path = 'Dataset_Student/unlabeled/video_'
to_tensor = transforms.ToTensor()

data_x = []
for video in tqdm(range(2000, 3000)): # eventually use 15000
    # get the image path for each frame of each video
    img_path = path + str(video)
    video = []
    for frame in range(0, 11):
        fin_path = os.path.join(img_path, 'image_' + str(frame)+ '.png')
        # open image and convert to tensor
        img = Image.open(fin_path)
        img = to_tensor(img)
        video.append(img)
    video = torch.stack(video, dim=0)
    data_x.append(video)
data_x = torch.stack(data_x, dim=0)
data_x = data_x.reshape(data_x.size(0), data_x.size(1)*data_x.size(2), data_x.size(3), data_x.size(4))
print('Final Size of data_x tensor:', data_x.size()) # (number of videos, number of frames*color channels, image width, image height)

# load in data from 22nd (last) frame for each video
data_y = []
for video in tqdm(range(2000, 3000)): # eventually use 15000
    # get the image path for the 22nd frame of each video
    img_path = path + str(video)
    fin_path = os.path.join(img_path, 'image_21.png')
    # open image and convert to tensor
    img = Image.open(fin_path)
    img = to_tensor(img).unsqueeze(0)
    data_y.append(img)
data_y = torch.stack(data_y, dim=0)
data_y = data_y.reshape(data_y.size(0), data_y.size(1)*data_y.size(2), data_y.size(3), data_y.size(4))
print('Final Size of data_y tensor:', data_y.size()) # (number of videos, number of frames*color channels, image width, image height)

# batch size for training
batch_size = 64

# feed into dataloader
data = u.MyDataset(data_x, data_y)
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

# set parameters for patch embeddings
image_size = (160, 240)
patch_size = 20
embed_dim = 64
expander_out = 256

model = m.VICReg(image_size, patch_size, embed_dim, expander_out=expander_out).to(device)

# pretrain
import time

num_epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

t0 = time.time()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = []
    for data in tqdm(dataloader, desc=f'epoch {epoch}'):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = model(x, y)
        
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} loss: {np.mean(epoch_loss)}')
    
print(f'Finished training in {(time.time()-t0)/60} minutes')