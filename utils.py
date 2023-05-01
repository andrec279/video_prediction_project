import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


def batch_cov(x):
    # data input of size (batch_size, num_samples, embed_size)
    b, n, m = x.size()

    # Flatten the tensor to size (b*n, m)
    flat_tensor = x.view(b*n, -1)

    # Compute the covariance matrix (b*n, b*n)
    cov_matrix = torch.matmul(flat_tensor, flat_tensor.transpose(1, 0))

    return cov_matrix


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    indices = torch.triu_indices(n, n, offset=1)
    flattened_tensor = x.flatten()
    off_diagonal_tensor = flattened_tensor[indices[0]*(n) + indices[1]]
    return off_diagonal_tensor


def create_pretrain_dataloader(folder, image_size, batch_size):
    to_tensor = transforms.ToTensor()
    list_video_paths = [folder + v for v in os.listdir(folder)]

    if os.path.exists('data_x_pretrain.pt') and os.path.exists('data_y_pretrain.pt'):
        print('loading pretraining data from disk...')
        data_x = torch.load('data_x_pretrain.pt')
        data_y = torch.load('data_y_pretrain.pt')
        print('done')
    
    else:
        data_y = torch.empty(len(list_video_paths), 3, image_size[0], image_size[1])
        data_x = torch.empty(len(list_video_paths), 11, 3, image_size[0], image_size[1]) # first 11 frames of each video (B, n_frames, C, H, W)
        
        i = 0
        for vid_path in tqdm(list_video_paths, desc='creating data_x and data_y for pretraining'): # eventually use 15000
            # get the image path for each frame of each video
            for frame in range(0, 11):
                frame_path = os.path.join(vid_path, 'image_' + str(frame)+ '.png')
                input_img = to_tensor(Image.open(frame_path))
                data_x[i][frame] = input_img
            
            target_frame_path = os.path.join(vid_path, 'image_21.png')
            target_img = to_tensor(Image.open(target_frame_path))
            data_y[i] = target_img
            
        data_x = data_x.reshape(data_x.size(0), data_x.size(1)*data_x.size(2), data_x.size(3), data_x.size(4))
        torch.save(data_x, 'data_x_pretrain.pt')
        torch.save(data_y, 'data_y_pretrain.pt')
        print('Final Size of data_x tensor:', data_x.size()) # (number of videos, number of frames*color channels, image width, image height)

    # feed into dataloader
    data = PretrainDataset(data_x, data_y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    return dataloader


def create_finetune_dataloader(folder, image_size, batch_size, train_or_val):

    to_tensor = transforms.ToTensor()
    list_video_paths = [folder + v for v in os.listdir(folder)]

    if os.path.exists(f'data_x_finetune_{train_or_val}.pt') and os.path.exists(f'data_y_finetune_{train_or_val}.pt'):
        print(f'loading {train_or_val} finetuning data from disk...')
        data_x = torch.load(f'data_x_finetune_{train_or_val}.pt')    
        data_y = torch.load(f'data_y_finetune_{train_or_val}.pt')
        print('done')
    
    else:
        data_y = torch.empty(len(list_video_paths), image_size[0], image_size[1]) # 22nd segmentation mask for each video (B, H, W)
        data_x = torch.empty(len(list_video_paths), 11, 3, image_size[0], image_size[1]) # first 11 frames of each video (B, n_frames, C, H, W)

        i = 0
        for video_path in tqdm(list_video_paths, desc=f'creating {train_or_val} data_x and data_y for finetuning'):
            last_mask = torch.Tensor(np.load(f"{video_path}/mask.npy"))[-1]
            last_mask[last_mask>48] = 0
            data_y[i] = last_mask

            for frame in range(0, 11):
                fin_path = os.path.join(video_path, 'image_' + str(frame)+ '.png')
                img = Image.open(fin_path)
                img = to_tensor(img)
                data_x[i][frame] = img

            i += 1

        data_x = data_x.reshape(data_x.size(0), data_x.size(1)*data_x.size(2), data_x.size(3), data_x.size(4))
        torch.save(data_x, f'data_x_finetune_{train_or_val}.pt')
        torch.save(data_y, f'data_y_finetune_{train_or_val}.pt')

    video_prediction_data = VideoPredictionDataset(data_x, data_y)
    video_prediction_dataloader = torch.utils.data.DataLoader(video_prediction_data, batch_size=batch_size, shuffle=True)

    return video_prediction_dataloader

def create_hidden_dataloader(folder, image_size, batch_size, hidden_set):

    to_tensor = transforms.ToTensor()
    list_video_paths = [folder + v for v in os.listdir(folder)]

    if os.path.exists(f'data_submission_{hidden_set}.pt'):
        print(f'loading {hidden_set} finetuning data from disk...')
        data_x = torch.load(f'data_submission_{hidden_set}.pt')    
        print('done')
    
    else:
        data_x = torch.empty(len(list_video_paths), 11, 3, image_size[0], image_size[1]) # all 11 frames of each video (B, n_frames, C, H, W)

        i = 0
        for video_path in tqdm(list_video_paths, desc=f'creating {hidden_set} data_x for submission'):
            for frame in range(0, 11):
                fin_path = os.path.join(video_path, 'image_' + str(frame)+ '.png')
                img = Image.open(fin_path)
                img = to_tensor(img)
                data_x[i][frame] = img

            i += 1

        data_x = data_x.reshape(data_x.size(0), data_x.size(1)*data_x.size(2), data_x.size(3), data_x.size(4))
        torch.save(data_x, f'data_submission_{hidden_set}.pt')

    video_prediction_data = VideoPredictionDataset_hidden(data_x)
    video_prediction_dataloader = torch.utils.data.DataLoader(video_prediction_data, batch_size=batch_size, shuffle=False)

    return video_prediction_dataloader


class PretrainDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        input_data = self.inputs[idx].to(torch.float32)
        label_data = self.labels[idx].to(torch.float32)
        return input_data, label_data
    

class VideoPredictionDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label_data = self.labels[idx].to(torch.int64)
        return input_data, label_data
    

class VideoPredictionDataset_hidden(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        return input_data