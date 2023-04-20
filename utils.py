import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

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

class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32)
        label_data = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_data, label_data