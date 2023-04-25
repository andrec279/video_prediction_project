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
from pretrain import PretrainVICReg
from finetune import FinetuneVideoPredictor


def main():
    VICReg_training = PretrainVICReg()
    VICReg_training.pretrain()
    
    video_predictor_finetuning = FinetuneVideoPredictor(VICReg_training.model_name)
    video_predictor_finetuning.finetune()

if __name__ == '__main__':
    main()