#  this code is heavily borrow from the nnUNet
#  For more details, please refer to https://github.com/MIC-DKFZ/nnUNet


import torch
from torch import nn
import torch.nn.functional as F


softmax_helper = lambda x: F.softmax(x, 1)