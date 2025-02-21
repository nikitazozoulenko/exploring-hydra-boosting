from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from models.base import FittableModule



class NaiveMean(FittableModule):
    def __init__(self):
        super(FittableModule, self).__init__()
        
        
    def fit_transform(self, X: Tensor, y: Tensor):
        self.train_mean = y.mean(dim=0, keepdim=True)
        
    
    def forward(self, X: Tensor):
        return self.train_mean.repeat(X.shape[0], 1)