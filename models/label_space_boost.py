from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type
import abc

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch import Tensor

from models.base import FittableModule, RidgeModule, RidgeLBFGS, RidgeSGD, FittableSequential, Identity, LogisticRegression
from models.random_feature_representation_boosting import InitialHydra



class HydraLabelBoost(FittableModule):
    def __init__(self,
                 n_estimators: int = 5,
                 n_kernels = 8,
                 n_groups = 64,
                 max_num_channels = 3,
                 hydra_batch_size = 512,
                 l2_reg: float = 10,
                 boost_lr: float = 0.1,
                 ridge_solver: Literal["solve", "LBFGS", "AdamW"] = "solve",
                 lr_ridge = 1,
                 max_iter_ridge = 300,
                 sgd_batch_size = 128,
                 ):
        super(FittableModule, self).__init__()
        self.n_estimators = n_estimators
        self.boost_lr = boost_lr
        
        assert n_estimators > 0
        AdamClass = None
        if ridge_solver.lower()=="solve".lower():
            RidgeClass = RidgeModule
        elif ridge_solver.lower()=="LBFGS".lower():
            RidgeClass = RidgeLBFGS
        elif ridge_solver.lower()=="AdamW".lower():
            RidgeClass = RidgeSGD
            AdamClass = torch.optim.AdamW
        elif ridge_solver.lower()=="Adam".lower():
            RidgeClass = RidgeSGD
            AdamClass = torch.optim.Adam
        else:
            raise RuntimeError(f"Invalid argument for ridge_solver. Given: {ridge_solver}")
        
        self.regs = nn.ModuleList([RidgeClass(l2_reg,
                                                lr=lr_ridge,
                                                max_iter=max_iter_ridge,
                                                batch_size=sgd_batch_size,
                                                AdamClass=AdamClass) 
                                   for _ in range(n_estimators)])
        
        self.feature_layers = nn.ModuleList([InitialHydra(n_kernels, 
                                                          n_groups, 
                                                          max_num_channels, 
                                                          hydra_batch_size) 
                                               for _ in range(n_estimators)])
        
        
    def fit_transform(self, X: Tensor, y: Tensor):
        residuals = y
        lr = 1
        for i in range(self.n_estimators):
            feat_layer = self.feature_layers[i]
            feat_layer.fit(X, residuals)
            features = feat_layer(X)
            reg = self.regs[i]
            reg.fit(features, residuals)
            residuals = residuals - lr * reg(features)
            lr = self.boost_lr
        return y - residuals
        
        
    def forward(self, X: Tensor):
        pred = self.regs[0](self.feature_layers[0](X))
        for i in range(1, self.n_estimators):
            pred += self.boost_lr * self.regs[i](self.feature_layers[i](X))
        return pred
    
    

#TODO i have only changed the name of the class, the code is the same
class HydraReuseLabelBoost(FittableModule):
    def __init__(self,
                 n_estimators: int = 5,
                 n_kernels = 8,
                 n_groups = 64,
                 max_num_channels = 3,
                 hydra_batch_size = 512,
                 l2_reg: float = 10,
                 boost_lr: float = 0.1,
                 ridge_solver: Literal["solve", "LBFGS", "AdamW"] = "solve",
                 lr_ridge = 1,
                 max_iter_ridge = 300,
                 sgd_batch_size = 128,
                 ):
        super(FittableModule, self).__init__()
        self.n_estimators = n_estimators
        self.boost_lr = boost_lr
        
        assert n_estimators > 0
        AdamClass = None
        if ridge_solver.lower()=="solve".lower():
            RidgeClass = RidgeModule
        elif ridge_solver.lower()=="LBFGS".lower():
            RidgeClass = RidgeLBFGS
        elif ridge_solver.lower()=="AdamW".lower():
            RidgeClass = RidgeSGD
            AdamClass = torch.optim.AdamW
        elif ridge_solver.lower()=="Adam".lower():
            RidgeClass = RidgeSGD
            AdamClass = torch.optim.Adam
        else:
            raise RuntimeError(f"Invalid argument for ridge_solver. Given: {ridge_solver}")
        
        self.regs = nn.ModuleList([RidgeClass(l2_reg,
                                                lr=lr_ridge,
                                                max_iter=max_iter_ridge,
                                                batch_size=sgd_batch_size,
                                                AdamClass=AdamClass) 
                                   for _ in range(n_estimators)])
        
        self.feature_layers = nn.ModuleList([InitialHydra(n_kernels, 
                                                          n_groups, 
                                                          max_num_channels, 
                                                          hydra_batch_size) 
                                               for _ in range(n_estimators)])
        
        
    def fit_transform(self, X: Tensor, y: Tensor):
        residuals = y
        lr = 1
        for i in range(self.n_estimators):
            feat_layer = self.feature_layers[i]
            feat_layer.fit(X, residuals)
            features = feat_layer(X)
            reg = self.regs[i]
            reg.fit(features, residuals)
            residuals = residuals - lr * reg(features)
            lr = self.boost_lr
        return y - residuals
        
        
    def forward(self, X: Tensor):
        pred = self.regs[0](self.feature_layers[0](X))
        for i in range(1, self.n_estimators):
            pred += self.boost_lr * self.regs[i](self.feature_layers[i](X))
        return pred