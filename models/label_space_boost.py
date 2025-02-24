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


def get_RidgeClass_and_AdamClass(ridge_solver: str):
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
    return RidgeClass, AdamClass
        


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
                 find_amount_of_say = False,
                 ):
        super(FittableModule, self).__init__()
        self.n_estimators = n_estimators
        self.boost_lr = boost_lr
        self.find_amount_of_say = find_amount_of_say
        
        assert n_estimators > 0

        RidgeClass, AdamClass = get_RidgeClass_and_AdamClass(ridge_solver)
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
        with torch.no_grad():
            residuals = y
            lr = 1
            self.alphas = torch.ones(self.n_estimators)
            for i in range(self.n_estimators):
                #fit regressors to residuals
                feat_layer = self.feature_layers[i]
                feat_layer.fit(X, residuals)
                features = feat_layer(X)
                reg = self.regs[i]
                pred = reg.fit_transform(features, residuals)
                
                #find amount of say
                if self.find_amount_of_say and i>0:
                    self.alphas[i] = pred.T @ residuals / (pred.T @ pred)
        
                #update residuals
                residuals = residuals - lr * self.alphas[i] * pred
                lr = self.boost_lr
            # print("self.alphas", self.alphas)
            return y - residuals
        
        
    def forward(self, X: Tensor):
        with torch.no_grad():
            pred = self.alphas[0] * self.regs[0](self.feature_layers[0](X))
            for i in range(1, self.n_estimators):
                pred += self.boost_lr * self.alphas[i] * self.regs[i](self.feature_layers[i](X))
            return pred
    
    


class HydraLabelReuseBoost(FittableModule):
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
                 find_amount_of_say = False,
                 ):
        super(FittableModule, self).__init__()
        self.n_estimators = n_estimators
        self.boost_lr = boost_lr
        self.find_amount_of_say = find_amount_of_say
        
        assert n_estimators > 0
        RidgeClass, AdamClass = get_RidgeClass_and_AdamClass(ridge_solver)
        self.reg1 = RidgeClass(l2_reg, lr=lr_ridge, max_iter=max_iter_ridge, 
                               batch_size=sgd_batch_size, AdamClass=AdamClass)
        self.reg2 = RidgeClass(l2_reg, lr=lr_ridge, max_iter=max_iter_ridge, 
                               batch_size=sgd_batch_size, AdamClass=AdamClass)
        self.hydra = InitialHydra(n_kernels, 
                                    n_groups, 
                                    max_num_channels, 
                                    hydra_batch_size)
        
        
    def fit_transform(self, X: Tensor, y: Tensor):
        with torch.no_grad():
            feat = self.hydra.fit_transform(X, y)
            pred = self.reg1.fit_transform(feat, y)
            residuals = y - pred
            for i in range(1, self.n_estimators):
                #fit regressor to residuals
                pred = self.reg2.fit_transform(feat, residuals)
                
                #find amount of say
                if self.find_amount_of_say:
                    a = pred.T @ residuals / (pred.T @ pred)
                    # print(f"a[{i}]", a)
                else:
                    a = 1.0

                #update residuals
                residuals = residuals - (self.boost_lr * a) * pred
                
                #update ensamble in self.reg1
                self.reg1.linear
                self.reg1.linear.weight += (self.boost_lr * a) * self.reg2.linear.weight
                self.reg1.b += (self.boost_lr * a) * self.reg2.b

            return y - residuals
    
        
    def forward(self, X: Tensor):
        with torch.no_grad():
            return self.reg1(self.hydra(X))
            