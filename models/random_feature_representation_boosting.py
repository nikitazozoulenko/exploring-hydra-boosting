from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type
import abc

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch import Tensor

from models.sandwiched_least_squares import sandwiched_LS_dense, sandwiched_LS_diag, sandwiched_LS_scalar
from models.base import FittableModule, RidgeModule, RidgeLBFGS, RidgeSGD, FittableSequential, Identity, LogisticRegression




############################################################################
################# Base classes for Random Feature Boosting #################
############################################################################


class RandomFeatureLayer(nn.Module):
    @abc.abstractmethod
    def fit_transform(self, Xt: Tensor, X0: Tensor, y: Tensor) -> Tensor:
        """Takes in both Xt and X0 and y and fits the random 
        feature layer and returns the random features"""

    @abc.abstractmethod
    def forward(self, Xt: Tensor, X0: Tensor) -> Tensor:
        """Takes in both Xt and X0 and returns the random features"""



class GhatBoostingLayer(nn.Module):
    @abc.abstractmethod
    def fit_transform(self, F: Tensor, Xt: Tensor, y: Tensor, top_level_module: FittableModule) -> Tensor:
        """Takes in the random features, resnet representations Xt, target y, 
        and the top level module and fits the boosting layer (functional gradient), 
        and returns the gradient estimates"""

    @abc.abstractmethod
    def forward(self, F: Tensor) -> Tensor:
        """Takes in the random features and returns the gradient estimates"""



class BaseGRFRBoost(FittableModule):
    def __init__(
            self,
            n_layers: int,
            initial_representation: nn.Module,
            top_level_modules: List[FittableModule],
            random_feature_layers: List[RandomFeatureLayer],
            ghat_boosting_layers: List[GhatBoostingLayer],
            boost_lr: float = 0.1,
            #subsample_per_boost: float = 1.0, #TODO 
            train_top_at: List[int] = [0, 5, 10],
            return_features: bool = False,  #logits or features
            ):
        """
        Base class for (Greedy/Gradient) Random Feature Representation Boosting.
        NOTE that we currently store all intermediary classifiers/regressors,
        for simplicity. We only use the topmost one for prediction.
        """
        super(BaseGRFRBoost, self).__init__()
        self.n_layers = n_layers
        self.initial_representation = initial_representation # simple upscale layer, same for everyone
        self.top_level_modules = nn.ModuleList(top_level_modules) # either ridge, or multiclass logistic, or binary logistic
        self.random_feature_layers = nn.ModuleList(random_feature_layers) # random features, same for everyone
        self.ghat_boosting_layers = nn.ModuleList(ghat_boosting_layers) # functional gradient boosting layers
        self.boost_lr = boost_lr
        #self.subsample_per_boost = subsample_per_boost
        self.train_top_at = train_top_at
        self.return_features = return_features




    def fit(self, X: Tensor, y: Tensor):
        """Fits the Random Feature Representation Boosting model.

        Args:
            X (Tensor): Input data, shape (N, in_dim)
            y (Tensor): Targets, shape (N, d) for regression, TODO for classification. TODO check regression dimensions
        """
        self.to(X.device)
        with torch.no_grad():
            # initial Phi_0 
            X0 = X
            X = self.initial_representation.fit_transform(X, y)          

            # Create top level regressor or classifier W_0
            if 0 in self.train_top_at:
                self.top_level_modules[0].fit(X, y)
                print("training W0")
                print("Phi0 shape", X.shape)

            # Feature boost
            for t in range(self.n_layers):
                # Step 1: Create random feature layer
                F = self.random_feature_layers[t].fit_transform(X, X0, y)
                # Step 2: Greedily or Gradient boost to minimize R(W_t, Phi_t + Delta F)
                Ghat = self.ghat_boosting_layers[t].fit_transform(F, X, y, self.top_level_modules[t])
                X = X + self.boost_lr * Ghat
                # Step 3: Learn top level classifier W_t
                if t+1 in self.train_top_at:
                    self.top_level_modules[t+1].fit(X, y)
                else:
                    self.top_level_modules[t+1] = self.top_level_modules[t]

        return self


    def forward(self, X: Tensor) -> Tensor:
        """Forward pass for random feature representation boosting.
        
        Args:
            X (Tensor): Input data shape (N, in_dim)"""
        with torch.no_grad():
            X0 = X
            X = self.initial_representation(X0)
            for randfeat_layer, ghat_layer in zip(self.random_feature_layers, 
                                                             self.ghat_boosting_layers):
                F = randfeat_layer(X, X0)
                Ghat = ghat_layer(F)
                X = X + self.boost_lr * Ghat
            # Top level regressor
            if self.return_features:
                return X
            else:
                return self.top_level_modules[-1](X)
        


############################################################################
#################    Random feature layer       #################
############################################################################


class RandomFeatureLayer(nn.Module, abc.ABC):
    @abc.abstractmethod
    def fit_transform(self, Xt: Tensor, X0: Tensor, y: Tensor) -> Tensor:
        """Takes in both Xt and X0 and y and fits the random 
        feature layer and returns the random features"""

    @abc.abstractmethod
    def forward(self, Xt: Tensor, X0: Tensor) -> Tensor:
        """Takes in both Xt and X0 and returns the random features"""



from models.hydra import HydraGPU, HydraMultivariateGPU, RobustSparseScaler

class HydraLayer(RandomFeatureLayer):
    def __init__(self, 
                n_kernels = 8,
                n_groups = 64,
                max_num_channels = 3,
                hydra_batch_size = 512
            ):
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.max_num_channels = max_num_channels
        self.hydra_batch_size = hydra_batch_size
        self.hydra = None
        self.robust_scaler = None
        super(HydraLayer, self).__init__()


    def fit_transform(self, Xt, X0, y):
        with torch.no_grad():
            N, D, T = X0.size()
            #initialize Hydra
            if D == 1:
                self.hydra = HydraGPU(X0.size(2), self.n_kernels, self.n_groups)
            else:
                self.hydra = HydraMultivariateGPU(X0.size(2), D, self.n_kernels, self.n_groups, self.max_num_channels)
            self.to(X0.device)
            
            #fit robust sparse scaler after forwarding training data
            self.robust_scaler = RobustSparseScaler()
            feats = self._forward_no_scaler(X0)
            feats = self.robust_scaler.fit_transform(feats)
        return feats
    

    def _forward_no_scaler(self, X0: Tensor) -> Tensor:
        with torch.no_grad():
            feats = [
                self.hydra(batch)
                for batch in torch.split(X0, self.hydra_batch_size)
                ]
            feats = torch.concat(feats, dim=0)
        return feats
    
    
    def forward(self, Xt: Tensor, X0: Tensor) -> Tensor:
        feats = self._forward_no_scaler(X0)
        return self.robust_scaler.transform(feats)



#################################
#### Initial feature layer ######
#################################

class InitialHydra(FittableModule):
    def __init__(self,
                 n_kernels = 8,
                 n_groups = 64,
                 max_num_channels = 3,
                 hydra_batch_size = 512,
                 ):
        super(InitialHydra, self).__init__()
        self.hydralayer = HydraLayer(n_kernels, n_groups, max_num_channels, hydra_batch_size) 
        
    def fit_transform(self, X: Tensor, y: Any):
        return self.hydralayer.fit_transform(None, X, y)
        
    def forward(self, X: Tensor):
        return self.hydralayer(None, X)
        
        
        


############################################################################
#################    Ghat layer, Gradient Boosting Regression       ########
############################################################################


class GhatGradientLayerMSE(GhatBoostingLayer):
    def __init__(self,
                 ridge: FittableModule,
                 ):
        super(GhatGradientLayerMSE, self).__init__()
        self.ridge = ridge


    def fit_transform(self, F: Tensor, Xt: Tensor, y: Tensor, auxiliary_reg: RidgeModule) -> Tensor:
        """Fits the functional gradient given features, resnet neurons, and targets,
        and returns the gradient predictions

        Args:
            F (Tensor): Features, shape (N, p)
            Xt (Tensor): ResNet neurons, shape (N, D)
            y (Tensor): Targets, shape (N, d)
            auxiliary_reg (RidgeModule): Auxiliary top level regressor.
        """
        # compute negative gradient, L_2(mu_N) normalized
        N = y.size(0)
        r = y - auxiliary_reg(Xt)
        G = r @ auxiliary_reg.linear.weight
        G = G / torch.norm(G) * N**0.5

        # fit to negative gradient (finding functional direction)
        Ghat = self.ridge.fit_transform(F, G)

        # line search closed form risk minimization of R(W_t, Phi_{t+1})
        self.linesearch = sandwiched_LS_scalar(r, auxiliary_reg.linear.weight.T, Ghat, 1e-5)
        return Ghat * self.linesearch
    

    def forward(self, F: Tensor) -> Tensor:
        return self.linesearch * self.ridge(F)
    
    
    
    



############################################################################
################# Random Feature Representation Boosting for Regression ###################
############################################################################


class HydraBoost(BaseGRFRBoost):
    def __init__(self,
                 n_layers: int = 5,
                 init_n_kernels = 16,
                 init_n_groups = 16,
                 n_kernels = 8,
                 n_groups = 64,
                 max_num_channels = 3,
                 hydra_batch_size = 512,
                 l2_reg: float = 10,
                 l2_ghat: float = 10,
                 boost_lr: float = 0.1,
                 train_top_at: List[int] = [0, 5, 10],
                 return_features: bool = False,  #logits or features
                 ridge_solver: Literal["solve", "LBFGS", "AdamW"] = "solve",
                 lr_ridge = 1,
                 max_iter_ridge = 300,
                 sgd_batch_size = 128,
                 ):
        
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

        initial_layer = InitialHydra(init_n_kernels, init_n_groups, max_num_channels, hydra_batch_size)
        top_level_regs = [RidgeClass(l2_reg, 
                                     lr=lr_ridge, 
                                     max_iter=max_iter_ridge,
                                     batch_size=sgd_batch_size,
                                     AdamClass=AdamClass) 
                          for _ in range(n_layers+1)]
        random_feature_layers = [HydraLayer(n_kernels, n_groups, max_num_channels, hydra_batch_size) for _ in range(n_layers)]
        ghat_boosting_layers = [GhatGradientLayerMSE(RidgeClass(l2_ghat,
                                                                lr=lr_ridge,
                                                                max_iter=max_iter_ridge,
                                                                batch_size=sgd_batch_size,
                                                                AdamClass=AdamClass)) 
                                for _ in range(n_layers)]
        super(HydraBoost, self).__init__(
            n_layers, initial_layer, top_level_regs, random_feature_layers, ghat_boosting_layers, boost_lr, train_top_at, return_features
        )


############################################
############# End Regression ###############
############################################