from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type
import abc

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch import Tensor

from models.sandwiched_least_squares import sandwiched_LS_dense, sandwiched_LS_diag, sandwiched_LS_scalar
from models.base import FittableModule, RidgeModule, RidgeCVModule, FittableSequential, Identity, LogisticRegression




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
            n_layers: int, # TODO switch
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
        with torch.no_grad():
            # initial Phi_0 
            X0 = X
            X = self.initial_representation.fit_transform(X, y)          

            # Create top level regressor or classifier W_0
            if 0 in self.train_top_at:
                self.top_level_modules[0].fit(X, y)

            # Feature boost
            for t in range(1, self.n_layers+1):
                # Step 1: Create random feature layer
                F = self.random_feature_layers[t].fit_transform(X, X0, y)
                # Step 2: Greedily or Gradient boost to minimize R(W_t, Phi_t + Delta F)
                Ghat = self.ghat_boosting_layers[t].fit_transform(F, X, y, self.top_level_modules[t])
                X = X + self.boost_lr * Ghat
                X = self.batchnorms[t](X)
                # Step 3: Learn top level classifier W_t
                if t in self.train_top_at:
                    self.top_level_modules[t].fit(X, y)
                else:
                    self.top_level_modules[t] = self.top_level_modules[t-1]

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



# TODO TODO TODO i am here
from aeon.transformations.collection.convolution_based import HydraTransformer

class RandFeatLayer(RandomFeatureLayer):
    def __init__(self, 
                    n_kernels = 8,
                    n_groups = 64,
                    max_num_channels = 3,
                 ):
        self.hydra = HydraTransformer(
            n_kernels = 8,
            n_groups = 64,
            max_num_channels = 3,
            ) 
        super(RandFeatLayer, self).__init__()


    def fit(self, Xt: Tensor, X0: Tensor, y: Tensor) -> Tensor:
        """Note that SWIM requires y to be onehot or binary"""
        self.hydra.fit(X0)  
        return self


    def fit_transform(self, Xt, X0, y):
        self.fit(Xt, X0, y)
        return self.forward(Xt, X0)
    

    def forward(self, Xt: Tensor, X0: Tensor) -> Tensor:
        return self.hydra.transform(X0)
        # features = []
        # if self.randfeat_xt_dim > 0:
        #     features.append(self.Ft(Xt))
        # if self.randfeat_x0_dim > 0:
        #     features.append(self.F0(X0))

        # if self.add_features:
        #     return sum(features)
        # else:
        #     return torch.cat(features, dim=1)




############################################################################
#################    Ghat layer, Gradient Boosting Regression       ########
############################################################################


class GhatGradientLayerMSE(GhatBoostingLayer):
    def __init__(self,
                 l2_ghat: float = 0.01,
                 ):
        self.l2_ghat = l2_ghat
        super(GhatGradientLayerMSE, self).__init__()
        self.ridge = RidgeModule(l2_ghat)


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
        G = r @ auxiliary_reg.W.T
        G = G / torch.norm(G) * N**0.5

        # fit to negative gradient (finding functional direction)
        Ghat = self.ridge.fit_transform(F, G)

        # line search closed form risk minimization of R(W_t, Phi_{t+1})
        self.linesearch = sandwiched_LS_scalar(r, auxiliary_reg.W, Ghat, 1e-5)
        return Ghat * self.linesearch
    

    def forward(self, F: Tensor) -> Tensor:
        return self.linesearch * self.ridge(F)


############################################################################
################# Random Feature Representation Boosting for Regression ###################
############################################################################

# TODO i might need to fix how the ridge is fitted, i cant give out_dim at initialization
class GradientRFRBoostRegressor(BaseGRFRBoost):
    def __init__(self,
                 n_layers: int = 5,
                 n_kernels = 8,
                 n_groups = 64,
                 max_num_channels = 3,
                 l2_reg: float = 0.0001,
                 l2_ghat: float = 0.0001,
                 boost_lr: float = 0.1,
                 train_top_at: List[int] = [0, 5, 10],
                 return_features: bool = False,  #logits or features
                 ):

        # TODO 'upscale'
        upscale = Upscale(in_dim, hidden_dim, upscale_type, iid_scale, SWIM_scale,
                          activation if feature_type=="SWIM" else nn.Tanh())

        # top level regressors TODO check with 'out_dim'
        top_level_regs = [RidgeModule(l2_reg) for _ in range(n_layers+1)]

        # random feature layers TODO change to new hydra features
        random_feature_layers = [
            RandFeatLayer(in_dim, hidden_dim, randfeat_xt_dim, randfeat_x0_dim, feature_type,
                          iid_scale, SWIM_scale, activation, add_features)
            for _ in range(n_layers)
        ]

        # ghat boosting layers
        ghat_boosting_layers = [
            GhatGradientLayerMSE(l2_ghat)
            for _ in range(n_layers)
        ]

        super(GradientRFRBoostRegressor, self).__init__(
            n_layers, upscale, top_level_regs, random_feature_layers, ghat_boosting_layers, boost_lr, train_top_at, return_features
        )


############################################
############# End Regression ###############
############################################