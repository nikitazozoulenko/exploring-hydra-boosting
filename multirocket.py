from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from base import TimeseriesFeatureExtractor



def four_multirocket_pooling(X: Tensor) -> Tensor:
    """4 pooling mechanisms used by MultiRocket

    Args:
        X (Tensor): Features of shape (N, D, T)

    Returns:
        Tensor: Pooled features of shape (N, D, 4)
    """
    N,D,T = X.shape

    #ppv (proportion of positive values)
    pos = (X>0)
    ppv = torch.mean(pos, dim=-1, dtype=X.dtype)

    #mpv (mean of positive values)
    mpv = torch.mean(X*pos, dim=-1)
    
    #mipv (mean of indices of positive values)
    ones = torch.ones_like(X)
    arange = torch.cumsum(ones, dim=-1)
    mipv = torch.mean(arange*pos, dim=-1) / T

    #lspv (longest stretch of positive values)
    X_pad = F.pad(X, (0,1), "constant", 0)
    pos = (X_pad>0).int()
    t = ((pos[..., :-1] - pos[..., 1:])==1).int()
    cumsum = torch.cumsum(pos[..., :-1], dim=-1)
    te = t*cumsum
    te_cummax = torch.cummax(te, axis=-1).values
    lspv = torch.max(te_cummax.diff(axis=-1), axis=-1).values / T

    return torch.concat([ppv, mpv, mipv, lspv], dim=-1)



class MultiRocketFeatures(nn.Module):
    def __init__(self, D, T, n_features, kernel_length=9):
        super().__init__()

        max_exponent = np.floor(np.log2((T - 1) / (kernel_length- 1))).astype(np.int64)
        max_exponent = max(max_exponent, 0)
        dilations = 2**np.arange(max_exponent + 1)
        n_kernels_per_dilation = int(n_features / len(dilations) / 4)
        n_kernels_per_dilation = max(n_kernels_per_dilation, 1)

        self.n_kernels = n_kernels_per_dilation * len(dilations)
        self.convs = nn.ModuleList(
            [nn.Conv1d(
                in_channels=D, 
                out_channels=n_kernels_per_dilation, 
                kernel_size=kernel_length,
                dilation = dilation,
                padding = "same",
                bias=True) 
             for dilation in dilations]
        )

    
    def init_biases(self, X: Tensor, chunk_size: int=1000):
        """Initializes the biases of the convolutional layers,
        using the quantiles of the data. Assumes the data to
        be shuffled.

        WARNING: Slow even for 10 points, 10 000 kernels, T=113. (1.3s elapsed)

        Args:
            X (Tensor): Shape (N, D, T).
            chunk_size (int): Batch size for computations
        """
        with torch.no_grad():
            # first set the biases to zero
            for conv in self.convs:
                conv.bias.data.zero_()

            #obtain output
            out_per_conv = [conv(X.float()) for conv in self.convs]

            #initalize bias using random quantiles
            for out, conv in zip(out_per_conv, self.convs):
                #out: (N, n_kernels_per_dilation, T)
                n_ker_per_dil = out.shape[1]
                quantiles = 0.8 * torch.rand(n_ker_per_dil, device=X.device) + 0.1
                q = torch.quantile(out.permute(0,2,1).reshape(-1, n_ker_per_dil), quantiles, dim=0)
                conv.bias.data = torch.diag(q)

        return self

    
    def forward(self, x): # can be made more memory efficient, as we don't need to store all the intermediate results
        # x: (N, D, T)
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = four_multirocket_pooling(x)
        return x
    

class MultiRocketOwn(TimeseriesFeatureExtractor):
    def __init__(self, n_features, max_batch=512):
        super().__init__(max_batch)
        self.n_features = n_features


    def fit(self, X: Tensor, y=None):
        """ creates the MultiRocketFeatures object and initializes it"""
        N, T, D = X.shape
        self.model = MultiRocketFeatures(D, T, self.n_features).to(X.device)
        self.model.init_biases(X[0:1].permute(0,2,1))
        return self

            
    def _batched_transform(
            self,
            X:Tensor,
        ):
        return self.model(X.permute(0,2,1))