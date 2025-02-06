from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import torch
from torch import Tensor
from aeon.transformations.collection.convolution_based import Rocket, MultiRocket, MiniRocket

from kernels.sig_trunc import cumsum_shift1
from base import TimeseriesFeatureExtractor

##########################################
####        ROCKET wrappers           ####
##########################################

class AbstractRocketWrapper(TimeseriesFeatureExtractor):
    def __init__(
            self,
            rocket, #ie Rocket(n_features)
            max_batch: int,
        ):
        super().__init__(max_batch)
        self.rocket = rocket


    def fit(self, X: Tensor, y=None): #shape (N, T, D)
        self.rocket.fit(X.cpu().numpy().transpose(0,2,1))
        return self


    def _batched_transform(self, X: Tensor) -> Tensor: # shape (N, T, D)
        X_np = X.cpu().numpy().transpose(0,2,1)
        features = self.rocket.transform(X_np)
        return torch.from_numpy(features).to(X.dtype).to(X.device)
    


class RocketWrapper(AbstractRocketWrapper):
    def __init__(
            self,
            n_features: int = 3000,
            max_batch: int = 10000,
        ):
        super().__init__(
            Rocket(max(1, n_features//2)),
            max_batch
            )



class MiniRocketWrapper(AbstractRocketWrapper):
    def __init__(
            self,
            n_features: int = 3000,
            max_batch: int = 10000,
        ):
        super().__init__(
            MiniRocket(max(4, n_features)), 
            max_batch
            )



class MultiRocketWrapper(AbstractRocketWrapper):
    def __init__(
            self,
            n_features: int = 3000,
            max_batch: int = 10000,
        ):
        """
        Wrapper for the MultiRocketTransform from the aeon library.
        Original paper: https://link.springer.com/article/10.1007/s10618-022-00844-1

        Args:
            max_batch (int): Maximum batch size for computations.
            n_features (int):  Number of random features.
        """
        super().__init__(
            MultiRocket(max(84, n_features//8)), 
            max_batch
            )
