###############################################################################################################
#### The following code is modified from https://github.com/angus924/aaltd2024/blob/main/code/hydra_gpu.py ####
###############################################################################################################

# Angus Dempster, Chang Wei Tan, Lynn Miller
# Navid Mohammadi Foumani, Daniel F Schmidt, and Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets
# AALTD 2024 (ECML PKDD 2024)

# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
# HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification
# https://doi.org/10.1007/s10618-023-00939-3

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

