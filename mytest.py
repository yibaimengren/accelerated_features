import torch
import torch.nn as nn


"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""

import numpy as np
import os
import torch
import tqdm

from modules.xfeat import XFeat

os.environ['CUDA_VISIBLE_DEVICES'] = 'CPU' #Force CPU, comment for GPU

xfeat = XFeat()

#Random input
x = torch.randn(2,3,480,640)
x1 = torch.tensor([[1, 1, 1, 1],
                   [2, 2, 2, 2],
                   [3, 3, 3, 3],
                   [4, 4, 4, 4]])
x2 = torch.tensor([[5, 5, 5, 5],
                   [6, 6, 6, 6],
                   [7, 7, 7, 7],
                   [8, 8, 8, 8]])
#Simple inference with batch = 1
xfeat.detectAndCompute(x)