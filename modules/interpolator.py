"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions.
        根据提供的坐标，从特征x中获取对应特征
    """
    def __init__(self, mode = 'bicubic', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        #把索引坐标转化为相对于原图分辨率的百分比坐标，0~1范围。
        #2*：0~2范围；-1：-1~1范围。
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype) #把原来的索引坐标根据原图长宽转化为-1~1范围，并扩展维度为B,N,1,2
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False) #B,C,N,1  grid_sample:https://blog.csdn.net/qq_40968179/article/details/128093033
        return x.permute(0,2,3,1).squeeze(-2)#B, N, C