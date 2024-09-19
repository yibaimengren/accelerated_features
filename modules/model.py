"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),#适用于训练深度神经网络时，加速训练过程、提高模型的泛化能力。
									  nn.ReLU(inplace = True),
									)

	def forward(self, x):
	  return self.layer(x)

class XFeatModel(nn.Module):
	"""
	   Implementation of architecture described in 
	   "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	"""

	def __init__(self):
		super().__init__()
		self.norm = nn.InstanceNorm2d(1) #对单个样本的单个通道执行归一化

		########### ⬇️ CNN Backbone & Heads ⬇️ ###########

		self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),#1×H/4×W/4
			  						 nn.Conv2d (1, 24, 1, stride = 1, padding=0) )#24×H/4×W/4

		self.block1 = nn.Sequential(
										BasicLayer( 1,  4, stride=1),#4×H×W
										BasicLayer( 4,  8, stride=2),#8×H/2×W/2
										BasicLayer( 8,  8, stride=1),#8×H/2×W/2
										BasicLayer( 8, 24, stride=2),#24×H/4×W/4  【这里确实是分辨率减半的同时三倍化通道数】
									)

		self.block2 = nn.Sequential(
										BasicLayer(24, 24, stride=1),#24×H/4×W/4
										BasicLayer(24, 24, stride=1),#24×H/4×W/4
									 )

		self.block3 = nn.Sequential(
										BasicLayer(24, 64, stride=2),#64×H/8×W/8 【这里就不是刚好三倍，而约等于是2.7倍】
										BasicLayer(64, 64, stride=1),#64×H/8×W/8
										BasicLayer(64, 64, 1, padding=0),#64×H/8×W/8
									 )
		self.block4 = nn.Sequential(
										BasicLayer(64, 64, stride=2),#64×H/16×W/16 【这里分辨率减半，但是通道数不变】
										BasicLayer(64, 64, stride=1),#64×H/16×W/16
										BasicLayer(64, 64, stride=1),#64×H/16×W/16
									 )

		self.block5 = nn.Sequential(
										BasicLayer( 64, 128, stride=2),#128×H/32×W/32【这里分辨率减半，通道数两倍】
										BasicLayer(128, 128, stride=1),#128×H/32×W/32
										BasicLayer(128, 128, stride=1),#128×H/32×W/32
										BasicLayer(128,  64, 1, padding=0),#64×H/32×W/32
									 )

		self.block_fusion =  nn.Sequential(
										BasicLayer(64, 64, stride=1),#64×H/8×W/8
										BasicLayer(64, 64, stride=1),#64×H/8×W/8
										nn.Conv2d (64, 64, 1, padding=0)#64×H/8×W/8
									 )

		self.heatmap_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),#64×H/8×W/8
										BasicLayer(64, 64, 1, padding=0),#64×H/8×W/8
										nn.Conv2d (64, 1, 1),#1×H/8×W/8
										nn.Sigmoid() #sigmid是每个是0-1，softmax是多个加起来是1
									)


		self.keypoint_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),#64×H/8×W/8
										BasicLayer(64, 64, 1, padding=0),#64×H/8×W/8
										BasicLayer(64, 64, 1, padding=0),#64×H/8×W/8
										nn.Conv2d (64, 65, 1),#65×H/8×W/8
									)


  		########### ⬇️ Fine Matcher MLP ⬇️ ###########

		self.fine_matcher =  nn.Sequential(
											nn.Linear(128, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 64),
										)

	def _unfold2d(self, x, ws = 2):
		"""
			Unfolds tensor in 2D with desired ws (window size) and concat the channels
		"""
		B, C, H, W = x.shape
		#x.unfold(dim, size, step) ，dim：int，表示需要展开的维度(可以理解为窗口的方向)；size：int，表示滑动窗口大小；step：int，表示滑动窗口的步长
		x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
			.reshape(B, C, H//ws, W//ws, ws**2)
		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


	def forward(self, x):
		"""
			input:
				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
			return:
				feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
				keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
				heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

		"""
		#dont backprop through normalization
		#把输入图像调整为单通道图像
		with torch.no_grad():
			x = x.mean(dim=1, keepdim = True)
			x = self.norm(x) #对单个样本的单个通道执行归一化

		#main backbone
		x1 = self.block1(x)#24×H/4×W/4 初步提高通道数同时降低分辨率
		x2 = self.block2(x1 + self.skip1(x)) #24×H/4×W/4 残差链接+进一步卷积处理
		x3 = self.block3(x2) #64×H/8×W/8 提高通道数（接近3倍，但不是）且降低分辨率为原有一半
		x4 = self.block4(x3) #64×H/16×W/16 分辨率再次减半
		x5 = self.block5(x4) #64×H/32×W/32 提高通道数为128且分辨率减半然后再降低通道数回到64

		#pyramid fusion
		x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear') #把x4的分辨率调整到和x3一致
		x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear') #把x5的分辨率调整到和x3一致
		feats = self.block_fusion( x3 + x4 + x5 ) #64×H/8×W/8 特征描述器

		#heads
		heatmap = self.heatmap_head(feats) # Reliability map #1×H/8×W/8 似乎是表示对应特征描述器向量能被匹配的概率

		keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits #unfold:64×H/8×W/8，keypoint_head:65×H/8×W/8

		return feats, keypoints, heatmap
