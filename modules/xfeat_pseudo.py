
"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import numpy as np
import os
import torch
import torch.nn.functional as F

import tqdm

from modules.model import *
from modules.interpolator import InterpolateSparse2d

class XFeat(nn.Module):
	""" 
		Implements the inference module for XFeat. 
		It supports inference for both sparse and semi-dense feature extraction & matching.
	"""

	def __init__(self, weights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat.pt', top_k = 4096, detection_threshold=0.05):
		super().__init__()
		self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.net = XFeatModel().to(self.dev).eval()
		self.top_k = top_k
		self.detection_threshold = detection_threshold

		if weights is not None:
			if isinstance(weights, str):
				print('loading weights from: ' + weights)
				self.net.load_state_dict(torch.load(weights, map_location=self.dev))
			else:
				self.net.load_state_dict(weights)

		self.interpolator = InterpolateSparse2d('bicubic')

		#Try to import LightGlue from Kornia
		self.kornia_available = False
		self.lighterglue = None
		try:
			import kornia
			self.kornia_available=True
		except:
			pass


	@torch.inference_mode()
	def detectAndCompute(self, x, top_k = None, detection_threshold = None):
		"""
			Compute sparse keypoints & descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return:
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
					'scores'       ->   torch.Tensor(N,): keypoint scores
					'descriptors'  ->   torch.Tensor(N, 64): local features
		"""
		#1.使用self.preprocess_tensor调整图像大小，以确保图像可以被32整除，rh1和rw1是图像缩放比例
		#2.使用self.net(x)得到M1:特征描述器64×H/8×W/8，k1:关键点特征图 65×H/8×W/8，H1：heatmap 1×H/8×W/8 似乎是表示对应特征描述器向量能被匹配的概率
		#3.使用self.get_kpts_heatmap(K1)在关键点特征图K1上对每8x8区域进行softmax，然后重新展开为(B,1,H,W)得到K1h
		#4.使用self.NMS(K1h, threshold=detection_threshold, kernel_size=5)获取每5x5区域内最大值大的索引mkpts
		#5.将关键点对应的局部重要性值k1h*关键点对应的heatmap值作为评估分数scores
		#6.根据scores选出得分前top_k的元素，获取每个元素对应关键点坐标、分数
		#7.再选出对应的前top_k个特征，（但是这里对应每个特征点的特征是通过插值得到的)

		M1, K1, H1 = self.net(x) #M1:特征描述器64×H/8×W/8，k1:关键点特征图 65×H/8×W/8，H1：heatmap 1×H/8×W/8 似乎是表示对应特征描述器向量能被匹配的概率
		M1 = F.normalize(M1, dim=1) #在通道维度进行归一化

		#Convert logits to heatmap and extract kpts
		K1h = self.get_kpts_heatmap(K1) #每个8x8区域进行softmax，然后重新展开为(B,1,H,W)
		mkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5) #(B,N,2),N是单张图关键点数量的最大值。
		#Compute reliability scores
		_nearest = InterpolateSparse2d('nearest')
		_bilinear = InterpolateSparse2d('bilinear')

		#它这个评分的原理或者说设想是什么？
		scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1) #B,N  两个线性插值函数分别采样得到关键点对应的①小区域占比值（8X8 softmax）和②可匹配概率

		scores[torch.all(mkpts == 0, dim=-1)] = -1 #B,N 。如果有(0,0)点,则将其设为-1
		#Select top-k features
		idxs = torch.argsort(-scores) #对概率值的负值进行升序排序，返回对应索引 （其实就是按分数进行降序排序的索引）

		#下面四句代码的含义其实就是根据idxs索引选出mkpts中前top_k个关键点及对应scores中的值
		mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)[:, :top_k] #mkpts[...,0]选出所有坐标的x值，gather再按idxs索引对x坐标进行排序，最后选择前k个关键点
		mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)[:, :top_k] #同上
		mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1) # mkpts_x[...,None]为扩展一个维度，最后再将二者在最后一个维度拼起来。
		scores = torch.gather(scores, -1, idxs)[:, :top_k]

		#Interpolate descriptors at kpts positions
		feats = self.interpolator(M1, mkpts, H = _H1, W = _W1) #选出前top_k个特征

		#L2-Normalize
		feats = F.normalize(feats, dim=-1) #归一化

		#Correct kpt scale
		mkpts = mkpts * torch.tensor([rw1,rh1], device=mkpts.device).view(1, 1, -1) #把关键点坐标缩放到原图尺寸

		valid = scores > 0

		return [  
				   {'keypoints': mkpts[b][valid[b]],
					'scores': scores[b][valid[b]],
					'descriptors': feats[b][valid[b]]} for b in range(B) 
			   ]

	@torch.inference_mode()
	def detectAndComputeDense(self, x, top_k = None, multiscale = True):
		"""
			Compute dense *and coarse* descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return: features sorted by their reliability score -- from most to least
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
					'scales'       ->   torch.Tensor(top_k,): extraction scale
					'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
		"""
		if top_k is None: top_k = self.top_k
		if multiscale:
			mkpts, sc, feats = self.extract_dualscale(x, top_k)
		else:
			mkpts, feats = self.extractDense(x, top_k)
			sc = torch.ones(mkpts.shape[:2], device=mkpts.device)

		return {'keypoints': mkpts,
				'descriptors': feats,
				'scales': sc }


	@torch.inference_mode()
	def match_lighterglue(self, d0, d1):
		"""
			Match XFeat sparse features with LightGlue (smaller version) -- currently does NOT support batched inference because of padding, but its possible to implement easily.
			input:
				d0, d1: Dict('keypoints', 'scores, 'descriptors', 'image_size (Width, Height)')
			output:
				mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
				
		"""
		if not self.kornia_available:
			raise RuntimeError('We rely on kornia for LightGlue. Install with: pip install kornia')
		elif self.lighterglue is None:
			from modules.lighterglue import LighterGlue
			self.lighterglue = LighterGlue()

		data = {
				'keypoints0': d0['keypoints'][None, ...],
				'keypoints1': d1['keypoints'][None, ...],
				'descriptors0': d0['descriptors'][None, ...],
				'descriptors1': d1['descriptors'][None, ...],
				'image_size0': torch.tensor(d0['image_size']).to(self.dev)[None, ...],
				'image_size1': torch.tensor(d1['image_size']).to(self.dev)[None, ...]
		}

		#Dict -> log_assignment: [B x M+1 x N+1] matches0: [B x M] matching_scores0: [B x M] matches1: [B x N] matching_scores1: [B x N] matches: List[[Si x 2]], scores: List[[Si]]
		out = self.lighterglue(data)

		idxs = out['matches'][0]

		return d0['keypoints'][idxs[:, 0]].cpu().numpy(), d1['keypoints'][idxs[:, 1]].cpu().numpy()


	@torch.inference_mode()
	def match_xfeat(self, img1, img2, top_k = None, min_cossim = -1):
		"""
			Simple extractor and MNN matcher.
			For simplicity it does not support batched mode due to possibly different number of kpts.
			input:
				img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				top_k -> int: keep best k features
			returns:
				mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
		"""
		if top_k is None: top_k = self.top_k
		img1 = self.parse_input(img1) #检测img的shape的长度是不是4，img的类型是不是tensor
		img2 = self.parse_input(img2)

		out1 = self.detectAndCompute(img1, top_k=top_k)[0] #获取batch索引为0的结果
		out2 = self.detectAndCompute(img2, top_k=top_k)[0]

		idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim )

		return out1['keypoints'][idxs0].cpu().numpy(), out2['keypoints'][idxs1].cpu().numpy()

	@torch.inference_mode()
	def match_xfeat_star(self, im_set1, im_set2, top_k = None):
		"""
			Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
			input:
				im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				top_k -> int: keep best k features
			returns:
				matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
		"""
		if top_k is None: top_k = self.top_k
		im_set1 = self.parse_input(im_set1)
		im_set2 = self.parse_input(im_set2)

		#Compute coarse feats
		out1 = self.detectAndComputeDense(im_set1, top_k=top_k)
		out2 = self.detectAndComputeDense(im_set2, top_k=top_k)

		#Match batches of pairs
		idxs_list = self.batch_match(out1['descriptors'], out2['descriptors'] )
		B = len(im_set1)

		#Refine coarse matches
		#this part is harder to batch, currently iterate
		matches = []
		for b in range(B):
			matches.append(self.refine_matches(out1, out2, matches = idxs_list, batch_idx=b))

		return matches if B > 1 else (matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy())

	def preprocess_tensor(self, x):
		""" Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
		"""确保图像可以被32整除"""
		if isinstance(x, np.ndarray) and len(x.shape) == 3:
			x = torch.tensor(x).permute(2,0,1)[None]
		x = x.to(self.dev).float()

		H, W = x.shape[-2:]
		_H, _W = (H//32) * 32, (W//32) * 32
		rh, rw = H/_H, W/_W #调整比例

		x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False) #双线性插值调整图像大小
		return x, rh, rw

	def get_kpts_heatmap(self, kpts, softmax_temp = 1.0):
		scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
		B, _, H, W = scores.shape
		heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
		heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
		return heatmap

	#非极大值抑制
	def NMS(self, x, threshold = 0.05, kernel_size = 5):
		B, _, H, W = x.shape
		pad=kernel_size//2
		local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x) #选取每5x5区域中的最大值点
		pos = (x == local_max) & (x > threshold) #B,1,H,W  bool型张量。结合上一步一起来看，如果某个点是该5x5区域值最大的点且大于阈值，那么就保留，否则就去除
		# pytorch中坐标系是[y,x],flip(-1)就是掉换成[x,y]
		'''nonzero()返回的是二维张量，指定非0元素在每一维度的索引，比如这里最后一维对应的多个数字分别表示B,C,H,W维度的索引；
			使用切片的方式选出H和W维度的索引，再通过flip将y,x转换为x,y
			最后得到的pos_batched是一个形状为包含B个tensor的列表，每个tensor形状为(count,2)，其中count是指pos中单个样本非零元素的个数
		'''
		pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

		pad_val = max([len(x) for x in pos_batched]) #选所有样本中pos元素个数的最大值，也就是单张图关键点数量的最大值
		pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device) #创建保存关键点索引的张量

		#Pad kpts and build (B, N, 2) tensor
		# 将关键点索引保存到张量中
		for b in range(len(pos_batched)):
			pos[b, :len(pos_batched[b]), :] = pos_batched[b]

		return pos

	@torch.inference_mode()
	def batch_match(self, feats1, feats2, min_cossim = -1):
		B = len(feats1)
		cossim = torch.bmm(feats1, feats2.permute(0,2,1))
		match12 = torch.argmax(cossim, dim=-1)
		match21 = torch.argmax(cossim.permute(0,2,1), dim=-1)

		idx0 = torch.arange(len(match12[0]), device=match12.device)

		batched_matches = []

		for b in range(B):
			mutual = match21[b][match12[b]] == idx0

			if min_cossim > 0:
				cossim_max, _ = cossim[b].max(dim=1)
				good = cossim_max > min_cossim
				idx0_b = idx0[mutual & good]
				idx1_b = match12[b][mutual & good]
			else:
				idx0_b = idx0[mutual]
				idx1_b = match12[b][mutual]

			batched_matches.append((idx0_b, idx1_b))

		return batched_matches

	def subpix_softmax2d(self, heatmaps, temp = 3):
		N, H, W = heatmaps.shape
		heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
		x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
		x = x - (W//2)
		y = y - (H//2)

		coords_x = (x[None, ...] * heatmaps)
		coords_y = (y[None, ...] * heatmaps)
		coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
		coords = coords.sum(1)

		return coords

	def refine_matches(self, d0, d1, matches, batch_idx, fine_conf = 0.25):
		idx0, idx1 = matches[batch_idx]
		feats1 = d0['descriptors'][batch_idx][idx0]
		feats2 = d1['descriptors'][batch_idx][idx1]
		mkpts_0 = d0['keypoints'][batch_idx][idx0]
		mkpts_1 = d1['keypoints'][batch_idx][idx1]
		sc0 = d0['scales'][batch_idx][idx0]

		#Compute fine offsets
		offsets = self.net.fine_matcher(torch.cat([feats1, feats2],dim=-1))
		conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
		offsets = self.subpix_softmax2d(offsets.view(-1,8,8))

		mkpts_0 += offsets* (sc0[:,None]) #*0.9 #* (sc0[:,None])

		mask_good = conf > fine_conf
		mkpts_0 = mkpts_0[mask_good]
		mkpts_1 = mkpts_1[mask_good]

		return torch.cat([mkpts_0, mkpts_1], dim=-1)

	@torch.inference_mode()
	def match(self, feats1, feats2, min_cossim = 0.82):
		cossim = feats1 @ feats2.t() #矩阵乘法，激素按余弦相似度
		cossim_t = feats2 @ feats1.t()#cossim的转置
		print(cossim)
		_, match12 = cossim.max(dim=1) #找出feats1中每一个向量与feats2哪个向量最相似
		_, match21 = cossim_t.max(dim=1) #找出feats2中每个向量与feats1哪个向量最相似
		print(match12)

		idx0 = torch.arange(len(match12), device=match12.device) #创建一个与match12长度相同的索引张量
		print(idx0)
		mutual = match21[match12] == idx0 #如果互相是最相似的，就设为true，这条代码真妙

		#min_cossim>0则筛选大于阈值的索引，否则不筛选
		if min_cossim > 0:
			cossim, _ = cossim.max(dim=1)
			good = cossim > min_cossim
			idx0 = idx0[mutual & good]
			idx1 = match12[mutual & good]
		else:
			idx0 = idx0[mutual]
			idx1 = match12[mutual]

		return idx0, idx1 #返回对应于feats1和feats2中匹配特征的索引

	def create_xy(self, h, w, dev):
		'''返回一个shape为h*w×2的张量，其中包含h×w大小网格的所有坐标'''
		y, x = torch.meshgrid(torch.arange(h, device = dev), 
								torch.arange(w, device = dev), indexing='ij')
		xy = torch.cat([x[..., None],y[..., None]], -1).reshape(-1,2)
		print('xy:',xy.shape)
		return xy

	def extractDense(self, x, top_k = 8_000):
		'''使用net提取特征后，获取heatmap中前top_k大的元素的索引，再根据这个索引选出对应的特征和关键点坐标'''
		if top_k < 1:
			top_k = 100_000_000

		x, rh1, rw1 = self.preprocess_tensor(x)

		M1, K1, H1 = self.net(x) #M1:特征描述器64×H/8×W/8，k1:关键点特征图 65×H/8×W/8，H1：heatmap 1×H/8×W/8 似乎是表示对应特征描述器向量能被匹配的概率
		B, C, _H1, _W1 = M1.shape
		
		xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B,-1,-1) #create返回一个shape为_H1*_W1×2的张量，其中包含h×w大小网格的所有坐标'''
		M1 = M1.permute(0,2,3,1).reshape(B, -1, C) #B, H/8*H/8, 64

		H1 = H1.permute(0,2,3,1).reshape(B, -1) #B, 1*H/8*W/8
		_, top_k = torch.topk(H1, k = min(len(H1[0]), top_k), dim=-1) #返回H1中大小排序前top_k个元素的(大小,索引) top_K:(B,k)

		feats = torch.gather( M1, 1, top_k[...,None].expand(-1, -1, 64)) #选出top_k个特征，每个特征占一行 B, top_k, 64
		mkpts = torch.gather(xy1, 1, top_k[...,None].expand(-1, -1, 2)) #选出top_k个特征对应的在H/8×W/8区域的坐标 B, top_k, 2
		mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1,-1) #B, top_k, 2

		return mkpts, feats #B, top_k, 2；B, top_k, 64;

	def extract_dualscale(self, x, top_k, s1 = 0.6, s2 = 1.3):
		'''将输入进行两种比例的缩放，然后从中提取不同个数的关键点，最后将二者的关节点坐标和特征进行拼接'''
		#1.对输入进行两种比例的线性插值得到x1和x2
		x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode='bilinear')
		x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode='bilinear')
		 
		B, _, _, _ = x.shape
		#使用net提取特征后，获取heatmap中前top_k大的元素的索引，再根据这个索引选出对应的特征和关键点坐标
		mkpts_1, feats_1 = self.extractDense(x1, int(top_k*0.20))#B, top_k, 2；B, top_k, 64;
		mkpts_2, feats_2 = self.extractDense(x2, int(top_k*0.80))

		mkpts = torch.cat([mkpts_1/s1, mkpts_2/s2], dim=1) #把两个不同比例图像计算出的关键点映射回原图位置并将坐标拼到一起 B，top_k*0.2+top_k*0.8, 2 = B, top_k, 2

		sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1/s1)
		sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1/s2)
		sc = torch.cat([sc1, sc2],dim=1)
		feats = torch.cat([feats_1, feats_2], dim=1)

		return mkpts, sc, feats #关键点位置拼接、缩放比例拼接、特征拼接

	def parse_input(self, x):
		if len(x.shape) == 3:
			x = x[None, ...]

		if isinstance(x, np.ndarray):
			x = torch.tensor(x).permute(0,3,1,2)/255

		return x
