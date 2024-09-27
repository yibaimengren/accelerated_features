import torch
import torch.nn as nn


"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""

from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *
from modules.model import XFeatModel
dev = "cuda" if torch.cuda.is_available() else "cpu"

net = XFeatModel().to(dev)

augmentor = AugmentationPipe(
                                        img_dir = "D:/project/datasets/flowers/flowers/train/daisy",
                                        device = dev, load_dataset = True,
                                        batch_size = 4,
                                        out_resolution = (800, 608),
                                        warp_resolution = (800, 608),
                                        sides_crop = 0.1,
                                        max_num_imgs = 200,
                                        num_test_imgs = 100,
                                        photometric = True,
                                        geometric = True,
                                        reload_step = 500
                                        )

p1s, p2s, H1, H2 = make_batch(augmentor, 0.3)

h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8  # 获取分辨率的长宽分别除以8后的大小
# get_corresponding_pts用来处理synthetic coco数据集。获取变换后图中每8x8区域的左上角坐标，然后通过变换计算原图对应点，把无效的点去除，再把坐标映射回H/8×W/8的坐标系。
_, positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, augmentor, h_coarse, w_coarse)  # 对来自synthetic coco数据集的两个图像生成对应点
positives_c = positives_s_coarse

net.train()

feats1, kpts1, hmap1 = net(p1s)
feats2, kpts2, hmap2 = net(p2s)

for b in range(len(positives_c)):
    # Get positive correspondencies
    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]  # 两个返回值shape应该都是k×2,pts1是原图对应点坐标，pts2是目标图对应点坐标

    # Grab features at corresponding idxs
    m1 = feats1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(1, 0)
    m2 = feats2[b, :, pts2[:, 1].long(), pts2[:, 0].long()].permute(1, 0)

    # grab heatmaps at corresponding idxs
    h1 = hmap1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()]
    h2 = hmap2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()]
    coords1 = net.fine_matcher(torch.cat([m1, m2], dim=-1))  # k×64

    loss_ds, conf = dual_softmax_loss(m1, m2)
    loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)
