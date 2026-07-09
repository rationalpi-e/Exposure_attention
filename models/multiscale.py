"""
Stage 4: MultiScale correction and uniform fusion.

Building a gussian pyramid(we are using gaussian blur and downsizing function.
if we stack the images on top of each other it would look like pyramid)
It also run  retinex correction on each level: the correction weights are not changed only the resolution changed.
upsamples all the images to original resolution and uses plain averaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pyramid import build_gaussian_pyramid
from models.retinex import RetinexExposureNet

class RetinexExposureNet(nn.Module):
    def __init__(self, channels = 32, iters = 8 , levels =2):
        super().__init__()
        '''Common Retinex model is used for each level with changed resolution'''
        self.retinex_net = RetinexExposureNet(channels= channels,iters = iters)
        self.levels = levels   #number of downsampled images excluding  the original image

    def forward(self, x):
        pyramid = build_gaussian_pyramid(x , self.levels)

        corrected , diagnostics = [], []
        for level_img in pyramid:
            out , R , L , L_corrected = self.retinex_net(level_img)
            corrected.append(out)
            diagnostics.append((R , L , level_img))

        full_size = pyramid[0].shape[-2:]
        upsampled = [F.interpolate(c, size= full_size, model = "bilinear", align_corners= False) for c in corrected]
        
        fused = torch.stack(upsampled , dim = 0).mean(dim = 0).clamp(0 , 1)

        return  fused, upsampled , diagnostics