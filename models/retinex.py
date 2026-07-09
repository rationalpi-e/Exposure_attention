import torch
import torch.nn as nn
import torch.nn.functional as F

from models.curve_net import le_curve

class IlluminationNet(nn.Module):
    def __init__(self, channels = 32):
        super().__init__()
        self.conv1 = nn.conv2d(3, channels , 3, padding=1)
        self.conv2 = nn.conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.conv2d(channels, 1 , 3, padding=1)
        self.relu = nn.ReLU(inplace= True)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        L = torch.sigmoid(self.conv3(h))
        return L.clamp(min = 1e-2)   #non-0 illumination  because R/L = can't be infinity
    

class IllumCurveNet(nn.Module):
    def __init__(self, channels = 32):
        super().__init__()
        self.conv1 = nn.conv2d(1 , channels , 3 , padding = 1)
        self.conv2 = nn.conv2d(channels , channels , 3, padding = 1)
        self.conv3 = nn.conv2d(channels, 1,3, padding = 1)
        self.relu = nn.reLU(inplace= True)

    def forward(self, L):
        h = self.relu(self.conv1(L))
        h = self.relu(self.conv2(h))
        alpha = torch.tanh(self.conv3(h))
        return alpha
    

class RetinexExposureNet(nn.Module):
    def __init__(self, channels = 32, iters = 8):
        super().__init()
        self.illum_net= IlluminationNet(channels)
        self.curve_net = IllumCurveNet(channels)
        self.iters = iters

    def forward(self, x):
        L = self.illum_net(x)
        R = x / (L + 1e-4)
        alpha = self.curve_net(L)
        L_corrected = le_curve(L , alpha , self.iters)
        out = (R * L_corrected).clamp(0 , 1)
        return R , L , L_corrected , out
    
def retinex_recon_loss(R , L , I):
    return F.l1_loss(R*L , I)

def structure_aware_smoothness(L , I , eps = 1e-3):
    gray = I.mean(1, keepdim = True)

    dLx = L[..., :, 1:] - L[..., :, :-1]
    dIx = gray[..., :, 1:] - gray[..., :, :-1]
    wx = torch.exp(-dIx.abs() / eps)
 
    dLy = L[..., 1:, :] - L[..., :-1, :]
    dIy = gray[..., 1:, :] - gray[..., :-1, :]
    wy = torch.exp(-dIy.abs() / eps)

    return (wx * dLx.abs()).mean() + (wy * dLy.abs()).mean()