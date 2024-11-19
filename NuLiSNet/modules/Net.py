import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from modules.Enhance import enhance_net as enhance
from modules.gamma_new import Global_pred
from modules.detail import detail
from util import save_img


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.enhance = Global_pred()
        self.detail = detail()

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)

        return torch.clamp(image, 1e-8, 1.0)

    def to_one(self, x):
        return (torch.tanh(x) + 1) / 2

    def forward(self, low_l, low_r):
        alpha1, beta1, gamma1 = self.enhance(low_l)
        alpha2, beta2, gamma2 = alpha1, beta1, gamma1
        low_l = low_l.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
        low_r = low_r.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
        b = low_l.shape[0]
        low_l = torch.stack([alpha1[i, :] * (low_l[i, :, :, :] ** gamma1[i, :] + beta1[i, :]) for i in range(b)], dim=0)
        low_l = low_l.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
        low_r = torch.stack([alpha2[i, :] * (low_r[i, :, :, :] ** gamma2[i, :] + beta2[i, :]) for i in range(b)], dim=0)
        low_r = low_r.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
        left, right = self.detail(low_l, low_r)
        left = self.to_one(left)
        right = self.to_one(right)
        return left, right
