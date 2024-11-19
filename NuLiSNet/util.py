import numpy as np
import torch
import cv2
import random

from torch import nn


def save_img(img, img_path):
    enhanced_image = torch.squeeze(img, 0)
    enhanced_image = enhanced_image.permute(1, 2, 0)
    enhanced_image = np.asarray(enhanced_image.cpu())
    enhanced_image = enhanced_image * 255.0
    cv2.imwrite(img_path, enhanced_image)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1
                 ):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)
        return self.TVLoss_weight * (h_tv.sum() / count_h + w_tv.sum() / count_w) / batch_size


class L_TV_low(nn.Module):
    def __init__(self, TVLoss_weight=5):
        super(L_TV_low, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow(((x[:, :, 1:, :] - x[:, :, :h_x - 1, :])), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size
