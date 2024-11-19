import torch
import numpy as np
import util
import os
import cv2
import argparse
from modules.Net import Net


def lowlight(config, file_name, i):

    # 加载 low_l 图像
    low_l = cv2.imread(config.low_l + "\\" + file_name)
    low_l = torch.from_numpy(np.asarray(low_l)).float()
    low_l = low_l.permute(2, 0, 1)
    low_l = low_l.unsqueeze(0).cuda() / 255.0

    # 加载 low_r 图像
    low_r = cv2.imread(config.low_r + "\\" + file_name.replace("left", "right"))
    low_r = torch.from_numpy(np.asarray(low_r)).float()
    low_r = low_r.permute(2, 0, 1)
    low_r = low_r.unsqueeze(0).cuda() / 255.0



    # model_loading
    model = Net().cuda()
    checkpoint = torch.load(config.snapshots_pth)
    model.load_state_dict(checkpoint)

    pre_l, pre_r = model(low_l, low_r)
    pre_l = pre_l.squeeze(0)
    pre_r = pre_r.squeeze(0)
    print("第", i, "张", " file_name: ", file_name)
    util.save_img(pre_l, config.save_left + "\\" + file_name)
    util.save_img(pre_r, config.save_right + "\\" + file_name.replace("left", "right"))


if __name__ == '__main__':
    with torch.no_grad():
        parser = argparse.ArgumentParser()
        parser.add_argument('--low_l', type=str, default=r"D:\mydataset\Uneven_exposure\test_noise\flickr\low\left")
        parser.add_argument('--low_r', type=str, default=r"D:\mydataset\Uneven_exposure\test_noise\flickr\low\right")
        parser.add_argument('--save_left', type=str, default=r"D:\mydataset\Uneven_exposure\pre\flickr\ours\left")
        parser.add_argument('--save_right', type=str, default=r"D:\mydataset\Uneven_exposure\pre\flickr\ours\right")
        parser.add_argument('--cuda', type=str, default="0")
        parser.add_argument('--snapshots_pth', type=str, default="./models/finally.pth")

        config = parser.parse_args()
        file_list = os.listdir(config.low_l)
        len = len(file_list)
        i = 0
        a_total = 0
        b_total = 0
        for file_name in file_list:
            i += 1
            print(i, "/", len)
            lowlight(config, file_name, i)


