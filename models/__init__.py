from copy import deepcopy

from models.yang import *
from models.yin import UNet
import torch
from torch import nn


class Blocks(nn.Module):
    def __init__(self, image_size, batch_size, trigram):
        super().__init__()
        self.bo = nn.Sequential()
        for i in trigram:
            if i == 0:
                self.yin = Bo(in_channels=3, out_channels=3, image_size=image_size, batch_size=batch_size)
                self.bo.add_module("yin" + str(i), self.yin)
            elif i == 1:
                self.yang = UNet(n_channels=3)
                self.bo.add_module("yang" + str(i), self.yang)
            else:
                raise Exception("您出现在了宇宙边缘，黑洞即将吞噬！！！")

    def forward(self, x):
        return self.bo(x)


class EightTrigrams(nn.Module):
    def __init__(self, image_size, batch_size):
        super().__init__()
        self.qian = Blocks(image_size, batch_size, [1, 1, 1])
        self.dui = Blocks(image_size, batch_size, [1, 1, 0])
        self.li = Blocks(image_size, batch_size, [1, 0, 1])
        self.zhen = Blocks(image_size, batch_size, [1, 0, 0])
        self.xun = Blocks(image_size, batch_size, [0, 1, 1])
        self.kan = Blocks(image_size, batch_size, [0, 1, 0])
        self.gen = Blocks(image_size, batch_size, [0, 0, 1])
        self.kun = Blocks(image_size, batch_size, [0, 0, 0])

    def forward(self, x):
        x = self.qian(x)
        x = self.dui(x)
        x = self.li(x)
        x = self.zhen(x)
        x = self.xun(x)
        x = self.kan(x)
        x = self.gen(x)
        x = self.kun(x)

        return x


def EightTrigrams_(cfg):

    loss_cfg = deepcopy(cfg)
    name = loss_cfg.pop('name')

    if name == 'YIN-YANG':
        return EightTrigrams(**loss_cfg)

    else:
        raise NotImplementedError(name)


# x = torch.randn(4, 3, 128, 128).to(device)
# model = EightTrigrams(128, 4).to(device)
# print(model(x))
