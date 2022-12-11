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


class ChannelWisePooling(torch.nn.Module):
    def __init__(self, pool_type='avg'):
        super(ChannelWisePooling, self).__init__()
        self.pool_type = pool_type

    def forward(self, x):
        # Get the number of channels in the input tensor
        num_channels = x.shape[1]
        if self.pool_type == 'avg':
            # Apply average pooling to each channel independently
            channels = [F.avg_pool2d(x[:, i, :, :].unsqueeze(1), kernel_size=2) for i in range(num_channels)]
        elif self.pool_type == 'max':
            # Apply max pooling to each channel independently
            channels = [F.max_pool2d(x[:, i, :, :].unsqueeze(1), kernel_size=2) for i in range(num_channels)]
        else:
            raise ValueError("Invalid pooling type: {}".format(self.pool_type))

        # Concatenate the pooled channels back together into a single tensor
        return torch.cat(channels, dim=1)


class Neck(torch.nn.Module):
    def __init__(self, num_channels):
        super(Neck, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        return x


class EightTrigrams(nn.Module):
    def __init__(self, image_size, batch_size):
        super().__init__()
        # self.qian = Blocks(image_size, batch_size, [1, 1, 1])
        # self.dui = Blocks(image_size, batch_size, [1, 1, 0])
        # self.li = Blocks(image_size, batch_size, [1, 0, 1])
        # self.zhen = Blocks(image_size, batch_size, [1, 0, 0])
        # self.xun = Blocks(image_size, batch_size, [0, 1, 1])
        # self.kan = Blocks(image_size, batch_size, [0, 1, 0])
        self.gen = Blocks(image_size, batch_size, [0, 0, 1])
        # self.kun = Blocks(image_size, batch_size, [0, 0, 0])
        self.channel = ChannelWisePooling()
        self.neck1 = Neck(3)
        self.neck2 = Neck(3)
        self.neck3 = Neck(3)

    def forward(self, x):
        # x = self.qian(x)
        x = self.gen(x)
        outputs = []
        # # x = self.li(x)
        # # x = self.zhen(x)
        # # x = self.xun(x)
        # # x = self.kan(x)
        # # x = self.gen(x)
        # # x = self.kun(x)

        x = self.channel(x)
        x1 = self.neck1(x)
        x2 = self.neck2(x)
        x3 = self.neck3(x)
        outputs.append(x1)
        outputs.append(x2)
        outputs.append(x3)

        return x

        # return  self.conv1(x)


def EightTrigrams_(cfg):
    loss_cfg = deepcopy(cfg)
    name = loss_cfg.pop('name')

    if name == 'YIN-YANG':
        return EightTrigrams(**loss_cfg)

    else:
        raise NotImplementedError(name)


# import torchviz
# from graphviz import Source

x = torch.randn(32, 3, 640, 640).to("cuda")
model = EightTrigrams(640, 32).to("cuda")

# dot = torchviz.make_dot(model(x))

x = model(x)
# 将可视化图输出为图像文件
# dot.render("new.pdf")
# print("start")
# for o in x:
#     print(o.shape)
