from models.yang import *
from models.yin import UNet
import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU: ', torch.cuda.get_device_name(0))

else:
    device = torch.device("cpu")
    print('No GPU available')


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
        # x1 = self.kun(x)
        x1 = self.qian(x)
        x2 = self.dui(x)
        x3 = self.li(x)
        x4 = self.zhen(x)
        x5 = self.xun(x)
        x6 = self.kan(x)
        x7 = self.gen(x)
        x8 = self.kun(x)
        return x1, x2, x3, x4, x5, x6, x7, x8


# x = torch.randn(4, 3, 128, 128).to(device)
# model = EightTrigrams(128, 4).to(device)
# print(model(x))
