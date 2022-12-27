import math

from losses.fcos import FCOSLoss
from losses.postprocessor import FCOSPostprocessor
from models.yang import *
from models.yin import UNet
import torch
from torch import nn, optim

from models.head.concat_feature_maps import concat_feature_maps
from collections import OrderedDict
from models.head import DyHead
from torch.utils.data import DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale


def init_conv_std(module, std=0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class FCOSHead(nn.Module):
    def __init__(self, in_channel, n_class, n_conv, prior):
        super().__init__()

        n_class = n_class - 1

        cls_tower = []
        bbox_tower = []

        for i in range(n_conv):
            cls_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channel))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            bbox_tower.append(nn.GroupNorm(32, in_channel))
            bbox_tower.append(nn.ReLU())

        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)

        self.cls_pred = nn.Conv2d(in_channel, n_class, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channel, 4, 3, padding=1)
        self.center_pred = nn.Conv2d(in_channel, 1, 3, padding=1)

        self.apply(init_conv_std)

        prior_bias = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_pred.bias, prior_bias)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def forward(self, input):
        logits = []
        bboxes = []
        centers = []

        for feat, scale in zip(input, self.scales):
            cls_out = self.cls_tower(feat)

            logits.append(self.cls_pred(cls_out))
            centers.append(self.center_pred(cls_out))

            bbox_out = self.bbox_tower(feat)
            bbox_out = torch.exp(scale(self.bbox_pred(bbox_out)))

            bboxes.append(bbox_out)

        return logits, bboxes, centers


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


class ChannelWisePooling(nn.Module):
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


class Neck(nn.Module):
    def __init__(self, num_channels, out_channel, H, W):
        super(Neck, self).__init__()
        a = out_channel // 4
        b = out_channel // 2

        self.conv1 = nn.Conv2d(num_channels, a, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(a, b, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(b, out_channel, kernel_size=3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(H, W))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        return x


class EightTrigrams(nn.Module):
    def __init__(self, image_size, batch_size, num_classes):
        super().__init__()
        self.batch_size = batch_size
        # self.qian = Blocks(image_size, batch_size, [1, 1, 1])
        # self.dui = Blocks(image_size, batch_size, [1, 1, 0])
        # self.li = Blocks(image_size, batch_size, [1, 0, 1])
        # self.zhen = Blocks(image_size, batch_size, [1, 0, 0])
        # self.xun = Blocks(image_size, batch_size, [0, 1, 1])
        # self.kan = Blocks(image_size, batch_size, [0, 1, 0])
        self.gen = Blocks(image_size, batch_size, [0, 0, 1])
        self.yin = Bo(in_channels=3, out_channels=3, image_size=image_size, batch_size=batch_size)
        # self.kun = Blocks(image_size, batch_size, [0, 0, 0])
        self.channel = ChannelWisePooling()
        self.neck0 = Neck(3, 256, 80, 80)
        self.neck1 = Neck(3, 256, 40, 40)
        self.neck2 = Neck(3, 256, 20, 20)
        self.neck3 = Neck(3, 256, 10, 10)
        self.neck4 = Neck(3, 256, 5, 5)

        self.head = FCOSHead(
            256, num_classes, 4, 0.01
        )
        self.postprocessor = FCOSPostprocessor(
            0.05,
            1000,
            0.6,
            100,
            0,
            num_classes,
        )
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.loss = FCOSLoss(
            [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]],
            2.0,
            0.25,
            'giou',
            True,
            self.fpn_strides,
            1.5,
        )

    def forward(self, x, targets, image_sizes=None):
        # x = self.qian(x)
        # x = torch.stack(x)
        x = self.gen(x)
        # x = self.gen(x)
        # # x = self.li(x)
        # # x = self.zhen(x)
        # # x = self.xun(x)
        # # x = self.kan(x)
        # # x = self.gen(x)
        # # x = self.kun(x)

        x = self.channel(x)
        x0 = self.neck0(x)
        x1 = self.neck1(x)
        x2 = self.neck2(x)
        x3 = self.neck3(x)
        x4 = self.neck4(x)
        features = [x0, x1, x2, x3, x4]
        cls_pred, box_pred, center_pred = self.head(features)
        # print(cls_pred, box_pred, center_pred)
        location = self.compute_location(features)

        if self.training:
            loss_cls, loss_box, loss_center = self.loss(
                location, cls_pred, box_pred, center_pred, targets
            )
            losses = {
                'loss_cls': loss_cls,
                'loss_box': loss_box,
                'loss_center': loss_center,
            }

            return None, losses

        else:
            boxes = self.postprocessor(
                location, cls_pred, box_pred, center_pred, image_sizes
            )

            return boxes, None

    def compute_location(self, features):
        locations = []

        for i, feat in enumerate(features):
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, self.fpn_strides[i], feat.device
            )
            locations.append(location_per_level)

        return locations

    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2

        return location


if __name__ == '__main__':
    # import torchviz
    # from graphviz import Source
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    x = torch.randn(1, 3, 640, 640).to(device)
    model = EightTrigrams(640, 1).to(device)
    # # model.eval()
    # print([params for params in model.parameters()])
    # # dot = torchviz.make_dot(model(x))
    #
    x = model(x)
    print(x.shape)
    # 将可视化图输出为图像文件
    # dot.render("new.pdf")
    # print("start")
    # for o in x:
    #     print(o.shape)
