from losses import SetCriterion
from losses.matcher import build_matcher
from models.yang import *
from models.yin import UNet
import torch
from torch import nn, optim
import pytorch_lightning as pl

from models.head.concat_feature_maps import concat_feature_maps
from collections import OrderedDict
from models.head import DyHead


class Blocks(pl.LightningModule):
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


class ChannelWisePooling(pl.LightningModule):
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


class Neck(pl.LightningModule):
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


class EightTrigrams(pl.LightningModule):
    def __init__(self, image_size, batch_size, num_classes):
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
        self.neck = Neck(3, 256, 56, 56)
        self.neck1 = Neck(3, 256, 28, 28)
        self.neck2 = Neck(3, 256, 14, 14)
        self.neck3 = Neck(3, 256, 7, 7)
        self.concat_layer = concat_feature_maps()
        self.pool = nn.MaxPool2d((1, 1), stride=(2, 2))
        self.class_embed = nn.Linear(256, num_classes + 1)
        self.bbox_embed = MLP2(256, 256, 4, 3)
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        dec_layers = 6
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        self.loss = SetCriterion(num_classes,matcher=build_matcher(), weight_dict= weight_dict, eos_coef=0.1, losses=['labels', 'boxes', 'cardinality'])

    def forward(self, x):
        # x = self.qian(x)
        x = self.gen(x)

        # # x = self.li(x)
        # # x = self.zhen(x)
        # # x = self.xun(x)
        # # x = self.kan(x)
        # # x = self.gen(x)
        # # x = self.kun(x)

        x = self.channel(x)

        # return torch.cat(x1, x2, x3)

        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # coverage_weights = prop_weights * batch_size
        img, ann = batch
        x = torch.stack(img)
        x = self.forward(x)
        d = OrderedDict()
        d["0"] = self.neck(x)
        d["1"] = self.neck1(x)
        d["2"] = self.neck2(x)
        d["3"] = self.neck3(x)
        d['pool'] = self.pool(self.neck3(x))
        F = self.concat_layer(d)
        # print('Shape: {}'.format(F.shape))
        L, S, C = F.shape[1:]
        num_blocks = 6  # This is the baseline given in the paper
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        full_head = DyHead(num_blocks, L, S, C).to(device)
        F = full_head(F)

        outputs_class = self.class_embed(F)
        outputs_coord = self.bbox_embed(F).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        losses = self.loss(out, ann)

        self.log("train_loss", losses)
        loss = sum(loss for loss in losses.values())

        return {'loss': loss, 'log': losses, 'progress_bar': losses}


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # def training_step(self, batch, batch_idx):
    #     # training_step defines the train loop. It is independent of forward
    #     x, y = batch
    #     # print(x.shape)
    #     x = self.gen(x)
    #     x = self.channel(x)
    #     x1 = self.neck1(x)
    #     x2 = self.neck2(x)
    #     x3 = self.neck3(x)
    #     z = torch.cat(x1, x2, x3)
    #     loss = F.mse_loss(z, x)
    #     self.log("train_loss", loss)
    #     return loss
    #
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer


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
