import  pytorch_lightning as pl
import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
backbone_name = 'resnet50'


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# backbone = resnet_fpn_backbone(backbone_name=backbone_name, pretrained=True).to(device)
torch.cuda.empty_cache()
x = torch.randn(1, 3, 640, 640).to(device)


class Neck(pl.LightningModule):
    def __init__(self, num_channels, out_channel):
        super(Neck, self).__init__()
        a = out_channel // 4
        b = out_channel // 2

        self.conv1 = nn.Conv2d(num_channels, a, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(a, b, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(b, out_channel, kernel_size=3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(56, 56))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        return x

a = torch.zeros([10, 4])
print(a.shape)


# neck = Neck(3, 256).to(device)
# print(neck(x).shape)
# m = nn.MaxPool2d((1, 1), stride=(2, 2))
# print(m(x).shape)
# print(backbone(x))