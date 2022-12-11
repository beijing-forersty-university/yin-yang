import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.convs import ConvModule

"""
    Learning to Resize Images for Computer Vision Tasks
    https://arxiv.org/pdf/2105.04714.pdf
"""


def conv1x1(in_chs, out_chs=16):
    return nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0)


def conv3x3(in_chs, out_chs=16):
    return nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=1, padding=1)


def conv7x7(in_chs, out_chs=16):
    return nn.Conv2d(in_chs, out_chs, kernel_size=7, stride=1, padding=3)


class ResBlock(nn.Module):
    def __init__(self, in_chs, out_chs=16):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(in_chs, out_chs),
            nn.BatchNorm2d(out_chs),
            nn.LeakyReLU(0.2),
            conv3x3(out_chs, out_chs),
            nn.BatchNorm2d(out_chs)
        )

    def forward(self, x):
        identity = x
        out = self.layers(x)
        out += identity
        return out


class Resizer(nn.Module):
    def __init__(self, in_chs, out_size, n_filters=16, n_res_blocks=1, mode='bilinear'):
        super(Resizer, self).__init__()
        self.interpolate_layer = partial(F.interpolate, size=out_size, mode=mode,
                                         align_corners=(
                                             True if mode in ('linear', 'bilinear', 'bicubic', 'trilinear') else None))
        self.conv_layers = nn.Sequential(
            conv7x7(in_chs, n_filters),
            nn.LeakyReLU(0.2),
            conv1x1(n_filters, n_filters),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(n_filters)
        )
        self.residual_layers = nn.Sequential()
        for i in range(n_res_blocks):
            self.residual_layers.add_module(f'res{i}', ResBlock(n_filters, n_filters))
        self.residual_layers.add_module('conv3x3', conv3x3(n_filters, n_filters))
        self.residual_layers.add_module('bn', nn.BatchNorm2d(n_filters))
        self.final_conv = conv7x7(n_filters, in_chs)

    def forward(self, x):
        identity = self.interpolate_layer(x)
        conv_out = self.conv_layers(x)
        conv_out = self.interpolate_layer(conv_out)
        conv_out_identity = conv_out
        res_out = self.residual_layers(conv_out)
        res_out += conv_out_identity
        out = self.final_conv(res_out)
        out += identity
        return out


class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels=2, head_count=2, value_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                    context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention


class MLP(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim=None,
                 out_dim=None,
                 drop=0.,
                 ):
        super(MLP, self).__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoder(nn.Module):
    """
    Encoder layer of transformer
    :param dim: feature dimension
    :param num_heads: number of attention heads
    :param mlp_ratio: hidden layer dimension expand ratio in MLP
    :param dropout_ratio: probability of an element to be zeroed
    :param activation: activation layer type
    :param kv_bias: add bias on key and values
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 dropout_ratio=0.0,
                 activation='GELU',
                 kv_bias=False
                 ):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_dim=dim, hidden_dim=dim * mlp_ratio,
                       drop=dropout_ratio)

    def forward(self, x):
        _x = self.norm1(x)
        x = x + self.mlp(self.norm2(x))
        return x





class TransformerBlock(nn.Module):
    """
    Block of transformer encoder layers. Used in vision task.
    :param in_channels: input channels
    :param out_channels: output channels
    :param num_heads: number of attention heads
    :param num_encoders: number of transformer encoder layers
    :param mlp_ratio: hidden layer dimension expand ratio in MLP
    :param dropout_ratio: probability of an element to be zeroed
    :param activation: activation layer type
    :param kv_bias: add bias on key and values
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads,
                 num_encoders=1,
                 mlp_ratio=1,
                 dropout_ratio=0.,
                 kv_bias=False,
                 activation='GELU'
                 ):
        super(TransformerBlock, self).__init__()
        self.conv = nn.Identity() if in_channels == out_channels else \
            ConvModule(in_channels, out_channels, 1)
        self.linear = nn.Linear(out_channels, out_channels)
        encoders = [TransformerEncoder(out_channels, num_heads, mlp_ratio,
                                       dropout_ratio, activation, kv_bias)
                    for _ in range(num_encoders)]
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        b, _, h, w = x.shape
        x = self.conv(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.encoders(x)
        x = x.permute(1, 2, 0).reshape(b, -1, h, w)
        return x


class Bo(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, batch_size):
        super().__init__()
        self.resizer = Resizer(in_channels, image_size)
        self.attention = EfficientAttention(in_channels, 1, 1, batch_size)
        self.transformer = TransformerBlock(in_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.resizer(x)
        x = self.attention(x)
        x = self.transformer(x)
        return x


if __name__ == "__main__":
    input = torch.randn(4, 3, 224, 224)
    model = Resizer(3, 224)
    model1 = EfficientAttention(3, 1, 1, 64)
    model2 = TransformerBlock(3, 3, 1, 1)
    x = model(input)
    x = model1(x)
    x = model2(x)

    print(x.shape)
