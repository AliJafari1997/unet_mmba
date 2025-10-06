import torch
import torch.nn as nn
import torch.nn.functional as F

class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x


class MMBA(nn.Module):
    def __init__(self, in_c):
        '''in_channels of encoder part'''
        super().__init__()
        self.conv1 = nn.Conv2d(in_c*2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_foreground = nn.Conv2d(in_c, in_c//2, kernel_size=3, padding=1)
        self.conv_background = nn.Conv2d(in_c, in_c//2, kernel_size=3, padding=1)
        self.conv_boundary = nn.Conv2d(in_c, in_c//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3*(in_c//2), in_c, kernel_size=3, padding=1)
        self.se = Squeeze_Excitation(in_c)

    def forward(self, e, d):
        # e.g. encoder = torch.randn(2, 16, 32, 32) , decoder = torch.randn(2, 32, 16, 16)
        d = self.conv1(d)
        d = self.sigmoid(d)
        d = self.upsample(d)

        foreground = d * e
        background = e * (1-d)
        boundary = e * (1 - (2 * torch.abs(d-0.5)))

        foreground = self.conv_foreground(foreground)
        background = self.conv_background(background)
        boundary = self.conv_boundary(boundary)

        out = torch.cat((foreground, background, boundary), dim=1)   # 3/2
        out = self.conv3(out)
        out = self.se(out)
        out = e + out

        return out


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, is_mmba=True):
        super().__init__()
        self.is_mmba = is_mmba
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
        self.mmba = MMBA(out_c)
    def forward(self, inputs, skip):
        if self.mmba:
            skip = self.mmba(skip, inputs)
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class deep_supervision(nn.Module):
    def __init__(self, in_c1, in_c2, in_c3, in_c4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c1, 1, kernel_size=1)
        self.upsample1 = nn.Upsample(scale_factor=8)
        self.conv2 = nn.Conv2d(in_c2, 1, kernel_size=1)
        self.upsample2 = nn.Upsample(scale_factor=4)
        self.conv3 = nn.Conv2d(in_c3, 1, kernel_size=1)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(in_c4, 1, kernel_size=1)
        self.conv5 = nn.Conv2d(4, 1, kernel_size=1)
    
    def forward(self, inp1, inp2, inp3, inp4):
        out1 = self.conv1(inp1)
        out1 = self.upsample1(out1)

        out2 = self.conv2(inp2)
        out2 = self.upsample2(out2)

        out3 = self.conv3(inp3)
        out3 = self.upsample3(out3)

        out4 = self.conv4(inp4)

        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.conv5(out)
        return out


class build_model(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512, is_mmba=False)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.deep_supervision = deep_supervision(512, 256, 128, 64)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.deep_supervision(d1, d2, d3, d4)

        return outputs
