import torch
import torch.nn as nn

from models.common.bottleneck import BottleneckCSP
from models.common.bottleneck import BottleneckSPPF
from models.common.conv import Conv


class Yolov5PAFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv11 = Conv(1024, 512, 1, 1, 0)
        self.up12 = nn.Upsample(scale_factor=2, mode='nearest')
        # cat 13
        self.cv14 = BottleneckCSP(1024, 512, 3, False)
        self.cv15 = Conv(512, 256, 1, 1, 0)
        self.up16 = nn.Upsample(scale_factor=2, mode='nearest')
        # cat 17
        self.cv18 = BottleneckCSP(512, 256, 3)
        self.cv19 = Conv(256, 256, 3, 2, 1)
        # cat 20
        self.cv21 = BottleneckCSP(512, 512, 3, False)
        self.cv22 = Conv(512, 512, 3, 2, 1)
        # cat 23
        self.cv24 = BottleneckCSP(1024, 1024, 3, False)
    def forward(self, x1, x2, x3):
        output = []
        x = self.cv11(x1);x11 = x
        x = self.up12(x)
        x = torch.cat([x2, x], 1)
        x = self.cv14(x)
        x = self.cv15(x);x12 = x
        x = self.up16(x)
        x = torch.cat([x3, x], 1)
        x = self.cv18(x);output.insert(0, x)
        x = self.cv19(x)
        x = torch.cat([x12, x], 1)
        x = self.cv21(x);output.insert(0, x)
        x = self.cv22(x)
        x = torch.cat([x11, x], 1)
        x = self.cv24(x);output.insert(0, x)
        return output
