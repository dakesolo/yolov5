import torch
import torch.nn as nn

from models.common.bottleneck import BottleneckCSP
from models.common.bottleneck import BottleneckSPPF
from models.common.conv import Conv


class Yolov5Head(nn.Module):
    def __init__(self, cat_n):
        super().__init__()
        self.cv25 = Conv(1024, (5 + cat_n) * 3, 1, 1, 0)
        self.cv26 = Conv(512, (5 + cat_n) * 3, 1, 1, 0)
        self.cv27 = Conv(256, (5 + cat_n) * 3, 1, 1, 0)
    def forward(self, x1, x2, x3):
        output = []
        output.insert(0, self.cv25(x1))
        output.insert(0, self.cv26(x2))
        output.insert(0, self.cv27(x3))
        return output
