import torch.nn as nn

from models.common.Bottleneck import BottleneckCSP
from models.common.Bottleneck import BottleneckSPPF
from models.common.Conv import Conv


class CSPDarknetP5(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = Conv(3, 64, 6, 2, 2)
        self.cv2 = Conv(64, 128, 3, 2, 1)
        self.cv3 = BottleneckCSP(128, 128, 2)
        self.cv4 = Conv(128, 256, 3, 2, 1)
        self.cv5 = BottleneckCSP(256, 256, 6)

        self.cv6 = Conv(256, 512, 3, 2, 1)
        self.cv7 = BottleneckCSP(512, 512, 9)

        self.cv8 = Conv(512, 1024, 3, 2, 1)
        self.cv9 = BottleneckCSP(1024, 1024, 3)

        self.cv10 = BottleneckSPPF(1024, 1024)

    def forward(self, x):
        output = []
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        x = self.cv4(x)
        x = self.cv5(x)
        output.append(x)
        x = self.cv6(x)
        x = self.cv7(x)
        output.append(x)
        x = self.cv8(x)
        x = self.cv9(x)
        x = self.cv10(x)
        output.append(x)
        return output
