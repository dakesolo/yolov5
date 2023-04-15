import torch

from models.backbone.CSPDarknet import CSPDarknetP5
from models.common.Bottleneck import BottleneckSPPF


def main():
    model = CSPDarknetP5()
    arr = model(torch.rand(1, 3, 640, 640))
    for item in arr:
        print(item.shape)



if __name__ == '__main__':
    main()
