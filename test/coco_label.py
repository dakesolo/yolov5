import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
import torch.utils.data as Data
from models.backbone.CSPDarknet import CSPDarknetP5
from models.head.yolov5_head import Yolov5Head
from models.neck.yolov5_PAFPN import Yolov5PAFPN

if __name__ == '__main__':

    # 模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model1 = CSPDarknetP5()
    model1.to(device)

    model2 = Yolov5PAFPN()
    model2.to(device)

    model3 = Yolov5Head(80)
    model3.to(device)


    dataDir = 'data/cat'
    dataType = 'all'
    annFile = '{}/annotations/annotations_{}.json'.format(dataDir, dataType)
    train_data_transforms = transforms.Compose([
        transforms.Resize([640, 640]),
        transforms.ToTensor()
    ])
    train_dataset = CocoDetection(root=dataDir+"/images", annFile=annFile, transform=train_data_transforms)
    train_data_loader = Data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=2)
    for index, (image, label) in enumerate(train_data_loader):
        arr1 = model1(image.to(device))
        arr2 = model2(arr1[0], arr1[1], arr1[2])
        arr3 = model3(arr2[0], arr2[1], arr2[2])
        for arr in arr3:
            print(arr.shape)
        break