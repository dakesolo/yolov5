import torch
from models.backbone.CSPDarknet import CSPDarknetP5
from models.neck.yolov5_PAFPN import Yolov5PAFPN
from models.head.yolov5_head import Yolov5Head
from models.common.bottleneck import BottleneckSPPF
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection

def main():
    test_2()

def test_3():


    dataDir = 'data/cat'
    dataType = 'all'
    annFile = '{}/annotations/annotations_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)
    # coco = CocoDetection(dataDir, )

    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

def test_2():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CSPDarknetP5()
    model.to(device)
    arr1 = model(torch.rand(1, 3, 640, 640, device=device))
    model2 = Yolov5PAFPN()
    model2.to(device)
    arr2 = model2(arr1[0], arr1[1], arr1[2])
    model3 = Yolov5Head(80)
    model3.to(device)
    arr3 = model3(arr2[0], arr2[1], arr2[2])
    for item in arr3:
        print(item.shape)

def test_1():
    model = CSPDarknetP5()
    arr = model(torch.rand(1, 3, 640, 640))
    for item in arr:
        print(item.shape)

if __name__ == '__main__':
    main()
