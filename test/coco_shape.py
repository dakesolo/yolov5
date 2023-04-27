import os
import random
import sys

import cv2
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
print(BASE_DIR)
sys.path.append(BASE_DIR)

import numpy as np
from pycocotools.coco import COCO
from models.yolov5 import Grid, Anchor, box_ciou, Anchor2, Grid2
import time


def get_iou():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 申明用GPU

    # 单张图片多个target
    targets = [[34, 500, 50, 80], [78, 44, 90, 100], [10, 199, 40, 200], [180, 67, 72, 31], [180, 67, 72, 31]]

    # 获取 anchors
    anchors = [
        [[68, 69], [154, 91], [143, 162]],  # P3/8
        [[242, 160], [189, 287], [391, 207]],  # P4/16
        [[353, 337], [539, 341], [443, 432]]  # P5/32
    ]
    anchor_object = Anchor2(torch.tensor(targets, device=device)[..., 2:], torch.tensor(anchors, device=device),
                            torch.tensor([4], device=device))
    anchor_positives = anchor_object.get_anchor_positives()

    # 获取 grids
    grids = [
        [80, 80],
        [40, 40],
        [20, 20]
    ]
    grid_object = Grid2(torch.tensor(targets, device=device)[..., :2], torch.tensor([640, 640], device=device),
                        torch.tensor(grids, device=device))
    grid_positives = grid_object.get_grid_positives()
    grid_positives.repeat(1, 1, 3)

    # [index_r, index_c, x, y, w, h]
    grid_anchor_positives = torch.cat([grid_positives.unsqueeze(2).repeat(1, 1, 3, 1), anchor_positives], dim=3)

    # torch.tensor(targets).unsqueeze(1).unsqueeze(1)维度必须和torch.tensor(grid_anchor_positives[..., 2:])一样才行
    iou = box_ciou(torch.tensor(targets, device=device).unsqueeze(1).unsqueeze(1),
                   grid_anchor_positives[..., 2:])
    iou.clamp_(min=0)
    print(grid_anchor_positives)
    return torch.cat([grid_anchor_positives, iou.unsqueeze(3)], dim=3)


def img_view():
    dataDir = '../data/cat'
    dataType = 'all'
    annFile = '{}/annotations/annotations_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)
    print(coco.dataset.keys())

    # 类别信息
    catIds = coco.getCatIds(catNms=['cat'])
    catInfo = coco.loadCats(catIds)
    print(f"catIds:{catIds}")
    print(f"catcls:{catInfo}")

    # 图像信息
    imgIds = coco.getImgIds(catIds=catIds)
    index = 5  # 随便选择一张图
    imgInfo = coco.loadImgs(imgIds[index])[0]
    print(f"imgIds:{imgIds}")
    print(f"img:{imgInfo}")

    # 标注信息
    annIds = coco.getAnnIds(imgIds=imgInfo['id'], catIds=catIds, iscrowd=None)
    annsInfo = coco.loadAnns(annIds)
    print(f"annIds:{annIds}")
    print(f"annsInfo:{annsInfo}")

    # 显示图像
    # i = io.imread(imgInfo['coco_url'])

    i = cv2.imread(os.path.join(r'../data/cat/images', imgInfo['file_name']))
    plt.imshow(i)
    plt.axis('off')
    coco.showAnns(annsInfo, True)
    plt.show()
    # plt.savefig('testbluelinew.jpg')


def get_label_shape(targets, anchors, grids, size=None):
    """
    :param targets:list
    targets = [[34, 500, 50, 80], [78, 44, 90, 100], [10, 199, 40, 200], [180, 67, 72, 31], [180, 67, 72, 31]]

    :param anchors:list
    anchors = [
        [[68, 69], [154, 91], [143, 162]],  # P3/8
        [[242, 160], [189, 287], [391, 207]],  # P4/16
        [[353, 337], [539, 341], [443, 432]]  # P5/32
    ]

    :param grids:list
    grids = [
        [80, 80],
        [40, 40],
        [20, 20]
    ]

    :param size:list
    grids = [
        640,640
    ]

    :return: tensor
    """
    if size is None:
        size = [640, 640]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 申明用GPU

    anchor_object = Anchor2(torch.tensor(targets, device=device)[..., 2:], torch.tensor(anchors, device=device),
                            torch.tensor([2], device=device))
    anchor_positives = anchor_object.get_anchor_positives()

    grid_object = Grid2(torch.tensor(targets, device=device)[..., :2], torch.tensor(size, device=device),
                        torch.tensor(grids, device=device))
    grid_positives = grid_object.get_grid_positives()
    grid_positives.repeat(1, 1, 3)

    # [index_r, index_c, x, y, w, h]
    grid_anchor_positives = torch.cat([grid_positives.unsqueeze(2).repeat(1, 1, 3, 1), anchor_positives], dim=3)

    # torch.tensor(targets).unsqueeze(1).unsqueeze(1)维度必须和torch.tensor(grid_anchor_positives[..., 2:])一样才行
    iou = box_ciou(torch.tensor(targets, device=device).unsqueeze(1).unsqueeze(1),
                   grid_anchor_positives[..., 2:])
    iou.clamp_(min=0)
    # print(grid_anchor_positives)
    return torch.cat([grid_anchor_positives, iou.unsqueeze(3)], dim=3)


def image_show(img, gt_boxs):
    for gt_box in gt_boxs:
        cv2.rectangle(img, gt_box, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    cv2.imshow('picture', img)
    cv2.waitKey(0)


def suofang(im, gt_boxs, target_height, target_width):
    height, width = im.shape[:2]  # 取彩色图片的长、宽。

    ratio_h = height / target_height
    ration_w = width / target_width

    ratio = max(ratio_h, ration_w)
    gt_boxs = gt_boxs / ratio
    # 缩小图像  resize(...,size)--size(width，height)
    size = (int(width / ratio), int(height / ratio))
    shrink = cv2.resize(im, size, interpolation=cv2.INTER_AREA)  # 双线性插值
    BLACK = [0, 0, 0]

    a = (target_width - int(width / ratio)) / 2
    b = (target_height - int(height / ratio)) / 2
    gt_boxs = gt_boxs + np.array([[a, b, 0, 0]])
    constant = cv2.copyMakeBorder(shrink, int(b), int(b), int(a), int(a), cv2.BORDER_CONSTANT, value=BLACK)
    constant = cv2.resize(constant, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return constant, gt_boxs


def retangle():
    dataDir = '../data/cat'
    dataType = 'all'
    annFile = '{}/annotations/annotations_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)
    print(coco.dataset.keys())

    # 类别信息
    catIds = coco.getCatIds(catNms=['cat'])
    catInfo = coco.loadCats(catIds)
    print(f"catIds:{catIds}")
    print(f"catcls:{catInfo}")

    # 图像信息
    imgIds = coco.getImgIds(catIds=catIds)
    index = random.randint(1, 30)  # 随便选择一张图
    imgInfo = coco.loadImgs(imgIds[index])[0]
    print(f"imgIds:{imgIds}")
    print(f"img:{imgInfo}")

    # 标注信息
    annIds = coco.getAnnIds(imgIds=imgInfo['id'], catIds=catIds, iscrowd=None)
    annsInfo = coco.loadAnns(annIds)
    print(f"annIds:{annIds}")
    print(f"annsInfo:{annsInfo}")

    img = cv2.imread(os.path.join(r'../data/cat/images', imgInfo['file_name']))
    gt_boxs = []
    for anns in annsInfo:
        gt_boxs.append(anns['bbox'])

    new_image_size = [1024, 768]
    new_img, new_gt_boxs = suofang(img, np.array(gt_boxs), new_image_size[0], new_image_size[1])
    # print(new_gt_boxs)
    label_shape = get_label_shape(
        targets=new_gt_boxs,
        anchors=[
            [[68, 69], [154, 91], [143, 162]],  # P3/8
            [[242, 160], [189, 287], [391, 207]],  # P4/16
            [[353, 337], [539, 341], [443, 432]]  # P5/32
        ],
        grids=[
            [80, 80],
            [40, 40],
            [20, 20]
        ],
        size=new_image_size
    )
    image_show(new_img, label_shape[..., 2:6].view(-1, 4).int().tolist())
    print(label_shape)


if __name__ == '__main__':
    current_time1 = time.time()
    # print(get_iou())
    # img_view()
    print(retangle())
    current_time2 = time.time()
    print(current_time2 - current_time1)
