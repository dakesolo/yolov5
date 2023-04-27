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

    anchor_object = Anchor2(
        torch.tensor(targets, device=device, dtype=torch.int16)[..., 2:],
        torch.tensor(anchors, device=device, dtype=torch.int16),
        torch.tensor([2], device=device, dtype=torch.int16)
    )
    anchor_positives = anchor_object.get_anchor_positives()

    grid_object = Grid2(
        torch.tensor(targets, device=device, dtype=torch.int16)[..., :2],
        torch.tensor(size, device=device, dtype=torch.int16),
        torch.tensor(grids, device=device, dtype=torch.int16)
    )

    grid_positives = grid_object.get_grid_positives()
    grid_positives.repeat(1, 1, 3)


    # [index_r, index_c, x_center, y_center, x_left, y_left, w, h]
    grid_anchor_positives = torch.cat([grid_positives.unsqueeze(2).repeat(1, 1, 3, 1), anchor_positives], dim=3)
    # torch.tensor(targets).unsqueeze(1).unsqueeze(1)维度必须和torch.tensor(grid_anchor_positives[..., 2:])一样才行
    iou = box_ciou(
        torch.tensor(targets, device=device, dtype=torch.int16).unsqueeze(1).unsqueeze(1),
        # grid_anchor_positives[..., 2:]
        # 这里传入中心点xy，而不是左上角
        torch.cat([grid_anchor_positives[..., 2:4], grid_anchor_positives[..., 6:8]], dim=-1)
    )
    iou.clamp_(min=0)
    return torch.cat([grid_anchor_positives, iou.unsqueeze(3)], dim=3)


def image_show(img, gt_boxs, boxs_iou=None):
    for index, gt_box in enumerate(gt_boxs):
        rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if boxs_iou[index] > 0:
            cv2.putText(img, str(round(boxs_iou[index], 3)), gt_box[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.75, rgb, 2)
        cv2.rectangle(img, gt_box, rgb, 2)
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

def get_shape():
    dataDir = '../data/cat'
    dataType = 'all'
    annFile = '{}/annotations/annotations_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)
    # print(coco.dataset.keys())

    # 类别信息
    catIds = coco.getCatIds(catNms=['cat'])
    catInfo = coco.loadCats(catIds)
    # print(f"catIds:{catIds}")
    # print(f"catcls:{catInfo}")

    # 图像信息
    imgIds = coco.getImgIds(catIds=catIds)
    # index = random.randint(1, 30)  # 随便选择一张图
    index = 0
    imgInfo = coco.loadImgs(imgIds[index])[0]
    # print(f"imgIds:{imgIds}")
    # print(f"img:{imgInfo}")

    # 标注信息
    annIds = coco.getAnnIds(imgIds=imgInfo['id'], catIds=catIds, iscrowd=None)
    annsInfo = coco.loadAnns(annIds)
    # print(f"annIds:{annIds}")
    print(f"annsInfo:{annsInfo}")

    img = cv2.imread(os.path.join(r'../data/cat/images', imgInfo['file_name']))
    gt_boxs = []
    for anns in annsInfo:
        gt_boxs.append([
            anns['bbox'][0] + anns['bbox'][2] / 2,
            anns['bbox'][1] + anns['bbox'][3] / 2,
            anns['bbox'][2],
            anns['bbox'][3]
        ])

    new_image_size = [640, 640]
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
    print(label_shape[..., 8:].view(-1).tolist())
    image_show(new_img, label_shape[..., 4:8].view(-1, 4).int().tolist(), label_shape[..., 8:].view(-1).tolist())


if __name__ == '__main__':
    current_time1 = time.time()
    # print(get_iou())
    # img_view()
    print(get_shape())
    current_time2 = time.time()
    print(current_time2 - current_time1)
