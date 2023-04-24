import os
import sys

import cv2
import torch
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
print(BASE_DIR)
sys.path.append(BASE_DIR)

from pycocotools.coco import COCO
from models.yolov5 import Grid, Anchor, box_ciou, Anchor2, Grid2
import time

def test_grid():
    # Grid
    grids = [
        [80, 80],
        [40, 40],
        [20, 20]
    ]
    for grid_rc in grids:
        grid = Grid([34, 89], [640, 640], grid_rc)
        # print(grid.get_grid_wh())
        # print(grid.get_grid_center_xy())
        # print(grid.get_grid_index_center_xy())
        # print(grid.get_grid_index_xy())
        print(grid.get_grid_positives())

def test_grid2():
    # Grid
    grids = [
        [80, 80],
        [40, 40],
        [20, 20]
    ]
    grid = Grid2(torch.tensor([[34, 500], [78, 44], [10, 199], [180, 67]]), torch.tensor([640, 640]), torch.tensor(grids))
    print(grid.get_grid_positives())

def test_anchor():
    anchors = [
        [[68, 69], [154, 91], [143, 162]],  # P3/8
        [[242, 160], [189, 287], [391, 207]],  # P4/16
        [[353, 337], [539, 341], [443, 432]]  # P5/32
    ]
    wh = [68, 120]
    for anchor in anchors:
        # anchor = Anchor([68, 120], [[68, 69], [154, 91], [143, 162]])\
        anchor_object = Anchor(wh, anchor, 4)
        print(anchor_object.get_anchor_positives())

def test_anchor2():
    anchors = [
        [[68, 69], [154, 91], [143, 162]],  # P3/8
        [[242, 160], [189, 287], [391, 207]],  # P4/16
        [[353, 337], [539, 341], [443, 432]]  # P5/32
    ]
    wh = [68, 120]
    anchor_object = Anchor2(torch.tensor(wh), torch.tensor(anchors), torch.tensor([4]))
    print(anchor_object.get_anchor_positives())

def test_iou():
    iou = box_ciou(torch.tensor([5, 20, 30, 40]), torch.tensor([[[10, 20, 30, 40], [20, 40, 12, 55]], [[10, 20, 30, 40], [20, 40, 12, 55]]]))
    print(iou)

    # tttt = torch.tensor([[5, 20, 30, 40], [5, 20, 30, 40]])
    # iou = box_ciou(tttt.unsqueeze(1).unsqueeze(1),
    #                torch.tensor([
    #                    [[[10, 20, 30, 40], [20, 40, 12, 55], [20, 40, 12, 55]], [[10, 20, 30, 40], [20, 40, 12, 55], [20, 40, 12, 55]], [[10, 20, 30, 40], [20, 40, 12, 55], [20, 40, 12, 55]]],
    #                    [[[10, 20, 30, 40], [20, 40, 12, 55], [20, 40, 12, 55]], [[10, 20, 30, 40], [20, 40, 12, 55], [20, 40, 12, 55]], [[10, 20, 30, 40], [20, 40, 12, 55], [20, 40, 12, 55]]]
    #                ]))
    # print(iou)

def test_iou_each():
    # 比较耗性能
    for i in range(100):
        iou = box_ciou(torch.tensor([5, 20, 30, 40]), torch.tensor([10, 20, 30, 40]))
        print(iou)

def test_iou_shape():
    grid_obj = Grid([34, 89], [640, 640], [80, 80])
    anchor_obj = Anchor([68, 120], [[68, 69], [154, 91], [143, 162]], 4)
    grid = grid_obj.get_grid_positives()
    anchor = anchor_obj.get_anchor_positives()
    anchor_grid = []
    for anchor_item in anchor:
        for grid_item in grid:
            anchor_grid.append(anchor_item + grid_item)

    # anchor_grid_tensor =
    # print(anchor_grid_tensor)
    # return
    # print(anchor_grid)

    for item in torch.tensor(anchor_grid)[...,:4]:
        iou = box_ciou(item, torch.tensor([68, 120, 34, 89]))
        iou = torch.where(iou < 0, 0, iou)
        print(iou)
    # print(item)
    # grid = torch.tensor(grid_obj.get_grid_positives())
    # anchor = torch.tensor(anchor_obj.get_anchor_positives())

    # anchor_grid = torch.tensor.
    #for item in grid.chunk(grid.shape[0]):
    #    _anchor = torch.cat((anchor, item.repeat(anchor.size(1), 1)), dim=-1)
    #    print(_anchor)
        # print(item[..., :2])

    # print(anchor)
    # iou = box_ciou(torch.tensor([5, 20, 30, 40]), torch.tensor([10, 20, 30, 40]))
    # print(iou)
def test_cat_grid_anchor():
    a = torch.tensor([
        [[2, 3], [5, 6], [9, 9]],
        [[1, 4], [6, 4], [6, 1]],
        [[6, 2], [8, 2], [5, 1]]
    ])
    a = torch.split(a, 1, dim=0)
    # a1 = a[0].repeat(3, 1, 1)
    # print(a1)
    b = torch.tensor([
        [[45, 69], [77, 91], [88, 162]],
        [[123, 160], [167, 287], [444, 207]],
        [[123, 160], [167, 287], [444, 207]],
    ])
    c1 = torch.cat([b, a[0].repeat(3, 1, 1)], dim=2)
    c2 = torch.cat([b, a[1].repeat(3, 1, 1)], dim=2)
    c3 = torch.cat([b, a[2].repeat(3, 1, 1)], dim=2)
    print(torch.stack([c1, c2, c3], dim=2))


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

if __name__ == '__main__':
    current_time1 = time.time()
    print(get_iou())
    # img_view()
    current_time2 = time.time()
    print(current_time2 - current_time1)