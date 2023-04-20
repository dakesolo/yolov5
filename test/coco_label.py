import os
import sys

import torch
from torch import nn

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from models.yolov5 import Grid, Anchor, box_ciou


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

def test_anchor():
    anchors = [
        [[68, 69], [154, 91], [143, 162]],  # P3/8
        [[242, 160], [189, 287], [391, 207]],  # P4/16
        [[353, 337], [539, 341], [443, 432]]  # P5/32
    ]
    for anchor in anchors:
        # anchor = Anchor([68, 120], [[68, 69], [154, 91], [143, 162]])\
        anchor_object = Anchor([68, 120], anchor, 4)
        print(anchor_object.get_anchor_positives())

def test_iou():
    iou = box_ciou(torch.Tensor([5, 20, 30, 40]), torch.Tensor([10, 20, 30, 40]))
    print(iou)

def test_iou_shape():
    grid_obj = Grid([34, 89], [640, 640], [80, 80])
    anchor_obj = Anchor([68, 120], [[68, 69], [154, 91], [143, 162]], 2)
    grid = torch.Tensor(grid_obj.get_grid_positives())
    anchor = torch.Tensor(anchor_obj.get_anchor_positives())
    print(grid)
    print(anchor)
    # iou = box_ciou(torch.Tensor([5, 20, 30, 40]), torch.Tensor([10, 20, 30, 40]))
    # print(iou)

if __name__ == '__main__':
    test_iou_shape()
    # test_anchor()
    # test_iou()

