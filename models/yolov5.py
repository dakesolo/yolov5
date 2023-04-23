import math
import torch


class Grid:
    def __init__(self, gt_xy, img_wh, grid_count_rc):
        self.gt_xy = gt_xy
        self.img_wh = img_wh
        self.grid_count_rc = grid_count_rc
        self.grid_wh = [img_wh[0] / self.grid_count_rc[0], img_wh[1] / self.grid_count_rc[1]]
        self.grid_center_xy = [self.grid_wh[0] / 2, self.grid_wh[1] / 2]
        self.grid_index_rc = [gt_xy[0] // self.grid_wh[0], gt_xy[1] // self.grid_wh[1]]
        self.grid_index_xy = [self.grid_index_rc[0] * self.grid_wh[0], self.grid_index_rc[1] * self.grid_wh[1]]
        self.grid_index_center_xy = [self.grid_index_xy[0] + self.grid_center_xy[0], self.grid_index_xy[1] + self.grid_center_xy[1]]


    def get_grid_wh(self):
        return self.grid_wh
    def get_grid_center_xy(self):
        return self.grid_center_xy
    def get_grid_index_rc(self):
        return self.grid_index_rc
    def get_grid_index_xy(self):
        return self.grid_index_xy
    def get_grid_index_center_xy(self):
        return self.grid_index_center_xy

    def get_grid_positives(self):
        grid_list = [self.create_extend_grid(0, 0)]
        if  self.gt_xy[0] > self.grid_index_center_xy[0] and self.gt_xy[1] > self.grid_index_center_xy[1]:
            if self.grid_index_rc[1] < self.grid_count_rc[1]:
                grid_list.append(self.create_extend_grid(0, 1))
            if self.grid_index_rc[0] < self.grid_count_rc[0]:
                grid_list.append(self.create_extend_grid(1, 0))

        elif self.gt_xy[0] > self.grid_index_center_xy[0] and self.gt_xy[1] < self.grid_index_center_xy[1]:
            if self.grid_index_rc[1] < self.grid_count_rc[1]:
                grid_list.append(self.create_extend_grid(0, 1))
            if self.grid_index_rc[0] > 0:
                grid_list.append(self.create_extend_grid(-1, 0))

        elif self.gt_xy[0] < self.grid_index_center_xy[0] and self.gt_xy[1] > self.grid_index_center_xy[1]:
            if self.grid_index_rc[1] > 0:
                grid_list.append(self.create_extend_grid(0, -1))
            if self.grid_index_rc[0] < self.grid_index_rc[0]:
                grid_list.append(self.create_extend_grid(1, 0))

        elif self.gt_xy[0] < self.grid_index_center_xy[0] and self.gt_xy[1] < self.grid_index_center_xy[1]:
            if self.grid_index_rc[1] > 0:
                grid_list.append(self.create_extend_grid(0, -1))
            if self.grid_index_rc[0] > 0:
                grid_list.append(self.create_extend_grid(-1, 0))

        return grid_list

    def create_extend_grid(self, arrow_r, arrow_c):
        r = self.grid_index_rc[0] + arrow_r
        c = self.grid_index_rc[1] + arrow_c
        _x = r * self.grid_wh[0]
        _y = c * self.grid_wh[1]
        x = _x + self.grid_center_xy[0]
        y = _y + self.grid_center_xy[1]
        return [
            x, y, r, c, _x, _y
        ]

class Grid2:
    def __init__(self, gt_xy, img_wh, grid_count_rc):
        self.gt_x = gt_xy[..., 0:1]
        self.gt_y = gt_xy[..., 1:2]
        self.img_w = img_wh[..., 0]
        self.img_h = img_wh[..., 1]
        self.grid_count_r = grid_count_rc[..., 0]
        self.grid_count_c = grid_count_rc[..., 1]
        self.grid_w = torch.div(self.img_w, self.grid_count_r)
        self.grid_h = torch.div(self.img_h, self.grid_count_c)
        self.grid_center_x = self.grid_w / 2
        self.grid_center_y = self.grid_h / 2

        # print(torch.div(torch.tensor([ [34.],  [78.],  [10.], [180.]]), self.grid_w, rounding_mode='floor'))
        self.grid_index_c = torch.div(self.gt_x, self.grid_w, rounding_mode='floor')
        self.grid_index_r = torch.div(self.gt_y, self.grid_h, rounding_mode='floor')
        self.grid_index_x = self.grid_index_c * self.grid_w
        self.grid_index_y = self.grid_index_r * self.grid_h
        self.grid_index_center_x = self.grid_index_x + self.grid_center_x
        self.grid_index_center_y = self.grid_index_y + self.grid_center_y

    def get_grid_wh(self):
        return self.grid_wh
    def get_grid_center_xy(self):
        return self.grid_center_xy
    def get_grid_index_rc(self):
        return self.grid_index_rc
    def get_grid_index_xy(self):
        return self.grid_index_xy
    def get_grid_index_center_xy(self):
        return self.grid_index_center_xy

    def get_grid_positives(self):
        # print(self.grid_index_center_y)
        # print(self.gt_y)
        # print(self.grid_index_r)

        grid_index_extend_c = torch.gt(self.grid_index_center_x, self.gt_x)
        grid_index_extend_c = torch.where(grid_index_extend_c == True, self.grid_index_c + 1, self.grid_index_c - 1)
        # print(grid_index_extend_c)


        grid_index_extend_r = torch.gt(self.grid_index_center_y, self.gt_y)
        grid_index_extend_r = torch.where(grid_index_extend_r == True, self.grid_index_r + 1, self.grid_index_r - 1)
        grid_positives = torch.stack([grid_index_extend_r, grid_index_extend_c, grid_index_extend_c * self.grid_w, grid_index_extend_r * self.grid_h], dim=2)
        return torch.where(grid_positives > 0, grid_positives, 0)
class Anchor:
    def __init__(self, gt_wh, anchor_list, anchor_thr = 4):
        self.gt_wh = gt_wh
        self.anchor_list = anchor_list
        self.anchor_thr = anchor_thr

    def get_anchor_positives(self):
        _anchor_list = []
        for anchor in self.anchor_list:
            rw = self.gt_wh[0] / anchor[0]
            rh = self.gt_wh[1] / anchor[1]
            rw = max(rw, 1 / rw)
            rh = max(rh, 1 / rh)
            r = max(rw, rh)
            if r < self.anchor_thr:
                _anchor_list.append(anchor)
        return _anchor_list

class Anchor2:
    def __init__(self, gt_wh, anchor, anchor_thr = torch.Tensor([4])):
        self.gt_wh = gt_wh
        self.anchor = anchor
        self.anchor_thr = anchor_thr

    def get_anchor_positives(self):
        gt_w = self.gt_wh[..., :1].unsqueeze(2).unsqueeze(3)
        gt_h = self.gt_wh[..., 1:].unsqueeze(2).unsqueeze(3)
        rw = torch.div(gt_w, self.anchor[...,:1])
        rh = torch.div(gt_h, self.anchor[...,1:])
        rw = torch.max(rw, 1 / rw)
        rh = torch.max(rh, 1 / rh)
        r = torch.max(rw, rh)
        i = torch.where(r < self.anchor_thr, 1, 0)
        # print(i)
        # i = i.unsqueeze(3)
        # print(i)
        # i = i.repeat( 1, 1, 2)
        return self.anchor * i



def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # print(b1_mins)
    # print(b2_mins)

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))

    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
        b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
    ciou = ciou - alpha * v
    return ciou


