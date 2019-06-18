# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2019/5/23
# @Last Modified by:   Hobo
# @Last Modified time: 2019/5/23 11:51
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

trainTransform = transforms.Compose([
    # 把图片 转换成形状为[C,H,W]，范围取值的英文[0, 1.0]的torch.FloadTensor
    transforms.ToTensor(),
    # 进行归一化  [-1,1]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def compute_accuracy(prob_cls, gt_cls,threshold=0.6):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    # ＃获取> = 0的掩码元素，只有0和1可以影响检测损失
    mask = torch.ge(gt_cls,0)
    # 选出 pos neg 样本 只包含 1 和 0
    valid_gt_cls = torch.masked_select(gt_cls,mask)
    valid_prob_cls = torch.masked_select(prob_cls,mask)

    # print("valid_gt_cls= ",valid_gt_cls)
    # print("valid_prob_cls= ", valid_prob_cls.size())

    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])

    # 这里选择是查看概率大于等于threshold，其实是选择 pos 预测的正确率，pos(正确)/(neg+pos)
    prob_ones = torch.ge(valid_prob_cls,threshold).float()
    right_ones = torch.eq(prob_ones,valid_gt_cls).float()

    # 计算向量的平均值包含1和0, 1 表示正确分类，0表示不正确
    return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))


def compute_recoll(prob_cls, gt_cls,threshold=0.6):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    # input pos >= 1
    mask = torch.ge(gt_cls, 1)
    # get valid element
    valid_gt_cls = torch.masked_select(gt_cls, mask)
    valid_prob_cls = torch.masked_select(prob_cls, mask)

    prob_ones = torch.ge(valid_prob_cls, threshold).float()
    # 正样本预测准确数
    right_ones = torch.eq(prob_ones, valid_gt_cls).float()

    return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(torch.sum(valid_gt_cls)))


def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    for b in bounding_boxes:

        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

    for p in facial_landmarks:
        for i in range(5):
            cv2.circle(img, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)

    cv2.imshow('image', img)
    cv2.imwrite("image.jpg", img)
    cv2.waitKey(10000)

def gpu_show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    for b in bounding_boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

    for p in facial_landmarks:
        for i in range(5):
            cv2.circle(img, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)

    cv2.imshow('image', img)
    cv2.imwrite("image.jpg",img)
    cv2.waitKey(10000)


def IOU(box, boxes):
    '''裁剪的box和图片所有人脸box的iou值'''
    # box面积
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    # boxes面积,[n,]
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # 重叠部左上分坐标
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    # 右下
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 重叠部分长宽
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    # 重叠部分面积
    inter = w * h
    return inter / (box_area + area - inter + 1e-10)

def gpu_iou(box1, box2):
    # 假设box1维度为[N,4]   box2维度为[M,4]
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(  # 左上角的点
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2]->[N,1,2]->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2]->[1,M,2]->[N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # 两个box没有重叠区域
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # (N,)
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # (M,)
    area1 = area1.unsqueeze(1).expand(N, M)  # (N,M)
    area2 = area2.unsqueeze(0).expand(N, M)  # (N,M)

    iou = inter / (area1 + area2 - inter)
    return iou

def nms(boxes, overlap_threshold=0.5, mode='union'):
    """ Pure Python NMS baseline. """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # argsort()默认从小到大排序，取反后就是从大到小 () 得到从小到大的索引
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # x1[i]和x1[order[1:]]逐位进行比较,选择最大值.
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if mode is 'min':
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]

    return keep

def gpu_nms(boxes, scores, overlap_threshold=0.5, mode='union'):
    # bboxes维度为[N,4]，scores维度为[N,], 均为tensor
    # torch.numel() 表示一个张量总元素的个数
    # torch.clamp(min, max) 设置上下限
    # tensor.item() 把tensor元素取出作为numpy数字
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)  # 降序排列

    keep = []
    while order.numel() > 0:  # torch.numel()返回张量元素个数
        if order.numel() == 1:  # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()  # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        '''
        torch.clamp(input, min, max, out=None) → Tensor
              | min, if x_i < min
        y_i = | x_i, if min <= x_i <= max
              | max, if x_i > max
        '''
        # 张量每个元素的限制到不小于min (取最最大值) (每个框的左上角<x1,y1>)

        xx1 = x1[order[1:]].clamp(min=x1[i])  # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        # 张量每个元素的限制到不大于max (取最最小值) (每个框的左上角<x2,y2>)
        xx2 = x2[order[1:]].clamp(max=x2[i])  # xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]

        if mode is 'min':
            iou = inter.float() / areas[order[1:]].clamp(max=areas[i]).float()
        else:
            iou = inter.float() / (areas[i] + areas[order[1:]] - inter).float()  # [N-1,]

        idx = (iou <= overlap_threshold).nonzero().squeeze()  # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx + 1]  # 修补索引之间(去掉位置)
    return torch.LongTensor(keep)  # Pytorch的索引值为LongTensor


def convert_to_square(bboxes):
    """将边界框转换为方形."""
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0

    return square_bboxes

def gpu_convert_to_square(bboxes):

    square_bboxes = torch.zeros_like(bboxes, device=torch.device("cuda"), dtype=torch.float32)
    x1, y1, x2, y2 = [bboxes[:, i].float() for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0

    # 有可能 超出边界
    max_side = torch.max(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0

    square_bboxes = torch.ceil(square_bboxes + 1).int()
    return square_bboxes

def get_image_boxes(bounding_boxes, img, size=24):
    """从图像中剪出框."""

    num_boxes = len(bounding_boxes)
    height = img.shape[0]
    width = img.shape[1]

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bounding_boxes, width, height)
    img_boxes = []
    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), 'uint8')

        img_array = np.asarray(img, 'uint8')
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
            img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = cv2.resize(
            img_box, (size, size), interpolation=cv2.INTER_AREA)

        # 必须和训练时候处理一样
        im_tensor = trainTransform(img_box)
        img_boxes.append(im_tensor)
    img_boxes = torch.stack(img_boxes)
    return img_boxes

def gpu_get_image_boxes(bounding_boxes, img, size=24):
    """从图像中剪出框."""
    height = img.shape[2]
    width = img.shape[3]

    # 改变 小于零的
    bboxes = torch.max(torch.ones_like(bounding_boxes, device=torch.device('cuda')), bounding_boxes)
    # 改变 大于宽 高
    sizes = torch.IntTensor([[width, height, width, height]] * bboxes.shape[0]).to(torch.device('cuda'))
    bboxes = torch.min(bboxes, sizes)

    candidate_faces = list()
    for box in bboxes:
        img_box = img[:, :, box[1]-1: box[3], box[0]-1: box[2]]

        # resize
        img_box = torch.nn.functional.interpolate(
            img_box, size=(size, size), mode='bilinear', align_corners=True)
        candidate_faces.append(img_box)

    candidate_faces = torch.cat(candidate_faces, 0)

    return candidate_faces,bboxes

def calibrate_box(bboxes, offsets):
    """将边界框转换为更像真正的边界框."""
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    # are offsets always such that
    # x1 < x2 and y1 < y2 ?

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes

def gpu_calibrate_box(bboxes, offsets):
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = torch.unsqueeze(w, 1)
    h = torch.unsqueeze(h, 1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    # 总是如此偏离
    # x1 < x2 and y1 < y2 ?

    translation = torch.cat([w, h, w, h],dim = 1).float() * offsets
    bboxes = bboxes + translation
    return bboxes

def correct_bboxes(bboxes, width, height):
    """裁剪框太大并且可以获得关于切口的坐标.
    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.
    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            just heights and widths of boxes.
        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list

def read_annotation(base_dir, label_path):
    '''读取文件的image，box'''
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # 图像地址
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/images/' + imagepath
        images.append(imagepath)
        # 人脸数目
        nums = labelfile.readline().strip('\n')

        one_image_bboxes = []
        for i in range(int(nums)):
            bb_info = labelfile.readline().strip('\n').split(' ')
            # 人脸框
            face_box = [float(bb_info[i]) for i in range(4)]

            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]

            one_image_bboxes.append([xmin, ymin, xmax, ymax])

        bboxes.append(one_image_bboxes)

    data['images'] = images
    data['bboxes'] = bboxes
    return data

def create_dirs(dirs=[]):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


def gpu_preprocess(img):

    if isinstance(img, str):
        img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = trainTransform(img).unsqueeze(0)
    img = img.to('cuda')

    return img

def gpu_calibrate_landmarks(bboxes, landmarks):

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = torch.unsqueeze(w, 1).float()
    h = torch.unsqueeze(h, 1).float()

    # 在左上角坐标的基础上，通过 w，h 确定脸各关键点的坐标。
    landmarks_pixel = torch.zeros_like(landmarks, device=torch.device('cuda'), dtype=torch.float)
    landmarks_pixel[:, 0:5] = (torch.unsqueeze(x1, 1).float() + w * landmarks[:, 0::2]).clone()
    landmarks_pixel[:, 5:10] = (torch.unsqueeze(y1, 1).float() + h * landmarks[:, 1::2]).clone()

    return landmarks_pixel