# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2019/6/11
# @Last Modified by:   Hobo
# @Last Modified time: 2019/6/11 14:47
import sys
sys.path.append('..')

import cv2
import math
import numpy as np
import torch

from core.models import PNet, RNet, ONet
from core.utils import nms, calibrate_box, convert_to_square, get_image_boxes, show_bboxes, trainTransform, gpu_nms, \
    gpu_calibrate_box, gpu_convert_to_square, gpu_get_image_boxes, gpu_preprocess, gpu_calibrate_landmarks


def create_mtcnn_model(
        p_model_path=None,
        r_model_path=None,
        o_model_path=None):
    pnet, rnet, onet = None, None, None
    if p_model_path is not None:
        pnet = PNet()
        pnet.load_state_dict(torch.load(p_model_path))
        pnet.to('cuda')
        pnet.eval()
        # pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))

    if r_model_path is not None:
        rnet = RNet()
        rnet.load_state_dict(torch.load(r_model_path))
        rnet.to('cuda')
        rnet.eval()

    if o_model_path is not None:
        onet = ONet()
        onet.load_state_dict(torch.load(o_model_path))
        onet.to('cuda')
        onet.eval()

    return pnet, rnet, onet

def pnet_detect_faces(image, p_model_path, min_face_size=12.0, thresholds=0.6, nms_thresholds=0.7):
        with torch.no_grad():
            pnet, _, _ = create_mtcnn_model(p_model_path=p_model_path)

            # 建立一个图像金字塔
            h = image.shape[0]
            w = image.shape[1]
            max_face_size = min(w, h)

            min_detection_size = 12
            factor = 0.707  # sqrt(0.5)
            # 在某种意义上，应用P-Net相当于用步幅2移动12x12窗口,
            stride = 2
            cell_size = 12

            bounding_boxes = []
            while min_face_size <= max_face_size:
                current_scale = min_detection_size/ min_face_size

                img_h = math.ceil(h * current_scale)  # 向上取整 貌似有问题哟
                img_w = math.ceil(w * current_scale)

                resize_img = cv2.resize(
                    image, (img_w, img_h), interpolation=cv2.INTER_AREA)

                # 必须和训练时候处理一样

                img = trainTransform(resize_img).unsqueeze(0).to('cuda')
                probs, offsets, landmarks = pnet(img)

                probs = probs.to('cpu').squeeze(0).numpy()
                offsets = offsets.to('cpu').numpy()
                # label:[b,], offset:[], landmark:[]

                # 可能有脸的boxs的索引，相当于窗口移动的位置
                inds = np.where(probs > thresholds)

                # 没有发现人脸
                if inds[0].size == 0:
                    bounding_boxes.append(None)
                else:
                    reg_x1, reg_y1, reg_x2, reg_y2 = [
                        offsets[0, i, inds[1], inds[2]] for i in range(4)]

                    # 它们被定义为:
                    # w = x2 - x1 + 1
                    # h = y2 - y1 + 1
                    # x1_true = x1 + tx1*w
                    # x2_true = x2 + tx2*w
                    # y1_true = y1 + ty1*h
                    # y2_true = y2 + ty2*h

                    offsets = np.array([reg_x1, reg_y1, reg_x2, reg_y2])
                    score = probs[0, inds[1], inds[2]]

                    # P-Net应用于缩放图像
                    # 所以我们需要重新调整边界框
                    _bounding_boxes = np.vstack([
                        np.round((stride * inds[2] + 1.0) / current_scale),  # x1
                        np.round((stride * inds[1] + 1.0) / current_scale),  # y1
                        np.round(
                            (stride * inds[2] + 1.0 + cell_size) / current_scale),  # x2
                        np.round(
                            (stride * inds[1] + 1.0 + cell_size) / current_scale),  # y2
                        score, offsets
                    ])

                    # print("offsets=",offsets)

                    boxes = _bounding_boxes.T

                    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)

                    bounding_boxes.append(boxes[keep])


                min_face_size /= factor  # 从12变到图像的 最小长或宽

            if bounding_boxes:
                # 从不同的尺度收集boxs（以及偏移和分数）
                bounding_boxes = [i for i in bounding_boxes if i is not None]
                if len(bounding_boxes) == 0:
                    return bounding_boxes

                bounding_boxes = np.vstack(bounding_boxes)

                keep = nms(bounding_boxes[:, 0:5], nms_thresholds)
                bounding_boxes = bounding_boxes[keep]

                # 使用pnet预测的偏移量来变换边界框，根据 w、h 对 x1,y1,x2,y2 的位置进行微调
                bounding_boxes = calibrate_box(
                    bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
                # shape [n_boxes, 5]
                # 将检测出的框转化成矩形
                bounding_boxes = convert_to_square(bounding_boxes)
                bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
                # 不需要 landmarks
            return bounding_boxes

def rnet_detect_faces(
    img, bounding_boxes,r_model_path, thresholds=0.7, nms_thresholds=0.6):
    with torch.no_grad():
        _, rnet, _ = create_mtcnn_model(r_model_path= r_model_path)

        img_boxes = get_image_boxes(bounding_boxes, img, size=24)
        img_boxes = img_boxes.float()
        img_boxes = img_boxes.to(
            torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))
        probs, offsets, _ = rnet(img_boxes)
        # probs = probs.to('cuda' if torch.cuda.is_available() else 'cpu')
        probs = probs.to("cpu").numpy()  # shape [n_boxes, 1]
        offsets = offsets.to("cpu").numpy()  # shape [n_boxes, 4]

        keep = np.where(probs >  thresholds)[0]

        if len(keep) > 0:
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep].reshape((-1,))
            offsets = offsets[keep]
        else:
            return [], []

        keep = nms(bounding_boxes, nms_thresholds)

        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        return bounding_boxes

def onet_detect_faces(img, bounding_boxes,o_model_path, thresholds=0.8, nms_thresholds=0.7):
    '''跟rnet 一样的方法'''
    with torch.no_grad():
        _, _, onet = create_mtcnn_model(o_model_path= o_model_path)

        img_boxes = get_image_boxes(bounding_boxes, img, size=48)
        img_boxes = img_boxes.float()
        img_boxes = img_boxes.to(
            torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))
        probs, offsets, landmarks = onet(img_boxes)
        probs = probs.to("cpu").numpy()  # shape [n_boxes, 1]
        offsets = offsets.to("cpu").numpy()  # shape [n_boxes, 4]
        landmarks = landmarks.to("cpu").numpy()   # shape[n_boxes，10]

        keep = np.where(probs > thresholds)[0]

        if len(keep) > 0:
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]
        else:
            return [], []

        # 计算 landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        # print('width:{},\nheight:{},\nxmin:{},\nymin:{}\n'.format(width, height, xmin, ymin))
        # landmark[,前5个为x，后5个为y]
        # 在左上角坐标的基础上，通过 w，h 确定脸各关键点的坐标。
        landmarks_pixel = np.zeros(landmarks.shape)
        landmarks_pixel[:, 0:5] = (np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0::2]).copy()
        landmarks_pixel[:, 5:10] = (np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 1::2]).copy()
        # for i in landmarks:print(i)
        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds, mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks_pixel = landmarks_pixel[keep]

        return bounding_boxes, landmarks_pixel


if __name__ == '__main__':
    p_model_path = '../data/models/pnet_epoch_10.pt'
    r_model_path = '../data/models/rnet_epoch_25.pt'
    o_model_path = '../data/models/onet_epoch_8.pt'

    # cpu
    ###################################################################################################################

    img = cv2.imread('../data/images/office15.jpg')
    pnet_rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bounding_boxes = pnet_detect_faces(pnet_rgb_img, p_model_path)

    # pnet_img_bk = img.copy()
    # show_bboxes(pnet_img_bk, bounding_boxes, [])

    # rnet_rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bounding_boxes = rnet_detect_faces(pnet_rgb_img, bounding_boxes, r_model_path)

    # rnet_img_bk = img.copy()
    # show_bboxes(rnet_img_bk, bounding_boxes, [])

    onet_rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bounding_boxes, landmarks = onet_detect_faces(onet_rgb_img, bounding_boxes, o_model_path)

    onet_img_bk = img.copy()
    show_bboxes(onet_img_bk, bounding_boxes, landmarks)
