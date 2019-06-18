# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2019/6/11
# @Last Modified by:   Hobo
# @Last Modified time: 2019/6/11 14:47
import sys
sys.path.append('..')

import cv2
import math
import torch

from core.models import PNet, RNet, ONet
from core.utils import gpu_nms, gpu_calibrate_box, gpu_convert_to_square, gpu_get_image_boxes, gpu_preprocess, \
    gpu_calibrate_landmarks, gpu_show_bboxes


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


def pnet_detect_faces_gpu(
        image,
        p_model_path,
        min_face_size=12.0,
        thresholds=0.6,
        nms_thresholds=0.7):
    with torch.no_grad():
        pnet, _, _ = create_mtcnn_model(p_model_path=p_model_path)

        # 建立一个图像金字塔
        h = image.shape[2]
        w = image.shape[3]

        max_face_size = min(h, w)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)
        # 在某种意义上，应用P-Net相当于用步幅2移动12x12窗口,
        stride = 2.0
        cell_size = 12

        candidate_boxes = torch.empty(
            (0, 4), dtype=torch.float, device=torch.device('cuda'))
        candidate_scores = torch.empty((0), device=torch.device('cuda'))
        candidate_offsets = torch.empty(
            (0, 4), dtype=torch.float, device=torch.device('cuda'))

        while min_face_size <= max_face_size:
            current_scale = min_detection_size / min_face_size
            min_face_size /= factor

            img_h = math.ceil(h * current_scale)  # 向上取整 貌似有问题哟
            img_w = math.ceil(w * current_scale)

            # https://www.cnblogs.com/ocean1100/p/9494640.html
            # 坑：cv2 把 [n,c,h,w] --> [n,c,w,h]
            resize_img = torch.nn.functional.interpolate(
                image, size=(img_h, img_w), mode='bilinear', align_corners=True)

            # prob: [n,1,m,n] offset: [n,4:(x1,y1,x2,y2),m,n]
            prob, offset, _ = pnet(resize_img)

            # 去掉多余的维度，[m,n]
            prob = prob.squeeze()
            inds = (prob > thresholds).nonzero()  # inds: [[rows_idx,clos_idx],]

            # 没有发现人脸
            if inds.shape[0] == 0:
                continue
            else:
                # 边界框的转换 i:四个坐标，inds[:, 0]:图片高，inds[:, 1]:图片宽
                reg_x1, reg_y1, reg_x2, reg_y2 = [
                    offset[0, i, inds[:, 0], inds[:, 1]] for i in range(4)]

                # [n] --> [n,4] == [n:(x1,y1,x2,y2),4]
                offset = torch.stack([reg_x1, reg_y1, reg_x2, reg_y2], dim=1)
                score = prob[inds[:, 0], inds[:, 1]]

                # 当前的候选框 [n] --> [4,n] == [4,{n:x1,n:y1,n:x2,n:y2)]
                bounding_boxes = torch.stack([
                    (stride*inds[:, 1]+ 1.0),     # x1
                    (stride*inds[:, 0] + 1.0),     # y1
                    (stride*inds[:, 1] + cell_size + 1.0),  # x2
                    (stride*inds[:, 0] + cell_size + 1.0),  # y2
                ], dim=0).transpose(0, 1).float()

                bounding_boxes = torch.round(
                    bounding_boxes.float() / current_scale)

                keep = gpu_nms(bounding_boxes, score, overlap_threshold=0.5)

                bboxes = bounding_boxes[keep]
                scores = score[keep]
                offset = offset[keep]

                candidate_boxes = torch.cat([candidate_boxes, bboxes])
                candidate_scores = torch.cat([candidate_scores, scores])
                candidate_offsets = torch.cat([candidate_offsets, offset])

            # nms
        if candidate_boxes.shape[0] != 0:
            keep = gpu_nms(candidate_boxes, candidate_scores, nms_thresholds)
            candidate_boxes = candidate_boxes[keep]

            # 使用pnet预测的偏移量来变换边界框，根据 w、h 对 x1,y1,x2,y2 的位置进行调整
            candidate_boxes = gpu_calibrate_box(
                candidate_boxes, candidate_offsets)

            # 将检测出的框转化成矩形
            candidate_boxes = gpu_convert_to_square(candidate_boxes)

        return candidate_boxes


def rnet_detect_faces_gpu(
        img,
        bounding_boxes,
        r_model_path,
        thresholds=0.7,
        nms_thresholds=0.7):
    with torch.no_grad():
        _, rnet, _ = create_mtcnn_model(r_model_path=r_model_path)

        img_boxes, bboxes = gpu_get_image_boxes(bounding_boxes, img, size=24)

        probs, offsets, _ = rnet(img_boxes)

        mask = (probs >= thresholds)
        boxes = bboxes[mask.squeeze()]
        box_regs = offsets[mask.squeeze()]
        scores = probs[mask].reshape((-1,))

        if boxes.shape[0] > 0:
            # print(boxes,box_regs)
            boxes = gpu_calibrate_box(boxes.float(), box_regs)
            # nms
            keep = gpu_nms(boxes, scores, nms_thresholds)
            boxes = boxes[keep]

            # 将检测出的框转化成矩形
            boxes = gpu_convert_to_square(boxes)

        return boxes


def onet_detect_faces_gpu(
        img,
        bounding_boxes,
        o_model_path,
        thresholds=0.9,
        nms_thresholds=0.7):
    '''跟rnet 一样的方法'''
    with torch.no_grad():
        _, _, onet = create_mtcnn_model(o_model_path=o_model_path)

        img_boxes, bboxes = gpu_get_image_boxes(bounding_boxes, img, size=48)

        probs, offsets, landmarks = onet(img_boxes)

        # filter negative boxes
        mask = (probs.squeeze() >= thresholds)
        boxes = bboxes[mask]
        box_regs = offsets[mask]
        scores = probs[mask].reshape((-1,))
        landmarks = landmarks[mask]

        if boxes.shape[0] > 0:
            # 计算面部地标点
            landmarks = gpu_calibrate_landmarks(boxes, landmarks)
            boxes = gpu_calibrate_box(boxes.float(), box_regs)

            height = img.shape[2]
            width = img.shape[3]

            bboxes = torch.max(
                torch.zeros_like(
                    boxes,
                    device=torch.device('cuda')),
                boxes)
            sizes = torch.IntTensor(
                [[width, height, width, height]] * bboxes.shape[0]).to(torch.device('cuda'))
            boxes = torch.min(bboxes.int(), sizes)

            # nms
            keep = gpu_nms(boxes, scores, nms_thresholds, mode='min')
            boxes = boxes[keep]
            landmarks = landmarks[keep]

        return boxes, landmarks


if __name__ == '__main__':
    p_model_path = '../data/models/pnet_epoch_8.pt'
    r_model_path = '../data/models/rnet_epoch_27.pt'
    o_model_path = '../data/models/onet_epoch_6.pt'

    img = cv2.imread('../data/images/office16.jpg')
    img_bk = img.copy()

    img = gpu_preprocess(img)
    bounding_boxes = pnet_detect_faces_gpu(img, p_model_path)

    # p_img_bk = img_bk.copy()
    # gpu_show_bboxes(p_img_bk, bounding_boxes, [])

    bounding_boxes = rnet_detect_faces_gpu(img, bounding_boxes, r_model_path)

    # r_img_bk = img_bk.copy()
    # gpu_show_bboxes(r_img_bk, bounding_boxes, [])

    bounding_boxes, landmarks = onet_detect_faces_gpu(
        img, bounding_boxes, o_model_path,nms_thresholds=0.1)

    gpu_show_bboxes(img_bk, bounding_boxes, landmarks)