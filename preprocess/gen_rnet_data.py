# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2019/6/11
# @Last Modified by:   Hobo
# @Last Modified time: 2019/6/11 15:08
import sys
sys.path.append('..')

import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from core.detecter import pnet_detect_faces
from core.utils import *

p_model_path = '../data/models/pnet_epoch_2.pt'

def create_rnet_data(save_size = 24):
    '''通过PNet或RNet生成下一个网络的输入'''
    # 图片数据路径
    base_dir = '../data/src/WIDER_train'
    # 处理后的图片存放地址
    data_dir = '../data/dst/%d' % (save_size)
    neg_dir = os.path.join(data_dir, 'negative')
    pos_dir = os.path.join(data_dir, 'positive')
    part_dir = os.path.join(data_dir, 'part')

    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    filename = '../data/src/wider_face_train_bbx_gt.txt'

    # 读取文件的image和box对应函数在utils中
    data = read_annotation(base_dir, filename)

    all_boxes = []

    save_path = data_dir
    save_file = os.path.join(save_path, 'detections.pkl')
    if not os.path.exists(save_file):
        # 将data制作成迭代器
        print('载入wider_face数据')
        for test_data in tqdm(data['images']):
            img = cv2.imread(test_data)
            pnet_rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bounding_boxes, landmarks = pnet_detect_faces(pnet_rgb_img, p_model_path)
            # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # show_bboxes(img, bounding_boxes, landmarks)
            all_boxes.append(bounding_boxes)
        with open(save_file, 'wb') as f:
            pickle.dump(all_boxes, f, 1)
        print('完成识别')

    print('开始生成图像')
    save_hard_example(save_size, data, neg_dir, pos_dir, part_dir, save_path)


def save_hard_example(save_size, data, neg_dir, pos_dir, part_dir, save_path):
    '''将网络识别的box用来裁剪原图像作为下一个网络的输入'''

    im_idx_list = data['images']

    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)

    # save files
    neg_label_file = "../data/dst/%d/neg_%d.txt" % (save_size, save_size)
    neg_file = open(neg_label_file, 'w')

    pos_label_file = "../data/dst/%d/pos_%d.txt" % (save_size, save_size)
    pos_file = open(pos_label_file, 'w')

    part_label_file = "../data/dst/%d/part_%d.txt" % (save_size, save_size)
    part_file = open(part_label_file, 'w')

    # read detect result
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))

    print(len(det_boxes), num_of_images)

    assert len(det_boxes) == num_of_images, "弄错了"

    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0

    for im_idx, dets, gts in tqdm(zip(im_idx_list, det_boxes, gt_boxes_list)):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        image_done += 1

        if len(dets) ==0 or dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        # 转换成正方形
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 除去过小的
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            Iou = IOU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (save_size, save_size),
                                    interpolation=cv2.INTER_LINEAR)

            # 划分种类
            if np.max(Iou) < 0.3 and neg_num < 60:

                save_file = os.path.join(neg_dir, "%s.jpg" % n_idx)

                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # 偏移量
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # pos和part
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


if __name__ == '__main__':
    create_rnet_data()