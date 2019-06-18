# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2019/6/6
# @Last Modified by:   Hobo
# @Last Modified time: 2019/6/6 16:26
import os
import random

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Mtcnn_dataset(Dataset):
    def __init__(self):
        super(Mtcnn_dataset, self).__init__()

    def gen_dataset(self, txt_path):
        imagelist = open(txt_path, 'r')
        dataset = []
        for line in tqdm(imagelist.readlines()):
            info = line.strip().split(' ')
            data_example = dict()
            bbox = dict()
            data_example['filename'] = info[0].replace('\\', '/')
            data_example['label'] = int(info[1])
            # neg的box默认为0,part,pos的box只包含人脸框，landmark的box只包含关键点
            for idx in {'xmin', 'ymin', 'xmax',
                        'ymax', 'xlefteye', 'ylefteye',
                        'xrighteye', 'yrighteye',
                        'xnose', 'ynose', 'xleftmouth',
                        'yleftmouth', 'xrightmouth', 'yrightmouth',
                        }:
                bbox[idx] = 0
            if len(info) == 6:
                bbox['xmin'] = float(info[2])
                bbox['ymin'] = float(info[3])
                bbox['xmax'] = float(info[4])
                bbox['ymax'] = float(info[5])
            if len(info) == 12:
                bbox['xlefteye'] = float(info[2])
                bbox['ylefteye'] = float(info[3])
                bbox['xrighteye'] = float(info[4])
                bbox['yrighteye'] = float(info[5])
                bbox['xnose'] = float(info[6])
                bbox['ynose'] = float(info[7])
                bbox['xleftmouth'] = float(info[8])
                bbox['yleftmouth'] = float(info[9])
                bbox['xrightmouth'] = float(info[10])
                bbox['yrightmouth'] = float(info[11])
            data_example['bbox'] = bbox
            dataset.append(data_example)
        return dataset

    def unpack_dataset(self, dataset, transform=None):
        np.random.shuffle(dataset)
        imgs, cls_labels, rois, landmarks = [], [], [], []
        for item in range(len(dataset)):
            imgfilename = dataset[item]['filename']
            img = cv2.cvtColor(cv2.imread(imgfilename), cv2.COLOR_BGR2RGB)
            if transform is not None:
                img = transform(img)
            cls_label = dataset[item]['label']
            bbox = dataset[item]['bbox']
            roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
            # 传入landmark的10个值
            landmark = [
                bbox['xlefteye'],
                bbox['ylefteye'],
                bbox['xrighteye'],
                bbox['yrighteye'],
                bbox['xnose'],
                bbox['ynose'],
                bbox['xleftmouth'],
                bbox['yleftmouth'],
                bbox['xrightmouth'],
                bbox['yrightmouth']]
            imgs.append(img)
            cls_labels.append(cls_label)
            rois.append(roi)
            landmarks.append(landmark)

        return torch.stack(imgs), torch.tensor(cls_labels).float(
        ), torch.tensor(rois), torch.tensor(landmarks)


class DataReader(Mtcnn_dataset):
    def __init__(
            self,
            txt_path,
            im_size,
            transform=None,
            ratios=(
                2,
                1,
                1,
                2),
            batch_size=128):
        super(DataReader, self).__init__()
        self.transform = transform
        self.batch_size = batch_size

        self.pos_dataset = self.gen_dataset(
            os.path.join(txt_path, 'pos_{}.txt'.format(im_size)))
        self.part_data = self.gen_dataset(os.path.join(
            txt_path, 'part_{}.txt'.format(im_size)))
        self.neg_data = self.gen_dataset(os.path.join(
            txt_path, 'neg_{}.txt'.format(im_size)))
        self.landmark_data = self.gen_dataset(os.path.join(
            txt_path, 'landmark_{}.txt'.format(im_size)))

        ratio_sum = float(sum(ratios))
        self.ratios = [round((i / ratio_sum) * batch_size) for i in ratios]
        print("ratio:", self.ratios)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self.get_batch()

    def get_batch(self):
        # 随机抽取
        datasets = random.sample(self.pos_dataset, self.ratios[0])
        datasets += random.sample(self.part_data, self.ratios[1])
        datasets += random.sample(self.neg_data, self.ratios[2])
        datasets += random.sample(self.landmark_data, self.ratios[3])

        return self.unpack_dataset(datasets, self.transform)


if __name__ == '__main__':
    # 加载数据的 txt 路径
    train_path = '../data/dst/12'

    # 加载数据
    # 数据预处理设置
    normMean = [0.4948052, 0.48568845, 0.44682974]
    normStd = [0.24580306, 0.24236229, 0.2603115]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    dataset = DataReader(train_path,im_size=12,transform=trainTransform)
    # print(dataset.next())
    for data in dataset:
        print(data)
