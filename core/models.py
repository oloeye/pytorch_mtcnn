# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2019/5/23
# @Last Modified by:   Hobo
# @Last Modified time: 2019/5/23 13:39

import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


class LossFn():
    def __init__(self):
        # 此函数可以认为是 nn.CrossEntropyLoss 函数的特
        # 例。其分类限定为二分类，y 必须是{0,1}
        self.loss_cls = nn.BCELoss()
        # i.e. ||x-y||^2_2
        self.loss_box = nn.MSELoss()
        self.loss_landmark = nn.MSELoss()

    def cls_loss(self, gt_label, pred_label):
        '''没有使用困难样本挖掘'''
        # pred_label: [batch_size, 1, 1, 1] to [batch_size]
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)

        # 得到 >= 0 的掩码元素，只有0和1可以影响检测损失 相当于只有 pos neg 样本影响 1 0 -1 -2
        mask = torch.ge(gt_label, 0)
        valid_gt_label = torch.masked_select(gt_label, mask)
        valid_pred_label = torch.masked_select(pred_label, mask)
        return self.loss_cls(valid_pred_label,valid_gt_label)

    def box_loss(self, gt_label, gt_offset, pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)

        # ＃获取！= 0的mask元素
        unmask = torch.eq(gt_label, 0)
        mask = torch.eq(unmask, 0)
        # convert mask to dim index
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)
        # 只有有效元素才能影响损失
        valid_gt_offset = gt_offset[chose_index, :]
        valid_pred_offset = pred_offset[chose_index, :]
        return self.loss_box(valid_pred_offset,valid_gt_offset)

    def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)
        gt_label = torch.squeeze(gt_label)
        mask = torch.eq(gt_label, -2)

        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.loss_landmark(valid_pred_landmark,valid_gt_landmark)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        '''
        :param input: [b, c, h, w]
        :return: [b, c*h*w]
        '''
        x = input.transpose(3, 2).contiguous()
        x = x.view(x.size(0), -1)
        return x


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        # backend
        self.pre_layer = nn.Sequential(
            # x:[b,3,12,12] => [b,10,]
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # PReLU1
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1

            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # PReLU2

            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.PReLU()  # PReLU3
        )
        # detection 原论文输出的是 2维，但是在计算loss时候，就使用正样本的loss
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

        # 初始化 权重
        self.apply(weights_init)

    def forward(self, input):
        x = self.pre_layer(input)
        labels = torch.sigmoid(self.conv4_1(x))
        offsets = self.conv4_2(x)
        landmarks = self.conv4_3(x)
        return labels, offsets, landmarks


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU(),

            Flatten(),
            nn.Linear(576, 128),
            nn.PReLU(128)
        )

        self.conv5_1 = nn.Linear(128, 1)
        self.conv5_2 = nn.Linear(128, 4)
        self.conv5_3 = nn.Linear(128, 10)

        self.apply(weights_init)

    def forward(self, input):
        x = self.pre_layer(input)
        labels = torch.sigmoid( self.conv5_1(x))
        offsets = self.conv5_2(x)
        landmarks = self.conv5_3(x)
        return labels, offsets, landmarks

class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU(),

            Flatten(),
            nn.Linear(1152, 256),
            nn.PReLU(256)
        )

        self.conv5 = nn.Linear(128 * 2 * 2, 256)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5

        self.conv5_1 = nn.Linear(256, 1)
        self.conv5_2 = nn.Linear(256, 4)
        self.conv5_3 = nn.Linear(256, 10)

        self.apply(weights_init)

    def forward(self, input):
        x = self.pre_layer(input)
        labels = torch.sigmoid(self.conv5_1(x))
        offsets = self.conv5_2(x)
        landmarks = self.conv5_3(x)
        return labels, offsets, landmarks