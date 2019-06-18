# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2019/5/23
# @Last Modified by:   Hobo
# @Last Modified time: 2019/5/23 13:13
import sys
sys.path.append('..')

import torch
import argparse
from core.load_data import *
from core.models import LossFn, PNet, RNet, ONet
from core.utils import compute_accuracy, compute_recoll
from visdom import Visdom

class Mtcnn(object):
    def __init__(self):
        # 数据预处理设置
        self.trainTransform = transforms.Compose([
            # 把图片 转换成形状为[C,H,W]，范围取值的英文[0, 1.0]的torch.FloadTensor
            transforms.ToTensor(),
            # 进行归一化  [-1,1]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        # visdom 实例
        self.viz = Visdom()

    def train_pnet(self,train_data_path):
        device = torch.device('cuda')
        lossfn = LossFn()
        net = PNet()
        # 返回 一样的 net = net.to(device)
        net.to(device)
        # 切换到train 状态  net.eval() 测试状态
        net.train()
        # print(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


        self.viz.line(Y=torch.FloatTensor([0.]),X=torch.FloatTensor([0.]),win='pnet_train_loss',opts=dict(title='train loss'))

        # 加载数据 ratios : pos:part:neg:landmark
        trian_datasets = DataReader(train_data_path, im_size=12,transform= self.trainTransform,batch_size=4096,ratios=(2,1,1,2))

        for epoch in range(1):
            print("epoch:", epoch)
            for step, (imgs, cls_labels, rois, landmarks) in enumerate(trian_datasets):

                # [b, 3, 12, 12],[b],[4],[10]
                im_tensor = imgs.to(device)
                cls_labels = cls_labels.to(device)
                rois = rois.to(device)
                landmarks = landmarks.to(device)

                cls_pred, box_offset_pred, landmarks_pred = net(im_tensor)

                # 貌似这里打印最后一个的loss,对于整体来说不怎么准确
                cls_loss = lossfn.cls_loss(cls_labels, cls_pred)
                box_offset_loss = lossfn.box_loss(cls_labels, rois, box_offset_pred)
                landmark_loss = lossfn.landmark_loss(cls_labels, landmarks, landmarks_pred)

                print("cls_loss:", cls_loss)
                print("box_offset_loss:", box_offset_loss)
                print("landmark_loss:", landmark_loss)

                all_loss = cls_loss * 1.0 + box_offset_loss * 0.5 + landmark_loss * 0.5

                self.viz.line(Y=torch.FloatTensor([all_loss]),X=torch.FloatTensor([step]),win='pnet_train_loss',update='append')

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                print("all_loss:", all_loss)
                print("-" * 40, "step:", step, "-" * 40)

                if step % 1000 == 0:
                    accuracy = compute_accuracy(cls_pred, cls_labels)
                    recoll = compute_recoll(cls_pred, cls_labels)
                    print("=" * 80, "\n\n=> acc:{}\n=> recoll:{}\n\n".format(accuracy, recoll), "=" * 80)

                if step % 1000 == 0:
                    torch.save(net.state_dict(), os.path.join("../data/models/", "pnet_epoch_%d.pt" % epoch))
                    torch.save(net, os.path.join("../data/models/", "pnet_epoch_model_%d.pkl" % epoch))
                    epoch += 1


    def train_rnet(self,train_data_path):
        device = torch.device('cuda')
        lossfn = LossFn()
        net = RNet()
        # 返回 一样的 net = net.to(device)
        net.to(device)
        # 切换到train 状态  net.eval() 测试状态
        net.train()
        # print(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        self.viz.line(Y=torch.FloatTensor([0.]), X=torch.FloatTensor([0.]), win='rnet_train_loss',
                      opts=dict(title='train loss'))

        # 加载数据 ratios ==> pos:part:neg:landmark
        trian_datasets = DataReader(train_data_path, im_size=24, transform=self.trainTransform, batch_size=4096,
                                    ratios=(1, 1, 3, 2))
        for epoch in range(1):
            print("epoch:", epoch)
            for step, (imgs, cls_labels, rois, landmarks) in enumerate(trian_datasets):
                # [b, 3, 24, 24],[b],[4],[10]
                imgs = imgs.to(device)
                cls_labels = cls_labels.to(device)
                rois = rois.to(device)
                landmarks = landmarks.to(device)

                cls_pred, box_offset_pred, landmarks_pred = net(imgs)

                # 貌似这里打印最后一个的loss,对于整体来说不怎么准确
                cls_loss = lossfn.cls_loss(cls_labels, cls_pred)
                box_offset_loss = lossfn.box_loss(cls_labels, rois, box_offset_pred)
                landmark_loss = lossfn.landmark_loss(cls_labels, landmarks, landmarks_pred)

                print("cls_loss:", cls_loss)
                print("box_offset_loss:", box_offset_loss)
                print("landmark_loss:", landmark_loss)

                all_loss = cls_loss * 1.0 + box_offset_loss * 0.5 + landmark_loss * 0.5

                self.viz.line(Y=torch.FloatTensor([all_loss]), X=torch.FloatTensor([step]), win='rnet_train_loss',
                              update='append')

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                print("all_loss:", all_loss)
                print("-" * 40, "step:", step, "-" * 40)

                if step % 1000 == 0:
                    accuracy = compute_accuracy(cls_pred, cls_labels,threshold=0.7)
                    recoll = compute_recoll(cls_pred, cls_labels,threshold=0.7)
                    print("=" * 80, "\n\n=> acc:{}\n=> recoll:{}\n\n".format(accuracy, recoll), "=" * 80)

                if step % 1000 == 0:
                    torch.save(net.state_dict(), os.path.join("../data/models/", "rnet_epoch_%d.pt" % epoch))
                    torch.save(net, os.path.join("../data/models/", "rnet_epoch_model_%d.pkl" % epoch))
                    epoch += 1

    def train_onet(self,train_data_path):
        device = torch.device('cuda')
        lossfn = LossFn()
        net = ONet()
        # 返回 一样的 net = net.to(device)
        net.to(device)
        # 切换到train 状态  net.eval() 测试状态
        net.train()
        # print(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        self.viz.line(Y=torch.FloatTensor([0.]), X=torch.FloatTensor([0.]), win='onet_train_loss',
                      opts=dict(title='train loss'))

        # 加载数据 ratios ==> pos:part:neg:landmark
        trian_datasets = DataReader(train_data_path, im_size=48, transform=self.trainTransform, batch_size=4096,
                                    ratios=(2, 1, 3, 2))
        for epoch in range(1):
            print("epoch:", epoch)
            for step, (imgs, cls_labels, rois, landmarks) in enumerate(trian_datasets):
                # [b, 3, 24, 24],[b],[4],[10]
                imgs = imgs.to(device)
                cls_labels = cls_labels.to(device)
                rois = rois.to(device)
                landmarks = landmarks.to(device)

                cls_pred, box_offset_pred, landmarks_pred = net(imgs)

                # 貌似这里打印最后一个的loss,对于整体来说不怎么准确
                cls_loss = lossfn.cls_loss(cls_labels, cls_pred)
                box_offset_loss = lossfn.box_loss(cls_labels, rois, box_offset_pred)
                landmark_loss = lossfn.landmark_loss(cls_labels, landmarks, landmarks_pred)

                print("cls_loss:", cls_loss)
                print("box_offset_loss:", box_offset_loss)
                print("landmark_loss:", landmark_loss)

                all_loss = cls_loss * 1.0 + box_offset_loss * 0.5 + landmark_loss * 1

                self.viz.line(Y=torch.FloatTensor([all_loss]), X=torch.FloatTensor([step]), win='onet_train_loss',
                              update='append')

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                print("all_loss:", all_loss)
                print("-" * 40, "step:", step, "-" * 40)

                if step % 1000 == 0:
                    accuracy = compute_accuracy(cls_pred, cls_labels, threshold=0.7)
                    recoll = compute_recoll(cls_pred, cls_labels, threshold=0.7)
                    print("=" * 80, "\n\n=> acc:{}\n=> recoll:{}\n\n".format(accuracy, recoll), "=" * 80)

                if step % 1000 == 0:
                    torch.save(net.state_dict(), os.path.join("../data/models/", "onet_epoch_%d.pt" % epoch))
                    torch.save(net, os.path.join("../data/models/", "onet_epoch_model_%d.pkl" % epoch))
                    epoch += 1

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('train_model_net', type=str,
                        help='输入需要训练模型(pnet/rnet/onet)')

    return parser.parse_args(argv)

if __name__ == '__main__':
    train_pnet_path = '../data/dst/12'
    train_rnet_path = '../data/dst/24'
    train_onet_path = '../data/dst/48'
    mtcnn = Mtcnn()
    args = parse_arguments(sys.argv[1:]).train_model_net
    if args == 'pnet':
        mtcnn.train_pnet(train_pnet_path)
    elif args == 'rnet':
        mtcnn.train_rnet(train_rnet_path)
    elif args == 'onet':
        mtcnn.train_onet(train_onet_path)
    else:
        print('输入需要训练模型(pnet/rnet/onet)')