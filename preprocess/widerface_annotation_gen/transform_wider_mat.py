import os
import cv2
import time
from scipy.io import loadmat

# 图片数据所在路径
# path_image = '/idata/data/wider_face/WIDER_train/images'

mat_to_label = './wider_face_train.mat'

transfrom_file = './wider_face_train_label.txt'


def loader_mat(mat_to_label, path_to_image=""):
    print("开始")
    mat_f = loadmat(mat_to_label)
    # 所在文件夹
    event_list = mat_f['event_list']
    # 文件名
    file_list = mat_f['file_list']
    # 人脸 坐标
    face_bbx_list = mat_f['face_bbx_list']

    for event_idx, event in enumerate(event_list):
        e = event[0][0]   # 0--Parade  文件夹
        for file, bbx in zip(file_list[event_idx][0],
                             face_bbx_list[event_idx][0]):
            f = file[0][0]
            path_of_image = os.path.join(path_to_image, e, f) + ".jpg"

            bboxes = []
            for i in range(bbx[0].shape[0]):
                xmin, ymin, xmax, ymax = bbx[0][i]
                bboxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))

            # return 一张图片路径 图片上的人脸框
            yield path_of_image, bboxes

def write_file(wider):
    print('start transforming....')
    line_count = 0
    box_count = 0
    t = time.time()

    with open(transfrom_file, 'w+') as f:
        # 按 ctrl-C 停止该过程
        for data in wider:
            line = []
            print(data)
            line.append(str(data[0]))
            line_count += 1
            for i, box in enumerate(data[1]):
                box_count += 1
                for j, bvalue in enumerate(box):
                    line.append(str(bvalue))

            line.append('\n')

            line_str = ' '.join(line)
            f.write(line_str)

    st = time.time() - t
    print('end transforming')

    print('spend time:%ld' % st)
    print('total line(images):%d' % line_count)
    print('total boxes(faces):%d' % box_count)


if __name__ == '__main__':
    wider = iter(loader_mat(mat_to_label=mat_to_label))
    write_file(wider)