import torch.utils.data as data
import os
import cv2
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt


import Config
from utils.Augmentations import SSDAugmentations
from utils.ImageTools import draw_bboxes, get_absolute_bboxes


class SSDDataset(data.Dataset):
    """
    input: image
    csv: img label
    taget: annotation
    """

    def __init__(self, pattern='train', transform=None):
        assert pattern in ['train', 'test'], pattern + 'does not exist!'
        self.pattern = pattern
        self.transform = transform
        data_file = os.path.abspath(
            os.path.join(r'tools/generate_dep_info', 'train' + Config.CSV_NAME.lstrip('train').lstrip('test') + '.csv'))
        assert os.path.exists(data_file), data_file + ' dose not exist!'
        if self.pattern == 'test' and os.path.exists(
                r'tools/generate_dep_info/' + 'test' + Config.CSV_NAME.lstrip('train').lstrip('test') + '.csv'):
            data_file = r'tools/generate_dep_info/' + 'test' + Config.CSV_NAME.lstrip('train').lstrip('test') + '.csv'

        self.data_info = pd.read_csv(data_file, index_col=0)
        self.size = len(self.data_info)

    def __getitem__(self, index):
        # dst_img, dst_gt, img_src, img_name, bboxes_src
        img, gt, img_src, img_name, bboxes_src = self.pull_item(index)
        return img, gt, img_src, img_name, bboxes_src

    def __len__(self):
        return self.size

    def txt_paraser(self, txt_name, img_width, img_height, mode=0):
        """
        class left top right bottom
        class difficult left top right bottom
        :param txt_name: [class_ID, left, right, top, bottom]
        :param mode: 0 : 2 267 -131 495 97
                     1 : (399, 20),(478, 82),4
                     2 : single wrong class 0 1 3 6
        :return:
        if no target []
        if target return [[class_ID, left, right, top, bottom],[]]
        """
        bboxs = []
        try:
            with open(txt_name, 'r') as f_reader:
                line = f_reader.readline().strip()
                while line:
                    if mode == 0:
                        line = line.split()
                    elif mode == 1:
                        line = line.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
                        line.insert(0, line.pop())
                    elif mode == 2:
                        line = line.split()[-4:]
                        line = [Config.CLASSES[1]] + line

                    if len(line) == 5:
                        class_, left, top, right, bottom = line
                        difficult = 0
                    elif len(line) == 6:
                        class_, difficult, left, top, right, bottom = line
                    else:
                        print("Label file error")
                        return
                    try:
                        class_ID = Config.CLASSES.index(class_)
                    except ValueError:
                        line = f_reader.readline().strip()
                        continue
                    left = min(max(float(left), 0.), img_width)
                    right = max(min(float(right), img_width), 0.)
                    top = min(max(float(top), 0.), img_height)
                    bottom = max(min(float(bottom), img_height), 0)
                    bboxs += [[class_ID, left, top, right, bottom]]
                    line = f_reader.readline().strip()
        except FileNotFoundError:
            pass
        # if no target return []
        # if target return [[class_ID, left, right, top, bottom],[]]
        bboxs = np.array(bboxs)
        return bboxs

    def pull_item(self, index):
        """
        :param index: image index
        :return: torch.from_numpy(img).permute(2, 0, 1), bboxs, width, height
        """
        data = self.data_info.iloc[index]
        img_name = data['img']
        txt_name = data['label']
        # print(img_name)
        # 3 channels
        img_src = cv2.imread(img_name, cv2.IMREAD_COLOR)
        # img_src = cv2.imdecode(np.fromfile(img_name), cv2.IMREAD_COLOR)
        assert img_src is not None, img_name + ' is not valid'
        img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
        img = img_src.copy()

        height, width, channels = img_src.shape
        bboxs_src = self.txt_paraser(txt_name, width, height, mode=Config.ANNO_MODE)
        bboxs = bboxs_src.copy()

        if self.transform is not None:
            # no target
            if bboxs.size == 0:
                img, boxes, labels = self.transform(self.pattern, img, bboxs, bboxs)
            # with target
            else:
                img, boxes, labels = self.transform(self.pattern, img, bboxs[:, 1:], bboxs[:, 0])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # H x W x C to （nSample）x C x H x W
            # img = img.transpose(2, 0, 1)
            # zip boxes labels
            # boxes is empty, no target
            if boxes.size != 0:
                bboxs = np.hstack((np.expand_dims(labels, axis=1), boxes))
            else:
                bboxs = np.zeros((1, 5))
        return torch.from_numpy(img).permute(2, 0, 1), bboxs, torch.from_numpy(img_src), img_name, bboxs_src

    def pull_image(self, index):
        """
        Return the original image object at index in opencv form
        :param index
        :return: opencv form image
        """
        data = self.data_info.iloc[index]
        img_name = data['img']
        img = None
        try:
            img = cv2.imdecode(np.fromfile(img_name, dtype='uint8'), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            assert img is not None, img_name + ' is not valid'
        except:
            print(img_name + ' is not valid')
            exit()
        return img

    def pull_anno(self, index):
        """
        Return the original annotation of image at index
        :param index:
        :return:
        list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        """
        data = self.data_info.iloc[index]
        txt_name = data['label']

        height, width, channels = self.pull_image(index).shape
        bboxs = self.txt_paraser(txt_name, width, height)
        # if no target return []
        # if target return [[class_ID, left, right, top, bottom],[]]
        return bboxs

    def pull_img_tensor(self, index):
        """
        Return the original image at an index in tensor form
        :param index:
        :return: tensorized version of img, squeezed
        """
        # unsqueeze_ is in_place operate
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


if __name__ == '__main__':
    dataset = SSDDataset(pattern='train', transform=SSDAugmentations())
    for i in range(1):
        img, gt, img_src, img_name, bboxes_src = dataset.pull_item(0)
        print(img.shape)
        gt = get_absolute_bboxes(gt, img.shape[1:], (512, 512))
        img_ = draw_bboxes(img.permute(1, 2, 0)[:, :, (0, 1, 2)], gt)
        cv2.imwrite('%s.jpg' % i, img_)
        plt.imshow(img_)
        plt.show()


