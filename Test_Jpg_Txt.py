# 按单张样本测试测试集的小图，得到txt文件， score class_name left top right bottom
import torch
import os
import os.path as op
import json
import torch.backends.cudnn as cudnn
import random
import numpy as np
import tqdm
import cv2
import matplotlib.pyplot as plt

import Config
import torch.utils.data as data
from utils.SSDDataset import SSDDataset
from utils.Augmentations import SSDAugmentations
from nets.RefineDet import build_refinedet
from nets.layers.PriorBox import PriorBox
from utils.TestNet import test_batch
from utils.ImageTools import get_absolute_bboxes, draw_bboxes, detection_collate, get_img_from_input

if torch.cuda.is_available() and Config.DEVICE == 'gpu':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(Config.RESULTS_LOG_PATH):
    os.makedirs(Config.RESULTS_LOG_PATH)
if not os.path.exists(op.join(Config.RESULTS_LOG_PATH, 'eval', 'img')):
    os.makedirs(op.join(Config.RESULTS_LOG_PATH, 'eval', 'img'))
if not os.path.exists(op.join(Config.RESULTS_LOG_PATH, 'eval', 'pred')):
    os.makedirs(op.join(Config.RESULTS_LOG_PATH, 'eval', 'pred'))

net = build_refinedet(Config.INPUT_SIZE, len(Config.CLASSES), True)
if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

device = torch.device('cpu')
if torch.cuda.is_available() and Config.DEVICE == 'gpu':
    device = torch.device('cuda')
    net.to(device)
    cudnn.benchmark = True

model_info = {'RESUME_EPOCH': 0, 'RESUME_MODEL': None}
with open('tools/generate_dep_info/model_info.json', 'r', encoding='utf-8') as f:
    model_info = json.load(f)

print('Loading resume network', model_info['RESUME_MODEL'])
state_dict = torch.load(model_info['RESUME_MODEL'])
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove module
    else:
        name = k
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
priorboxes = PriorBox(Config.CFG)
priors = priorboxes.forward()

def eval():
    print('Loading Test Dataset...')
    test_dataset = SSDDataset(pattern='test', transform=SSDAugmentations())
    net.eval()
    global model_info
    test_dataset = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=detection_collate)
    bar = tqdm.tqdm(len(test_dataset))
    for idx, test_data in enumerate(test_dataset):
        bar.update(1)
        img_test, gt_test, img_src_test, img_name_test, bboxes_src_test = test_data
        img_test = img_test.to(device)
        output_test = net(img_test)
        predictions_test = test_batch(output_test, priors, is_refine=True)
        # return [score, classID, l, t, r, b] with img_size
        img_size = (img_src_test[0].shape[1], img_src_test[0].shape[0])
        true_bboxes = get_absolute_bboxes(predictions_test[0], img_size, Config.INPUT_SIZE)
        img_ = draw_bboxes(img_src_test[0], true_bboxes)
        img_path = op.join(Config.RESULTS_LOG_PATH, 'val', 'img', op.basename(img_name_test[0]).split('.')[0] + Config.IMG_FORMAT)
        cv2.imwrite(img_path, img_)
        txt_path = op.join(Config.RESULTS_LOG_PATH, 'val', 'txt', op.basename(img_name_test[0]).split('.')[0] + '.txt')
        if len(true_bboxes) == 0:
            f = open(txt_path, 'w')
            f.close()
        else:
            class_name = np.array(Config.CLASSES, dtype='U32')
            true_bboxes.astype(dtype='U32')
            true_bboxes[:, 1] = class_name[true_bboxes[:, 1].astype('int')]
            np.savetxt(txt_path, true_bboxes)


if __name__ == '__main__':
    eval()

