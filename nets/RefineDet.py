import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict

import Config
from .layers.L2Norm import L2Norm


# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(in_channel=3, batch_norm=False):
    # VGG16 Model
    num_kernel_each_layer = [64, 64, 'MaxPool', 128, 128, 'MaxPool',
                             256, 256, 256, 'CeilModel', 512, 512, 512, 'MaxPool', 512, 512, 512]
    layers = []
    for value in num_kernel_each_layer:
        if value is 'MaxPool':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif value is 'CeilModel':
            # when True, will use ceil instead of floor to compute the output shape
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            Conv2d = nn.Conv2d(in_channels=in_channel, out_channels=value, kernel_size=3, padding=1)
            if batch_norm:
                layers += [Conv2d, nn.BatchNorm2d(value), nn.ReLU()]
            else:
                layers += [Conv2d, nn.ReLU()]
            in_channel = value
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(), conv7, nn.ReLU()]
    return layers


def getNumPriorBoxesEachLayer():
    numBoxes = np.ones(4, dtype=np.int)
    if Config.CFG['max_sizes']:
        numBoxes += 1
    for i, ar in enumerate(Config.CFG['aspect_ratios']):
        numBoxes[i] += 2 * len(ar)
    return numBoxes


class RefineSSD(nn.Module):
    """Single Shot Multibox Architecture
        The network is composed of a base VGG network followed by the
        added multibox conv layers.  Each multibox layer branches into
            1) conv2d for class conf scores
            2) conv2d for localization predictions
            3) associated priorbox layer to produce default bounding
               boxes specific to the layer's feature map size.
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.

        Args:
            phase: (string) Can be "test" or "train"
            base: VGG16 layers for input, size of either 300 or 500
            extras: extra layers that feed to multibox loc and conf layers
            head: "multibox head" consists of loc and conf conv layers
    """
    def __init__(self, num_classes, is_refine=False):
        super(RefineSSD, self).__init__()
        self.num_classes = num_classes
        self.is_refine = is_refine
        self.numPriorBoxesEachLayer = getNumPriorBoxesEachLayer()

        # SSD network
        self.base_network = nn.ModuleList(vgg(in_channel=3, batch_norm=False))
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(512, 8)

        self.extra_layers = nn.Sequential(
            OrderedDict([('conv1', nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)),
                         ('relu1', nn.ReLU()),
                         ('conv2', nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)),
                         ('relu2', nn.ReLU())])
        )

        if self.is_refine:
            self.arm_loc = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(512, self.numPriorBoxesEachLayer[0] * 4, kernel_size=3, stride=1, padding=1)),
                ('conv2', nn.Conv2d(512, self.numPriorBoxesEachLayer[1] * 4, kernel_size=3, stride=1, padding=1)),
                ('conv3', nn.Conv2d(1024, self.numPriorBoxesEachLayer[2] * 4, kernel_size=3, stride=1, padding=1)),
                ('conv4', nn.Conv2d(512, self.numPriorBoxesEachLayer[3] * 4, kernel_size=3, stride=1, padding=1))
            ]))
            self.arm_conf = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(512, self.numPriorBoxesEachLayer[0] * 2, kernel_size=3, stride=1, padding=1)),
                ('conv2', nn.Conv2d(512, self.numPriorBoxesEachLayer[1] * 2, kernel_size=3, stride=1, padding=1)),
                ('conv3', nn.Conv2d(1024, self.numPriorBoxesEachLayer[2] * 2, kernel_size=3, stride=1, padding=1)),
                ('conv4', nn.Conv2d(512, self.numPriorBoxesEachLayer[3] * 2, kernel_size=3, stride=1, padding=1)),
            ]))

        # 有出入
        self.last_layer_trans = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('relu3', nn.ReLU())
        ]))

        # trans_layers, correspond with output shape
        self.trans_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            )
        ])

        # up_layers, 3 layers is the same
        self.up_layers = nn.ModuleList([
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
            ])
        # latent layers, 3 layers is the same
        self.latent_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU()),
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU()),
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU())
        ])

        self.odm_loc = nn.ModuleList([
            nn.Conv2d(256, self.numPriorBoxesEachLayer[0] * 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, self.numPriorBoxesEachLayer[1] * 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, self.numPriorBoxesEachLayer[2] * 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, self.numPriorBoxesEachLayer[3] * 4, kernel_size=3, stride=1, padding=1),
        ])
        self.odm_conf = nn.ModuleList([
            nn.Conv2d(256, self.numPriorBoxesEachLayer[0] * self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, self.numPriorBoxesEachLayer[1] * self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, self.numPriorBoxesEachLayer[2] * self.num_classes, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, self.numPriorBoxesEachLayer[3] * self.num_classes, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x):
        """
        Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        arm_sources = list()
        arm_loc_list = list()
        arm_conf_list = list()
        odm_sources = list()
        odm_loc_list = list()
        odm_conf_list = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base_network[k](x)
        # 3 maxpooling shape 512/2**3 = 64, [batchsize, 512, 64,64]
        # conv4_3上牵出一个arm检测分支
        s = self.L2Norm_4_3(x)
        arm_sources.append(s)

        # apply vgg up to conv5_3
        for k in range(23, 30):
            x = self.base_network[k](x)
        # 1 maxpooling shape 64/2 = 32, [batchsize, 512, 32,32]
        # conv5_3上牵出一个arm检测分支
        s = self.L2Norm_5_3(x)
        arm_sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.base_network)):
            x = self.base_network[k](x)
        # 1 maxpooling shape = 16
        # 1 dilation shape = 16, [batchsize, 1024, 16,16]
        # vgg-reduced版本下，全卷积的fc7上牵出一个arm检测分支
        arm_sources.append(x)

        # conv6_2, 接在vgg base后的新增层
        x = self.extra_layers(x)
        # 1 conv strde=2, shape=16/2=8, [batchsize, 512, 8,8]
        # vgg base后新增的extras后，全卷积的conv6_2上牵出一个arm检测分支，相当于是最后一个conv layer
        arm_sources.append(x)

        # apply multibox head to arm branch, arm分支的cls + loc预测，刚好对应arm_sources新增的4个分支
        if self.is_refine:
            for (arm_s, arm_l, arm_c) in zip(arm_sources, self.arm_loc, self.arm_conf):
                # self.arm_loc nsamples*12*featuremapSize
                arm_loc_list.append(arm_l(arm_s).permute(0, 2, 3, 1).contiguous())
                arm_conf_list.append(arm_c(arm_s).permute(0, 2, 3, 1).contiguous())
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)  # concatenate as rows
            arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf_list], 1)

        '''
        refinedet最高层，也即全卷积的conv6_2上牵出来的tcb模块，
        对应论文fig 1中最后一个tcb，无需高层feature map，
        直接类似fig 2的上半部分操作即可
        '''
        x = self.last_layer_trans(x)
        # odm预测分支
        # [batchsize, 512, 8, 8]
        odm_sources.append(x)

        # get transformed layers，特征层转换，对应fig 2中eltw-sum的上半部分，也是从arm_sources分支上牵出来的
        trans_layer_list = list()
        for (input_x, trans) in zip(arm_sources, self.trans_layers):
            trans_layer_list.append(trans(input_x))

        # fpn Module
        trans_layer_list.reverse()
        for (trans_layer_x, up_l, latent_l) in zip(trans_layer_list, self.up_layers, self.latent_layers):
            x = latent_l(up_l(x) + trans_layer_x)
            odm_sources.append(x)
        # odm构建好了，从top-down调换成down-top结构，方便预测
        odm_sources.reverse()
        # odm分支的cls + loc预测，与arm分支刚好对应
        for (odm_x, odm_l, odm_c) in zip(odm_sources, self.odm_loc, self.odm_conf):
            # nsamples*12*featuremapSize
            odm_loc_list.append(odm_l(odm_x).permute(0, 2, 3, 1).contiguous())
            odm_conf_list.append(odm_c(odm_x).permute(0, 2, 3, 1).contiguous())
        # 所有分支上的loc结果，做一个concate聚合
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc_list], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf_list], 1)

        # apply multibox head to source layers
        if self.is_refine:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                arm_conf.view(arm_conf.size(0), -1, 2),  # conf preds
                odm_loc.view(odm_loc.size(0), -1, 4),  # loc preds
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),  # conf preds
            )
        else:
            output = (
                odm_loc.view(odm_loc.size(0), -1, 4),  # loc preds
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),  # conf preds
            )
        return output


def build_refinedet(size=Config.INPUT_SIZE, classes=len(Config.CLASSES), is_refine=True):
    assert size in [(320, 320),  (512, 512)], "ERROR: Your size " + repr(size) + " not recognized"
    return RefineSSD(classes, is_refine)
