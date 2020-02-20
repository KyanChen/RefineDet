import torch
import itertools
import Config
from math import sqrt as sqrt


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source feature map.
        Note:
        This 'layer' has changed between versions of the original SSD
        paper, so we include both versions, but note v2 is the most tested and most
        recent version of the paper.
        cfg: the parameters of net, dict form
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.feature_maps = cfg['feature_maps']
        self.img_size = Config.INPUT_SIZE[0]
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']

    def forward(self):
        prior_boxs = []
        # 遍历多尺度多feature map
        for index, value in enumerate(self.feature_maps):
            # 遍历feature map每个像素
            # itertools.product 返回A、B中的元素的笛卡尔积的元组
            # repeat是一个关键字参数，指定重复生成序列的次数。
            for row, col in itertools.product(range(value), repeat=2):
                # unit center x,y
                # 将特征图的坐标对应回原图坐标，然后缩放成0-1的相对距离
                # (index_i + 0.5) * downSampleRate / imgSize
                cx = (row + 0.5) / value
                cy = (col + 0.5) / value

                # aspect_ratio: 1
                # rel size: min_size
                size_k = self.min_sizes[index] / self.img_size
                prior_boxs += [cx, cy, size_k, size_k]
                # aspect_ratio: 1
                # rel size: sqrt(size_k * size_k+1)
                if self.max_sizes:
                    size_k_prime = sqrt(self.min_sizes[index] * self.max_sizes[index]) / self.img_size
                    prior_boxs += [cx, cy, size_k_prime, size_k_prime]
                # aspect_ratio: !1
                for ar in self.aspect_ratios[index]:
                    prior_boxs += [cx, cy, size_k*sqrt(ar), size_k/sqrt(ar)]
                    prior_boxs += [cx, cy, size_k/sqrt(ar), size_k*sqrt(ar)]
        # transfor to tensor
        output = torch.Tensor(prior_boxs).view(-1, 4)
        # 截断
        if self.clip:
            output.clamp_(min=0, max=1)
        return output




