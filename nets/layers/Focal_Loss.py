import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, num_classes=2, gamma=2.0, reduction='sum'):
        """
        focal_loss损失函数, -α(1-yi)**γ *cross_entropy_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:
            阿尔法α, 类别权重. 当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],
            常用于目标检测算法中抑制背景类, retina_net中设置为0.25
        :param gamma:   伽马γ, 难易样本调节参数. retina_net中设置为2
        :param num_classes:  类别数量
        :param size_average: 损失计算方式, 默认取均值
        """
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        if isinstance(alpha, list):
            # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            assert len(alpha) == num_classes
            # print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(self.alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            assert alpha < 1
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = alpha * torch.ones(num_classes)
            self.alpha[1:] = 1 - self.alpha[0]  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

    def forward(self, preds_labels, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]  分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        preds = preds_labels.view(-1, preds_labels.size(-1))
        self.alpha = self.alpha.to(preds.device)
        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_softmax = F.softmax(preds, dim=1)
        # t = preds_softmax[torch.any(preds_softmax[:, 1:] > 0.5, dim=1), :]
        preds_softmax = torch.clamp(preds_softmax, min=1e-5, max=1-1e-5)
        preds_logsoft = torch.log(preds_softmax)
        # 这部分实现nll_loss (crossempty = log_softmax + nll)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1).long())
        # t = preds_softmax[labels.view(-1, 1) > 0]
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1).long())
        # m = preds_logsoft[labels.view(-1, 1) > 0]
        self.alpha = self.alpha.gather(0, labels.view(-1).long())
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = - torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
