import torch
import torch.nn as nn
import torch.nn.functional as F

import Config
from utils.BoxUtils import refine_match, log_sum_exp
from nets.layers.Focal_Loss import FocalLoss


class RefineMultiBoxLoss(nn.Module):
    """
    SSD Weighted Loss Function
        Compute Targets:
            1) Produce Confidence Target Indices by matching  ground truth boxes
               with (default) 'priorboxes' that have jaccard index > threshold parameter
               (default threshold: 0.5).
            2) Produce localization target by 'encoding' variance into offsets of ground
               truth boxes and their matched  'priorboxes'.
            3) Hard negative mining to filter the excessive number of negative examples
               that comes with using a large number of default bounding boxes.
               (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
        """
    def __init__(self, neg_iou_threshold=0.5, pos_iou_threshold=0.5, neg_ratio_to_pos=-1, arm_filter_socre=0.05, is_solve_odm=False):
        super(RefineMultiBoxLoss, self).__init__()
        self.is_solve_odm = is_solve_odm
        self.neg_iou_threshold = neg_iou_threshold
        self.pos_iou_threshold = pos_iou_threshold
        self.variance = Config.CFG['variances']
        self.arm_filter_score = arm_filter_socre
        self.num_classes = len(Config.CLASSES) if self.is_solve_odm else 2
        # 如果是-1， 采用全部样本的Focal_loss
        self.neg_ratio_to_pos = neg_ratio_to_pos
        self.focal_loss = FocalLoss(alpha=1e-1, num_classes=self.num_classes, reduction='mean')

    def forward(self, predict_data, priors, targets):
        """Multibox Loss
            Args:
                predictions (tuple): A tuple containing loc preds, conf preds,
                and prior boxes from SSD net.
                    conf shape: torch.size(batch_size,num_priors,num_classes)
                    loc shape: torch.size(batch_size,num_priors,4)
                    priors shape: torch.size(num_priors,4)

                ground_truth (tensor): Ground truth boxes and labels for a batch,
                    shape: [batch_size,num_objs,5] (last idx is the label).
                arm_data (tuple): arm branch containg arm_loc and arm_conf
                filter_object: whether filter out the  prediction according to the arm conf score
        """
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data = predict_data
        if self.is_solve_odm:
            loc_data, conf_data = odm_loc_data, odm_conf_data
        else:
            loc_data, conf_data = arm_loc_data, arm_conf_data
        # detach就是截断反向传播的梯度流
        defaults = priors.detach()
        batch_size = loc_data.size(0)
        num_priors = priors.size(0)

        # match priors (default boxes) and ground truth boxes
        # 获取匹配每个prior box的 ground truth
        # 创建 encode_loc 和 encode_conf 保存真实box的位置和类别
        encode_loc = torch.Tensor(batch_size, num_priors, 4)
        encode_conf = torch.Tensor(batch_size, num_priors)
        for idx in range(batch_size):
            truths = targets[idx][:, 1:].detach()
            labels = targets[idx][:, 0].detach()
            if self.is_solve_odm:
                # arm调整后，gt bbox与prior的匹配，其实对应着odm流程，毕竟prior经过arm调整过一次了
                refine_match(neg_iou_threshold=self.neg_iou_threshold, pos_iou_threshold=self.pos_iou_threshold,
                             truths=truths, priors=defaults,
                             variances=self.variance, labels=labels, encode_loc=encode_loc,
                             encode_conf=encode_conf, idx=idx, arm_loc=arm_loc_data[idx].detach())
            else:
                # 没有arm的帮助，gt bbox与prior的匹配，对应着arm流程
                refine_match(neg_iou_threshold=self.neg_iou_threshold, pos_iou_threshold=self.pos_iou_threshold,
                             truths=truths, priors=defaults,
                             variances=self.variance, labels=labels, encode_loc=encode_loc,
                             encode_conf=encode_conf, idx=idx, arm_loc=None)
        encode_loc = encode_loc.to(loc_data.device)
        encode_conf = encode_conf.to(loc_data.device)

        # 经过arm后，多了一层筛选
        if self.is_solve_odm:
            # [batch, numpriors, 2]
            arm_conf_data_P = F.softmax(arm_conf_data, dim=2)
            arm_obj_conf = arm_conf_data_P[:, :, 1]
            # 根据IoU划分的，encode_conf = 0 对应bg，否则就是fg，得到目标的索引
            index_pos = encode_conf > 0
            # objectness得分太小，判定为non-object
            index_pos[arm_obj_conf <= self.arm_filter_score] = 0
        else:
            # 取出有物体的索引
            index_pos = encode_conf > 0
            # 把所有默认为物体的框标记为1, 进行二分类
            encode_conf[index_pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # pred offsets
        loc_predict = loc_data[index_pos.unsqueeze(index_pos.dim()).expand_as(loc_data)].view(-1, 4)
        # gt offsets
        loc_gt = encode_loc[index_pos.unsqueeze(index_pos.dim()).expand_as(loc_data)].view(-1, 4)

        # 注意用mean时，输入不能为空[]
        if len(loc_gt) == 0:
            loss_l = 0
        else:
            loss_l = F.smooth_l1_loss(loc_predict, loc_gt, reduction='mean')

        # -1 就采用focal_loss
        if self.neg_ratio_to_pos == -1:
            # 取出正样本和负样本，去除模糊样本
            neg_pos_pos = encode_conf > -1
            encode_conf = encode_conf[neg_pos_pos]
            conf_data = conf_data[neg_pos_pos, :]
            loss_c = self.focal_loss(conf_data, encode_conf)
        else:
            # Compute max conf across batch for hard negative mining
            # 针对所有batch的confidence，按照置信度误差进行降序排列，取出前top_k个负样本。
            # shape[b * M, num_classes]
            batch_conf = F.softmax(conf_data.view(-1, self.num_classes), dim=1)
            # 使用logsoftmax，计算置信度,shape[b*M, 1]
            # 置信度误差越大，实际上就是预测背景的置信度越小。
            # 把所有conf进行logsoftmax处理(均为负值)，预测的置信度越小，
            # 则logsoftmax越小，取绝对值，则|logsoftmax|越大，
            # 降序排列-logsoftmax，取前 top_k 的负样本。
            conf_logP = log_sum_exp(batch_conf) - batch_conf.gather(1, encode_conf.view(-1, 1).long())
            # hard Negative Mining
            # shape[batch, num_priors]
            conf_logP = conf_logP.view(batch_size, -1)
            # 把正样本和模糊排除，剩下的就全是负样本，可以进行抽样
            conf_logP[encode_conf > 0] = -1
            # 排除模糊样本
            conf_logP[encode_conf < 0] = -1
            # 两次sort排序，能够得到每个元素在降序排列中的位置idx_rank
            # descending 表示降序
            _, index = conf_logP.sort(1, descending=True)
            _, index_rank = index.sort(1)

            # 抽取负样本
            # 每个batch中正样本的数目，shape[batch,1]
            num_pos = index_pos.long().sum(1, keepdim=True)

            # 如果改图无正样本，则选10个负样本
            num_neg = torch.clamp(self.neg_ratio_to_pos*num_pos, min=5, max=torch.sum(encode_conf == 0)-1)
            # 抽取前top_k个负样本，shape[b, M]
            index_neg = index_rank < num_neg.expand_as(index_rank)
            # shape[b,M] --> shape[b,M,num_classes]
            pos_idx = index_pos.unsqueeze(2).expand_as(conf_data)
            neg_idx = index_neg.unsqueeze(2).expand_as(conf_data)

            # 提取出所有筛选好的正负样本(预测的和真实的)
            # gt函数比较a中元素大于（这里是严格大于）b中对应元素，大于则为1，不大于则为0
            conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            conf_target = encode_conf[(index_pos + index_neg).gt(0)].long()
            # 计算conf交叉熵
            # loss_c = self.focal_loss(conf_p, conf_target)
            loss_c = F.cross_entropy(conf_p, conf_target, reduction='mean')
        loss_l = loss_l
        loss_c = loss_c
        return loss_l, loss_c


# 调试代码使用
if __name__ == "__main__":
    for _ in range(100):
        loss = RefineMultiBoxLoss(is_solve_odm=True)
        predict = (torch.randn(1, 100, 4), torch.randn(1, 100, 2), torch.randn(1, 100, 4), torch.randn(1, 100, 21))
        prior = torch.randn(100, 4)
        t = torch.zeros(1, 10, 4)
        tt = torch.randint(1, (1, 10, 1))
        t = torch.cat((tt.float(), t), dim=2)
        # predict_data, priors, targets
        l, c = loss(predict, prior, t)
        # 随机randn,会导致g_wh出现负数，此时结果会变成 nan
        print('loc loss:', l)
        print('conf loss:', c)












