import torch


# 位置坐标转换
# left, top,  right,  bottom
# centerX, centerY, width, height
def pointsToCenter(boxes):
    return torch.cat(
        ((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), 1)


def centerToPoints(boxes):
    return torch.cat(
        (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


# IOU计算 Intersection over Union
def iou(box_a, box_b):
    '''
    :param box_a: Ground truth bounding box: shape[N, 4]
    :param box_b: Priors bounding box: shape[M, 4]
    :return:  Shape: [box_a.size(0), box_b.size(0)]
    '''
    N = box_a.size(0)
    M = box_b.size(0)
    # 左上角
    LT = torch.max(box_a[:, :2].unsqueeze(1).expand(N, M, 2),
                   box_b[:, :2].unsqueeze(0).expand(N, M, 2))
    RB = torch.min(box_a[:, 2:].unsqueeze(1).expand(N, M, 2),
                   box_b[:, 2:].unsqueeze(0).expand(N, M, 2))
    wh = RB - LT
    wh[wh < 0] = 0
    # A∩B
    intersection = wh[:, :, 0] * wh[:, :, 1]
    # box_a和box_b的面积
    # shape N
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    # shape M
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    # 把面积的shape扩展为inter一样的（N，M)
    area_a = area_a.unsqueeze(1).expand_as(intersection)
    area_b = area_b.unsqueeze(0).expand_as(intersection)
    iou = intersection / (area_a + area_b - intersection + 1e-10)

    return iou

# 位置编码和解码
def encode(matched, priors, variances):
    '''
        将来自于priorbox的差异编码到ground truth box中
        Args:
            matched: 每个prior box 所匹配的ground truth,
                     Shape[M,4],坐标(xmin,ymin,xmax,ymax)
            priors: 先验框box, shape[M,4],坐标(cx, cy, w, h)
            variances: 方差，list(float)
    '''
    g_center = (matched[:, :2] + matched[:, 2:]) / 2
    # shape [M,2]
    encode_delta_cxcy = (g_center - priors[:, :2]) / (priors[:, 2:] * variances[0])
    # 防止出现log出现负数，从而使loss为 nan
    eps = 1e-7
    g_wh = matched[:, 2:] - matched[:, :2]
    # shape[M,2]
    encode_delta_wh = torch.log(g_wh / priors[:, 2:] + eps) / variances[1]
    # shape[M,4]
    return torch.cat((encode_delta_cxcy, encode_delta_wh), 1)


def decode(loc, priors, variances):
    boxes = torch.cat(
        (priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
         priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    # 转化坐标为 (xmin, ymin, xmax, ymax)类型
    boxes = centerToPoints(boxes)
    return boxes


# 先验框匹配
def refine_match(threshold, truths, priors, variances, labels, encode_loc, encode_conf, idx, arm_loc=None):
    """Match each arm bbox with the ground truth box of the highest jaccard
        overlap, encode the bounding boxes, then return the matched indices
        corresponding to both confidence and location preds.
        Args:
            threshold: (float) The overlap threshold used when mathing boxes.
            truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
            priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
            variances: (tensor) Variances corresponding to each prior coord,
                Shape: [num_priors, 4].
            labels: (tensor) All the class labels for the image, Shape: [num_obj].
            loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
            conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
            idx: (int) current batch index
            arm_loc: (tensor) arm loc data,shape: [n_priors,4]
        Return:
            The matched indices corresponding to 1)location and 2)confidence preds.
        """
    if arm_loc is None:
        overlaps = iou(truths, centerToPoints(priors))
    else:
        decode_arm = decode(arm_loc, priors=priors, variances=variances)
        overlaps = iou(truths, decode_arm)

    # [1,num_objects] 和每个ground truth box 交集最大的 prior box, 输出为N
    # best_prior_overlap, best_prior_index = overlaps.max(1)
    best_prior_overlap, best_prior_index = overlaps.max(1, keepdim=True)
    # [1,num_priors] 和每个prior box 交集最大的 ground truth box， 输出为M
    # best_truth_overlap, best_truth_index = overlaps.max(0)
    best_truth_overlap, best_truth_index = overlaps.max(0, keepdim=True)
    best_truth_index.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_index.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # 1,保证每个ground truth box 与某一个prior box 匹配，固定值为 2 > threshold
    best_truth_overlap.index_fill_(0, best_prior_index, 2)
    # 2,保证每一个ground truth 匹配它的都是具有最大IOU的prior
    # 根据 best_prior_dix 锁定 best_truth_idx里面的最大IOU prior
    for j in range(best_prior_index.size(0)):
        best_truth_index[best_prior_index[j]] = j

    # 提取出所有匹配的ground truth box, Shape: [num_priors,4]
    mathes = truths[best_truth_index]
    if arm_loc is None:
        # 提取出所有GT框的类别， Shape:[num_priors]
        conf = labels[best_truth_index]
        loc = encode(mathes, priors, variances)
    else:
        conf = labels[best_truth_index]
        loc = encode(mathes, pointsToCenter(decode_arm), variances)
    # 把 iou < threshold 的框类别设置为 bg,即为0
    # label as background
    conf[best_truth_overlap < threshold] = 0
    encode_loc[idx] = loc  # [num_priors,4] encoded offsets to learn
    encode_conf[idx] = conf  # [num_priors] top class label for each prior


# 首先需要使用 hard negative mining 将正负样本按照 1:3 的比例把负样本抽样出来，抽样的方法是：
# 思想： 针对所有batch的confidence，按照置信度误差进行降序排列，取出前top_k个负样本。
def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
        This will be used to determine unaveraged confidence loss across
        all examples in a batch.
        Args:
            x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.detach().max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


if __name__ == '__main__':
    threshold = 0.5
    truths = torch.Tensor([[2, 4, 5, 6], [0, 0, 3, 6]])
    truths = torch.Tensor([[0, 0, 0, 0]])
    arm_loc = torch.Tensor([[0.1, 0.1, -0.01, -0.02], [0.1, 0.2, -0.02, -0.02], [0.3, 0.1, -0.03, -0.01]])
    priors = torch.Tensor([[2, 4, 7, 9], [2, 5, 7, 10], [0, 2, 5, 10]])
    variances = [1, 1]
    labels = torch.Tensor([0])
    refine_match(threshold, truths, priors, variances, labels, None, None, 1, arm_loc)