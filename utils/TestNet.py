import Config
import torch.nn.functional as F
import torch
import numpy as np

from utils.BoxUtils import decode, pointsToCenter


def decode_output(output, priors, obj_score=0.015, is_refine=True):
    if is_refine:
        arm_loc, arm_conf, loc_data, conf_data = [output_i.detach() for output_i in output]
        arm_conf, conf_data = F.softmax(arm_conf, dim=2), F.softmax(conf_data, dim=2)

    else:
        loc_data, conf_data = [output_i.detach() for output_i in output]
        conf_data = F.softmax(conf_data)

    priors = priors.detach()
    batch_size = loc_data.size(0)
    if is_refine:
        nobj_index = arm_conf[:, :, 1:] <= obj_score
        conf_data[nobj_index.expand_as(conf_data)] = 0

    boxes = torch.zeros_like(loc_data)
    scores = torch.zeros_like(conf_data)

    # Decode predictions into bboxes.
    for i in range(batch_size):
        if is_refine:
            defaults = decode(arm_loc[i], priors, Config.CFG['variances'])
            defaults = pointsToCenter(defaults)
        else:
            defaults = priors

        boxes[i] = decode(loc_data[i], defaults, Config.CFG['variances'])
        scores[i] = conf_data[i].clone()
    # shape:batch * numpriors * 4
    # shape:batch * numpriors * classes
    return boxes, scores


def test_batch(output, priors, is_refine=True):
    # output shape: (arm_l, arm_c, odm_l, odm_c)
    # arm_l shape:batch * numpriors * 4
    # priors shape: numpriors * 4

    # boxes shape:batch * numpriors * 4
    # scores shape:batch * numpriors * classes
    predictions = []
    boxes, scores = decode_output(output, priors, obj_score=0.02, is_refine=is_refine)
    boxes, scores = boxes.cpu().numpy(), scores.cpu().numpy()
    for batch_idx in range(boxes.shape[0]):
        # delete nboj
        max_score_index = scores[batch_idx].argmax(axis=1)
        obj_index = max_score_index > 0
        # boxes_single shape: [obj_num, 4]
        boxes_single = boxes[batch_idx, obj_index, :]
        # class_id shape: obj_num
        class_id = max_score_index[obj_index]
        # scores_single shape: obj_num
        scores_single = scores[batch_idx, obj_index, class_id]
        predict_single = []
        for obj in set(class_id):
            # return detected each class obj
            obj_index_each_class = class_id == obj
            boxes_single_to_nms = boxes_single[obj_index_each_class]
            scores_single_to_nms = scores_single[obj_index_each_class]
            reserve_index = nms(boxes_single_to_nms, scores_single_to_nms, threshold=0.4)
            predict_single += [np.concatenate((scores_single_to_nms[reserve_index].reshape(-1, 1),
                                               obj.repeat(len(reserve_index)).reshape(-1, 1),
                                               boxes_single_to_nms[reserve_index].reshape(-1, 4)), axis=1)]
        predict_single = np.concatenate(predict_single, axis=0) if len(predict_single) else []
        predictions += [predict_single]
    return predictions


def nms(boxes, scores, threshold=0.5, top_k=200):
    '''
    :param boxes: shape: [numpriors, 4]
    :param scores: [numpriors, classes]
    :param threshold: 0.5
    :param top_k:
    :return:the max num of box top score to remain
    '''
    keep_index = []
    # scores的值降序排列,得到索引
    idx = np.argsort(-scores, axis=0)
    # 取前top_k个进行nms
    # idx = idx[:top_k]
    while idx.size:
        keep_index += [idx[0]]
        # 移除已经保存的index
        if idx.size == 1:
            return keep_index
        idx = np.delete(idx, 0)
        iou_list = iou(boxes[keep_index[-1]], boxes[idx])
        idx = idx[iou_list < threshold]
    return keep_index


# IOU计算 Intersection over Union
def iou(box, boxes):
    # 左上角
    LT = np.maximum(box[:2], boxes[:, :2])
    # 右下角
    RB = np.minimum(box[2:], boxes[:, 2:])
    wh = RB - LT
    wh[wh < 0] = 0
    # A∩B
    intersection = wh[:, 0] * wh[:, 1]
    # box和boxes的面积
    # shape 1
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    # shape N
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou_ = intersection / (1e-10 + area_a + area_b - intersection)

    return iou_


if __name__ == '__main__':
    boxes = np.array([[1, 1, 5, 10], [1, 1, 3, 11], [5, 10, 8, 12], [3, 5, 7, 9]])
    scores = np.array([0.9, 0.8, 0.5, 0.4])
    index = nms(boxes, scores)
    print(index)





