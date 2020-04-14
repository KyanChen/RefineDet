import numpy as np
import cv2
import torch

import Config


def get_absolute_bboxes(boxes, absolute_img_size, relative_img_size):
    # boxes shape: [M, 6] or
    # boxes shape: [M, 5]
    if len(boxes) == 0:
        return boxes
    else:
        offset = 0
        if len(boxes[0]) == 6:
            offset = 1
        boxes[:, 1+offset::2] = boxes[:, 1+offset::2] * relative_img_size[0] * absolute_img_size[0] / relative_img_size[0]
        boxes[:, 1+offset+1::2] = boxes[:, 1+offset+1::2] * relative_img_size[1] * absolute_img_size[1] / relative_img_size[1]
    return boxes


def get_img_from_input(dst_img, mean, std):
    return dst_img.permute(1, 2, 0).cpu().numpy() * np.float32(std) + np.array(mean, np.float32)


# draw bboxes on image, bboxes with classID
def draw_bboxes(img, bboxes):
    assert img is not None, "In draw_bboxes, img is None"
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    img = img.astype(np.uint8).copy()

    try:
        for bbox in bboxes:
            bbox_coordinate = bbox
            if len(bbox) == 6:
                score = bbox[0]
                bbox_coordinate = bbox[1:]
            bbox_coordinate = list(map(int, bbox_coordinate))
            cv2.rectangle(img, pt1=tuple(bbox_coordinate[1:3] - np.array([1, 20])),
                          pt2=tuple(bbox_coordinate[1:3] + np.array([30, 1])), color=(0, 0, 255), thickness=-1)
            if len(bbox) == 6:
                cv2.putText(img, text='%s:%.2f' % (Config.CLASSES[bbox_coordinate[0]], score),
                            org=tuple(bbox_coordinate[1:3] - np.array([1, 8])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255), thickness=2)
            else:
                cv2.putText(img, text='%s' % Config.CLASSES[bbox_coordinate[0]],
                            org=tuple(bbox_coordinate[1:3] - np.array([1, 8])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255), thickness=2)
            cv2.rectangle(img, pt1=tuple(bbox_coordinate[1:3]), pt2=tuple(bbox_coordinate[3:5]), color=(0, 0, 255), thickness=2)
    except TypeError:
        print("IamgeTools, bboxes TypeError")
        print(bboxes)
    return img


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).
        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations
        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    # batch: img, gt, img_src, img_name, bboxes_src

    def toTensor(item):
        if torch.is_tensor(item):
            return item
        elif isinstance(item, type(np.array(0))):
            return torch.from_numpy(item).float()
        elif isinstance(item, type('0')):
            return item
    img_ = []
    gt_ = []
    img_src_ = []
    img_name_ = []
    bboxes_src_ = []
    for sample in batch:
        img, gt, img_src, img_name, bboxes_src = sample
        img_.append(toTensor(img))
        gt_.append(toTensor(gt))
        img_src_.append(toTensor(img_src))
        img_name_.append(toTensor(img_name))
        bboxes_src_.append(toTensor(bboxes_src))
    return torch.stack(img_, dim=0), gt_, img_src_, img_name_, bboxes_src_





