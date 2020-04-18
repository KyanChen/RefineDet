import numpy as np
import cv2
from math import ceil, floor

import Config


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    # restrict the value between a_min and a_max
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) * (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes=None, labels=None):
        for t in self.transforms:
            img, bboxes, labels = t(img, bboxes, labels)
        return img, bboxes, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ConvertUcharToFloat(object):
    """
    Convert img form uchar to float32
    """
    def __call__(self, img, bboxes=None, labels=None):
        return img.astype(np.float32), bboxes, labels


class ToAbsoluteCoords(object):
    """
    Convert bboxes from relative to absolute coords
    """
    def __call__(self, img, bboxes=None, labels=None):
        if len(bboxes) > 0:
            height, width, channels = img.shape
            bboxes[:, ::2] *= width
            bboxes[:, 1::2] *= height
        return img, bboxes, labels


class ToRelativeCoords(object):
    """
    Convert bboxes from absolute to relative coords
    """
    def __call__(self, img, bboxes=None, labels=None):
        if None is bboxes:
            return img, bboxes, labels
        if len(bboxes):
            height, width, channels = img.shape
            bboxes[:, ::2] /= width
            bboxes[:, 1::2] /= height
            bboxes[:, ::2] = np.clip(bboxes[:, ::2], a_min=0, a_max=1.)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], a_min=0, a_max=1.)
        return img, bboxes, labels


class RandomContrast(object):
    """
    Get random contrast img
    """
    def __init__(self, lower=0.8, upper=1.2, prob=0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob
        assert self.upper >= self.lower, "contrast upper must be >= lower!"
        assert self.lower > 0, "contrast lower must be non-negative!"

    def __call__(self, img, bboxes=None, labels=None):
        """
        :param img: expect float data
        :param bboxes:
        :param labels:
        :return:
        """
        if np.random.random() < self.prob:
            alpha = np.random.uniform(self.lower, self.upper)
            img *= alpha
        return img, bboxes, labels


class RandomBrightness(object):
    """
    Get random brightness img
    """
    def __init__(self, delta=8, prob=0.5):
        self.delta = delta
        self.prob = prob
        assert 0. <= self.delta < 255., "brightness delta must between 0 to 255"

    def __call__(self, img, bboxes=None, labels=None):
        """
        :param img: expect float data
        :param bboxes:
        :param labels:
        :return:
        """
        if np.random.random() < self.prob:
            delta = np.random.uniform(- self.delta, self.delta)
            img += delta
        return img, bboxes, labels


class ConvertColor(object):
    """
    Convert img color BGR to HSV or HSV to BGR for later img distortion.
    """
    def __init__(self, current='BGR', target='HSV'):
        self.current = current
        self.target = target

    def __call__(self, img, bboxes=None, labels=None):
        if self.current is 'BGR' and self.target is 'HSV':
            cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.current is 'HSV' and self.target is 'BGR':
            cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError("Convert color fail!")
        return img, bboxes, labels


class RandomSaturation(object):
    """
    get random saturation img
    apply the restriction on saturation S
    """
    def __init__(self, lower=0.8, upper=1.2, prob=0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob
        assert self.upper >= self.lower, "saturation upper must be >= lower!"
        assert self.lower > 0, "saturation lower must be non-negative!"

    def __call__(self, img, bboxes=None, labels=None):
        """
        :param img: expect float data
        :param bboxes:
        :param labels:
        :return:
        """
        if np.random.random() < self.prob:
            img[:, :, 1] *= np.random.uniform(self.lower, self.upper)
        return img, bboxes, labels


class RandomHue(object):
    """
    get random Hue img
    apply the restriction on Hue H
    """
    def __init__(self, delta=10., prob=0.5):
        self.delta = delta
        self.prob = prob
        assert 0 <= self.delta < 360, "Hue delta must between 0 to 360!"

    def __call__(self, img, bboxes=None, labels=None):
        """
        :param img: expect float data
        :param bboxes:
        :param labels:
        :return:
        """
        if np.random.random() < self.prob:
            img[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img, bboxes, labels


class ShuffleChannels(object,):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, img, swaps):
        self.img = img
        self.swaps = swaps

    def __call__(self):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = self.img[:, :, self.swaps]
        return image


class RandomChannelNoise(object):
    """
    Get random shuffle channels
    """
    def __init__(self, prob=0.5):
        self.prob = prob
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img, bboxes=None, labels=None):
        if np.random.random() < self.prob:
            swap = self.perms[np.random.randint(len(self.perms))]
            img = ShuffleChannels(img, swap)()  # shuffle channels
        return img, bboxes, labels


class ImgDistortion(object):
    """
    Change img by distortion
    """
    def __init__(self, prob=0.5):
        self.prob = prob
        self.operation = [
            RandomContrast(),
            ConvertColor(current='BGR', target='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', target='BGR'),
            RandomContrast()
        ]
        self.random_brightness = RandomBrightness()
        self.random_light_noise = RandomChannelNoise()

    def __call__(self, img, bboxes=None, labels=None):
        img, bboxes, labels = self.random_brightness(img, bboxes, labels)
        if np.random.random() < self.prob:
            distort = Compose(self.operation[:-1])
        else:
            distort = Compose(self.operation[1:])
        img, bboxes, labels = distort(img, bboxes, labels)
        return self.random_light_noise(img, bboxes, labels)


class ExpandImg(object):
    """
    Get expand img
    """
    def __init__(self, prior_mean_std, prob=0.5):
        self.prior_mean_std = prior_mean_std
        self.prob = prob

    def __call__(self, img, bboxes=None, labels=None):
        if np.random.random() < self.prob:
            return img, bboxes, labels
        height, width, channels = img.shape
        ratio_width = 0.2 * np.random.random()
        ratio_height = 0.2 * np.random.random()
        left, right = np.random.randint(max(1, width * ratio_width), size=2)
        top, bottom = np.random.randint(max(1, width * ratio_height), size=2)
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.prior_mean_std['mean'])
        if len(bboxes) > 0:
            bboxes[:, ::2] += left
            bboxes[:, 1::2] += top
        return img, bboxes, labels


class RandomSampleCrop(object):
    """
    Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
        img (Image): the cropped image
        boxes (Tensor): the adjusted bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
    """
    def __init__(self, prob=0.5, ratios=[0.8, 1.2]):
        self.prob = prob
        self.sample_options = [
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            [0.1, None],
            [0.3, None],
            [0.7, None],
            [0.9, None],
            # randomly sample a patch
            [None, None]]
        self.ratios = np.array(ratios)

    def __call__(self, img, bboxes=None, labels=None):
        ratios = self.ratios.copy()
        # 需要调整两处的crop的w,h
        height, width, _ = img.shape
        if len(bboxes) is 0:
            if np.random.random() < self.prob and Config.IS_SRC_IMG_SIZE_NEAR_NET_SIZE:
                return img, bboxes, labels
            while True:
                w = np.random.uniform(min(width - 1, ratios[0] * Config.INPUT_SIZE[0]),
                                      min(width, ratios[1] * Config.INPUT_SIZE[0]))
                h = np.random.uniform(min(height - 1, ratios[0] * Config.INPUT_SIZE[1]),
                                      min(height, ratios[1] * Config.INPUT_SIZE[1]))
                # aspect ratio constraint b/t .5 & 2
                if h / w < 8. / 10 or h / w > 10. / 8:
                    if not width / height < 8. / 10 or width / height > 10. / 8:
                        continue
                left = np.random.uniform(0, width - w)
                top = np.random.uniform(0, height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array(
                    [max(0, int(left)), max(0, int(top)), min(width - 1, int(left + w)), min(height - 1, int(top + h))])
                img = img[rect[1]:rect[3], rect[0]:rect[2], :]
                return img, bboxes, labels
        if Config.IS_BBOX_SCALE_VARY_MUCH:
            bboxes_max_width = max(bboxes[:, 2] - bboxes[:, 0])
            bboxes_max_height = min(bboxes[:, 3] - bboxes[:, 1])

            max_width_ratio = bboxes_max_width / Config.INPUT_SIZE[0]
            max_height_ratio = bboxes_max_height / Config.INPUT_SIZE[1]
            max_ratio = max(max_width_ratio, max_height_ratio)
            if max_ratio > 1:
                ratios *= max_ratio
            # if max_ratio < 0.3:
                # ratios *= 0.8
        for _ in range(6):
            for _ in range(40):
                if np.random.random() < self.prob and Config.IS_SRC_IMG_SIZE_NEAR_NET_SIZE:
                    return img, bboxes, labels
                # randomly choose a mode
                mode = self.sample_options[np.random.randint(0, 5)]

                min_iou, max_iou = mode
                if min_iou is None:
                    min_iou = float('-inf')
                if max_iou is None:
                    max_iou = float('inf')

                # max trails (50)
                for _ in range(150):
                    current_img = img
                    # generate w and h
                    w = np.random.uniform(min(width-1, ratios[0] * Config.INPUT_SIZE[0]), min(width, ratios[1] * Config.INPUT_SIZE[0]))
                    h = np.random.uniform(min(height-1, ratios[0] * Config.INPUT_SIZE[1]), min(height, ratios[1] * Config.INPUT_SIZE[1]))
                    # aspect ratio constraint b/t .5 & 2
                    if h / w < 7. / 10 or h / w > 10. / 7:
                        if not width / height < 7. / 10 or width / height > 10. / 7:
                            continue
                    left = np.random.uniform(0, width - w)
                    top = np.random.uniform(0, height - h)

                    # convert to integer rect x1,y1,x2,y2
                    rect = np.array([max(0, int(left)), max(0, int(top)), min(width-1, int(left+w)), min(height-1, int(top+h))])

                    # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                    overlap = jaccard_numpy(bboxes, rect)

                    # is min and max overlap constraint satisfied? if not try again
                    if overlap.min() < min_iou and max_iou > overlap.max():
                        continue

                    # cut the crop from the image
                    current_img = current_img[rect[1]:rect[3], rect[0]:rect[2], :]

                    # keep overlap with gt box IF center in sampled patch
                    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0

                    # mask in all gt boxes that above and to the left of centers
                    m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                    # mask in all gt boxes that under and to the right of centers
                    m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                    # mask in that both m1 and m2 are true
                    mask = m1 * m2

                    # have any valid boxes? try again if not
                    if not mask.any():
                        continue

                    # take only matching gt boxes
                    current_boxes = bboxes[mask, :].copy()

                    # take only matching gt labels
                    current_labels = labels[mask]

                    # should we use the box left and top corner or the crop's
                    current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, :2] -= rect[:2]

                    current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, 2:] -= rect[:2]

                    return current_img, current_boxes, current_labels
            ratios *= 1.2
        return img, bboxes, labels


class RandomMirror(object):
    def __init__(self, prob=0.4):
        self.prob = prob

    def __call__(self, img, bboxes=None, labels=None):
        _, width, _ = img.shape
        if np.random.random() < self.prob:
            img = img[:, ::-1]
            if len(bboxes) is not 0:
                bboxes[:, 0::2] = width - bboxes[:, 2::-2]
        return img, bboxes, labels


class Resize(object):
    def __init__(self, size=None):
        self.size = size
        assert self.size, 'Resize error!'

    def __call__(self, img, bboxes=None, labels=None):
        height, width, _ = img.shape

        image = cv2.resize(img, self.size)
        if bboxes is None:
            return image, bboxes, labels
        if len(bboxes):
            bboxes[:, ::2] = bboxes[:, ::2] / width * self.size[0]
            bboxes[:, 1::2] = bboxes[:, 1::2] / height * self.size[1]
        return image, bboxes, labels


class Normalize(object):
    def __init__(self, prior_mean_std=Config.PRIOR_MEAN_STD):
        self.prior_mean_std = prior_mean_std

    def __call__(self, img, bboxes=None, labels=None):
        img = (img - np.array(self.prior_mean_std['mean'], np.float32))\
              / (np.float32(self.prior_mean_std['std']) + 0.000000001)
        return img, bboxes, labels


class SSDAugmentations(object):
    def __init__(self, size=Config.INPUT_SIZE, prior_mean_std=Config.PRIOR_MEAN_STD):
        self.size = size
        self.prior_mean_std = prior_mean_std
        self.augment = {'train': Compose([
            ConvertUcharToFloat(),
            ImgDistortion(),
            ExpandImg(self.prior_mean_std),
            RandomSampleCrop(ratios=[0.8, 1.3]),
            RandomMirror(),
            Resize(self.size),
            ToRelativeCoords(),
            Normalize(self.prior_mean_std),
            # ToAbsoluteCoords()
        ]), 'test': Compose([
            ConvertUcharToFloat(),
            # RandomSampleCrop(ratios=[0.8, 1.2]),
            Resize(self.size),
            Normalize(self.prior_mean_std),
            ToRelativeCoords(),
        ])}

    def __call__(self, pattern, img_src, boxes, labels):
        """
        :param img_src: opencv form, 3 channels
        :param boxes: numpy form,[[1, 2, 3, 4],[],...]
        :param labels: numpy form,[1,1,1,1]
        :return:
        """
        return self.augment[pattern](img_src, boxes, labels)

