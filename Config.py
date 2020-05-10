# Fixed parameters
SCALAR = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
KEEP_DIFFICULT = False
RESULTS_LOG_PATH = 'results'
WEIGHT_DECAY = 0
MOMENTUM = 0.9
OPTIMIZER = 'Adam'  # SGD, Adam, RMSprop
DEVICE = 'gpu'  # cpu or gpu
IS_TENSORBOARDX = True
MAX_EPOCH = 100000
MODEL_PATH = r'model'
INPUT_SIZE = (512, 512)
# 打印调试信息
IS_DEBUG = False

# Previous use
CSV_NAME = 'test_TestLevir'  # for generate train.csv or test.csv
READ_PATH_DATASET = [
    r'F:\DataSet\LEVIR\test'
    ]
IMG_FORMATS = ['.jpg']

# Often Modify
LR = 0.0001
MODEL_SAVE_EPOCH_FREQUENCY = 20
MODEL_LOG_ITERATION_FREQUENCY = 10
MODEL_TEST_ITERATION_FREQUENCY = 30
BATCH_SIZE = 2
TEST_BATCH_SIZE = 2
# 0 : 2 267 -131 495 97
# 1 : (399, 20),(478, 82),4
# 2 : single wrong class 0 1 3 6
ANNO_MODE = 0
# 图像的尺寸和网络尺寸是否接近，若不接近则在图上进行切割，如果接近随机切割
IS_SRC_IMG_SIZE_NEAR_NET_SIZE = True
# 同类目标的标注框在大图上是否变化十分大，表现为大的基本贴合大图，
# 小的十分小，如果为True，则判断目标与输入网络尺寸的大小，将切割时长宽的大小进行适应性调整
IS_BBOX_SCALE_VARY_MUCH = False

# noraml_net
'''
CLASSES = ['background', '2']
CFG = {
    'feature_maps': [64, 32, 16, 8],
    'min_sizes': [5, 151, 298, 445],
    'max_sizes': [151, 298, 445, 592],
    'aspect_ratios': [[1.4, 1.7], [1.4, 1.7, 2.5], [1.4, 1.7], [1.4, 1.7]],
    'clip': True,
    'variances': [0.1, 0.2]
}
'''

# Levir
CLASSES = ['background', '1', '2', '3']
PRIOR_MEAN_STD = {"mean": [90.89983832590433, 98.24835848422458, 98.47987188360861], "std": 33.03479500632769}
CFG = {
    'feature_maps': [128, 64, 32, 16, 8],
    'min_sizes': [5, 115, 225, 335, 445],
    'max_sizes': [115, 225, 335, 445, 555],
    'aspect_ratios': [[1.3], [1.3, 1.6], [1.3, 1.6], [1.3, 1.6], [1.3, 1.6]],
    'clip': True,
    'variances': [0.1, 0.2]
}


# NWPU10
'''
CLASSES = ['background', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
PRIOR_MEAN_STD = {"mean": [82.18392317038423, 91.76547634717218, 86.42974329884335], "std": 40.587659824356216}
CFG = {
    'feature_maps': [64, 32, 16, 8],
    'min_sizes': [102, 221, 341, 460],
    'max_sizes': [221, 341, 460, 580],
    'aspect_ratios': [[1.5, 2, 2.5], [1.5, 2, 2.5], [1.5, 2.5], [1.5, 2.5]],
    'clip': True,
    'variances': [0.1, 0.2]
}
'''

# WKK
'''
CLASSES = ['background', '2']
PRIOR_MEAN_STD = {"mean": [60.94673879975087, 61.1032557768852, 56.83122350238761], "std": 38.15972769815662}
CFG = {
    'feature_maps': [64, 32, 16, 8],
    'min_sizes': [25, 179, 332, 486],
    'max_sizes': [179, 332, 486, 640],
    'aspect_ratios': [[1.5, 2, 2.5], [1.5, 2, 2.5], [1.5, 2.5], [1.5, 2.5]],
    'clip': True,
    'variances': [0.1, 0.2]
}
'''
