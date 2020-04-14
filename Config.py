
# Fixed parameters
SCALAR = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
CLASSES = ['background', '1', '2', '3']
KEEP_DIFFICULT = False
RESULTS_LOG_PATH = 'results'
WEIGHT_DECAY = 0
MOMENTUM = 0.9
OPTIMIZER = 'Adam'  # SGD, Adam, RMSprop
DEVICE = 'gpu'  # cpu or gpu
LR = 0.0001
IS_TENSORBOARDX = True
MODEL_SAVE_EPOCH_FREQUENCY = 15
MODEL_LOG_ITERATION_FREQUENCY = 10
MODEL_TEST_ITERATION_FREQUENCY = 30

# Previous use
MODEL_STATUS = 'test'  # for generate train.csv or test.csv and indicate the nets status
READ_PATH_DATASET = r'F:\DataSet\LEVIR\imageWithLabel\val'  # should be abspath
PRIOR_MEAN_STD = {"mean": [90.89983832590433, 98.24835848422458, 98.47987188360861], "std": 33.03479500632769}

# Train use
IMG_FORMAT = '.jpg'
MAX_EPOCH = 1000
MODEL_PATH = r'model'
INPUT_SIZE = (512, 512)
BATCH_SIZE = 2
TEST_BATCH_SIZE = 2
# 0 : 2 267 -131 495 97
# 1 : (399, 20),(478, 82),4
# 2 : single wrong class 0 1 3 6
ANNO_MODEL = 0


# 图像的尺寸和网络尺寸是否接近，若不接近则在图上进行切割，如果接近随机切割
IS_SRC_IMG_SIZE_NEAR_NET_SIZE = True

# 同类目标的标注框在大图上是否变化十分大，表现为大的基本贴合大图，
# 小的十分小，如果为True，则判断目标与输入网络尺寸的大小，将切割时长宽的大小进行适应性调整
IS_BBOX_SCALE_VARY_MUCH = False

# 打印调试信息
IS_DEBUG = False

# net
CFG = {
    'feature_maps': [64, 32, 16, 8],
    'min_sizes': [25, 162, 298, 435],
    'max_sizes': [162, 298, 435, 571],
    'aspect_ratios': [[1.5, 2, 2.5], [1.5, 2, 2.5], [1.5, 2.5], [1.5, 2.5]],
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
    'min_sizes': [102, 221, 341, 460],
    'max_sizes': [221, 341, 460, 580],
    'aspect_ratios': [[1.5, 2, 2.5], [1.5, 2, 2.5], [1.5, 2.5], [1.5, 2.5]],
    'clip': True,
    'variances': [0.1, 0.2]
}
'''