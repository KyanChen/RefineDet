
# Fixed parameters
SCALAR = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
CLASSES = ['background', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
KEEP_DIFFICULT = False
RESULTS_LOG_PATH = 'results'
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
OPTIMIZER = 'RMSprop'  # SGD, Adam, RMSprop
DEVICE = 'gpu'  # cpu or gpu
IS_TENSORBOARDX = True
MODEL_SAVE_EPOCH_FREQUENCY = 10
MODEL_LOG_ITERATION_FREQUENCY = 10
MODEL_TEST_ITERATION_FREQUENCY = 40

# Previous use
MODEL_STATUS = 'train'  # for generate train.csv or test.csv and indicate the nets status
READ_PATH_DATASET = r'F:\DataSet\NWPU VHR-10 dataset\positive image set\train'  # should be abspath
PRIOR_MEAN_STD = {"mean": [82.18392317038423, 91.76547634717218, 86.42974329884335], "std": 40.587659824356216}

# Train use
IMG_FORMAT = '.jpg'
MAX_EPOCH = 10000
MODEL_PATH = r'model'
IS_SRC_IMG_SIZE_NEAR_NET_SIZE = False
INPUT_SIZE = (512, 512)
BATCH_SIZE = 2
TEST_BATCH_SIZE = 2
LR = 0.0001


# net
CFG = {
    'feature_maps': [64, 32, 16, 8],
    'min_sizes': [102, 221, 341, 460],
    'max_sizes': [221, 341, 460, 580],
    'aspect_ratios': [[1.5, 2, 2.5], [1.5, 2, 2.5], [1.5, 2.5], [1.5, 2.5]],
    'clip': True,
    'variances': [0.1, 0.2]

}

