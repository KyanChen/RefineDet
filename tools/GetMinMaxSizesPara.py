import numpy as np

import Config

layers = len(Config.CFG['feature_maps'])
minRatio = 0.05
maxRatio = 0.9

step = (maxRatio - minRatio) / (layers - 1)
sizes = Config.INPUT_SIZE[0] * np.array([minRatio + i * step for i in range(layers+1)])
minSizes = sizes[:-1].astype(int).tolist()
maxSizes = sizes[1:].astype(int).tolist()
print('\'min_sizes\': {},\n\'max_sizes\': {},'.format(minSizes, maxSizes))
