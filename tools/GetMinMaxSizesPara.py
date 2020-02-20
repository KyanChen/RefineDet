import numpy as np

import Config
layers = 4
minRatio = 0.2
maxRatio = 0.9

step = (maxRatio - minRatio) / (layers - 1)
sizes = Config.INPUT_SIZE[0] * np.array([minRatio + i * step for i in range(layers+1)])
minSizes = sizes[:-1].astype(int)
maxSizes = sizes[1:].astype(int)
print('minSizes:{}\nmaxSizes:{}'.format(minSizes, maxSizes))
