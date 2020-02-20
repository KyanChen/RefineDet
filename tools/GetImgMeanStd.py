import os
import pandas as pd
import cv2
import numpy as np
import json


class GetImgMeanStd:
    def __init__(self):
        self.data_file = os.path.abspath('generate_dep_info/train.csv')
        assert os.path.exists(self.data_file), 'train.csv dose not exist!'
        self.data_info = pd.read_csv(self.data_file, index_col=0)
        self.save_path_mean_std_info = 'generate_dep_info'
        self.mean = None
        self.std = None

    def get_img_mean_std(self):
        means = []
        stds = []
        for row in self.data_info.iterrows():
            img_name = row[1]['img']
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            assert img is not None, img_name + 'is not valid'
            # height*width*channels, axis=0 is the first dim
            mean = np.mean(np.mean(img, axis=0), axis=0)
            means.append(mean)
            std = np.std(img)
            stds.append(std)
        self.mean = np.mean(np.array(means), axis=0).tolist()
        self.std = np.mean(np.array(stds)).tolist()
        return {'mean': self.mean, 'std': self.std}

    def write_mean_std_information(self):
        info = self.get_img_mean_std()
        writer = os.path.join(self.save_path_mean_std_info, 'mean_std_info.json')
        with open(writer, 'w') as f_writer:
            json.dump(info, f_writer)
        print('mean:%s\nstd:%s\n' % (info['mean'], info['std']))


if __name__ == '__main__':
    getImgMeanStd = GetImgMeanStd()
    getImgMeanStd.write_mean_std_information()


