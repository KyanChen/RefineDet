import os
import glob
import pandas as pd
import cv2
import sys
import tqdm
sys.path.append('../')

import Config


class GetTrainTestCSV:
    """
    img                 txt
    img abspath         txt abspath or None
    """
    def __init__(self, dataset_path=Config.READ_PATH_DATASET, csv_name=Config.CSV_NAME):
        self.data_path = dataset_path
        self.img_formats = Config.IMG_FORMATS
        self.save_path_csv = r'generate_dep_info'
        if not os.path.exists(self.save_path_csv):
            os.makedirs(self.save_path_csv)
        self.csv_name = csv_name

    def get_csv(self):
        data_information = {'img': [], 'label': []}
        for idx, data_dir in enumerate(self.data_path):
            if len(self.data_path) == len(self.img_formats):
                img_format = self.img_formats[idx]
            else:
                img_format = self.img_formats[0]
            assert os.path.exists(data_dir), 'No dir ' + data_dir
            img_file_list = glob.glob(
                os.path.join(data_dir, '*{0}'.format(img_format)))
            assert len(img_file_list), 'No data in DATASET_PATH!'
            pbar = tqdm.tqdm(total=len(img_file_list))
            for img_file in img_file_list:
                pbar.update(1)
                img = cv2.imread(img_file)
                if img is None:
                    continue
                data_information['img'].append(img_file)
                txt_file = img_file.replace(img_format, '.txt')
                if not os.path.exists(txt_file):
                    txt_file = None
                data_information['label'].append(txt_file)
            pbar.close()
        data_annotation = pd.DataFrame(data_information)
        writer_name = os.path.join(self.save_path_csv, '{0}.csv'.format(self.csv_name))
        data_annotation.to_csv(writer_name)
        print(os.path.basename(writer_name) + ' file saves successfully!')


if __name__ == '__main__':
    getTrainTestCSV = GetTrainTestCSV()
    getTrainTestCSV.get_csv()
