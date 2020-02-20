import os
import glob
import pandas as pd
import cv2

import Config


class GetTrainTestCSV:
    """
    img                 txt
    img abspath         txt abspath or None
    """
    def __init__(self, dataset_path=Config.READ_PATH_DATASET):
        self.data_path = os.path.abspath(dataset_path)
        self.img_format = Config.IMG_FORMAT
        self.save_path_csv = r'generate_dep_info'
        if not os.path.exists(self.save_path_csv):
            os.makedirs(self.save_path_csv)
        self.status = Config.MODEL_STATUS

    def get_csv(self):
        assert os.path.exists(self.data_path), 'No dir ' + self.data_path
        img_file_list = glob.glob(
            os.path.join(self.data_path, '*{0}'.format(self.img_format)))
        assert len(img_file_list), 'No data in DATASET_PATH!'
        data_information = {'img': [], 'label': []}
        for img_file in img_file_list:
            img = cv2.imread(img_file)
            if img is None:
                continue
            data_information['img'].append(img_file)
            txt_file = img_file.replace(self.img_format, '.txt')
            if not os.path.exists(txt_file):
                txt_file = None
            data_information['label'].append(txt_file)
        data_annotation = pd.DataFrame(data_information)
        writer_name = os.path.join(self.save_path_csv, '{0}.csv'.format(self.status))
        data_annotation.to_csv(writer_name)
        print(os.path.basename(writer_name) + ' file saves successfully!')


if __name__ == '__main__':
    getTrainTestCSV = GetTrainTestCSV()
    getTrainTestCSV.get_csv()
