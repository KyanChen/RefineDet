# coding=utf-8
import os
import glob
from bs4 import BeautifulSoup
import tqdm
import datetime
import threading
import numpy as np
import ogr
import torch
import json
import time

import Config
from nets.RefineDet import build_refinedet
import utils.GDAL_utils as gdal_utils
import torch.backends.cudnn as cudnn
from utils.Augmentations import Resize, Normalize, ToRelativeCoords
from nets.layers.PriorBox import PriorBox
from utils.TestNet import test_batch


class Block:
    """图像块结构体
    """
    def __init__(self, img_name, img_part, x, y, class_num):
        """初始化结构体
        """
        self.img_name = img_name
        self.img_block = img_part
        height, width, _ = self.img_block.shape
        self.mask_block = np.zeros((height, width, class_num), np.uint8)
        self.offset_x = x
        self.offset_y = y
        self.is_processed = False


class TestFullImg:
    """光学图像目标检测
    """
    def __init__(self, info_path):
        """参数初始化
        """
        # 先验参数
        self.queue_maxsize = 150
        self.class_names = ['tail_mine']
        self.img_format = 'tiff'

        self.run_info = {'bndboxes': {}, 'StartTime': '', 'EndTime': '', 'ProcessTime': '',
                         'ProduceTime': '', 'ReceiveTime': '', 'SolarAzimuth': '',
                         'SolarZenith': '', 'TopLeftLatitude': '', 'TopLeftLongitude': '',
                         'TopRightLatitude': '', 'TopRightLongitude': '', 'BottomRightLatitude': '',
                         'BottomRightLongitude': '', 'BottomLeftLatitude': '', 'BottomLeftLongitude': '',
                         'SatelliteID': '', 'AIModelVersion': '1.0', 'ManCorrect': 'False'}
        self.block_queue = []
        self.thread_lock = threading.Lock()
        self.read_finished = False

        # 输入参数
        self.model_path = ''
        self.threshold = 0.5
        self.outputFolder_Path = ''
        self.logfp = ''
        self.overlap_factor = 0.2
        self.block_height = 512
        self.block_width = 512
        self.input_img_list = []
        self.load_config(info_path)

        self.write_log('输入路径：%s' % os.path.dirname(self.input_img_list[0]))
        self.write_log('输出路径：%s' % self.outputFolder_Path)

    def get_file_list(self, inputFilePath):
        if isinstance(inputFilePath, list):
            self.input_img_list = inputFilePath
        elif os.path.isfile(inputFilePath):
            self.input_img_list = [inputFilePath]
        elif os.path.isdir(inputFilePath):
            self.input_img_list = glob.glob(os.path.join(inputFilePath, '*.%s' % self.img_format))
        else:
            raise Exception("Invalid Input_path!")
        for i in self.input_img_list:
            img_name = os.path.splitext(os.path.basename(i))[0]
            out_dir = os.path.join(self.outputFolder_Path, img_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            self.run_info['bndboxes'][img_name] = []

    def load_config(self, info_path):
        """读取配置文件
        """
        print(info_path)
        assert os.path.exists(info_path), "Not Found: %s" % info_path
        with open(info_path, 'r', encoding='utf-8') as f_reader:
            info = json.load(f_reader)
        try:
            self.threshold = info['threshold']
            # zoom为输入图像大小：具体为zoom*Net_Size
            # overlapRatio为覆盖的比例
            scale_factor = info['scale_factor']
            self.overlap_factor = info['overlap_factor']
            self.block_height = int(self.block_height * scale_factor)
            self.block_width = int(self.block_width * scale_factor)
            self.model_path = info["model_file"]
            input_file_path = info['input_path']
            self.outputFolder_Path = info['output_path']
            self.get_file_list(input_file_path)
            if not os.path.exists(self.outputFolder_Path):
                os.makedirs(self.outputFolder_Path)
            log_path = os.path.join(self.outputFolder_Path, 'log')
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            logfile = os.path.join(log_path,
                                   '%s.txt' % (datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
            self.logfp = open(logfile, 'a+', encoding='utf-8')
        except Exception as e:
            print('load_config error: %s' % e)

    def write_log(self, log_content):
        """写处理日志
        """
        self.logfp.write('%s\n' % str(log_content))
        self.logfp.flush()

    def multi_thread_read(self):
        """多线程读图
        """
        for i in self.input_img_list:
            self.write_log('Reading %s' % os.path.basename(i))
            try:
                img_read = gdal_utils.read_full_image(i)
                height, width, _ = img_read.shape
            except Exception as e:
                self.write_log('%s: Reading %s error!' % (e, os.path.basename(i)))
                continue
            y = 0
            flag_x_skip = False
            flag_y_skip = False
            while y < height:
                x = 0
                if self.block_height + y > height:
                    y = height - self.block_height
                    flag_y_skip = True
                flag_x_skip = False
                while x < width:
                    if x + self.block_width > width:
                        x = width - self.block_width
                        flag_x_skip = True
                    img_part = img_read[y:y+self.block_height, x:x+self.block_width, :]
                    while len(self.block_queue) >= self.queue_maxsize:
                        time.sleep(0.1)
                    self.thread_lock.acquire()
                    self.block_queue.append(Block(os.path.splitext(os.path.basename(i))[0], img_part, x, y, len(self.class_names)))
                    self.thread_lock.release()
                    if flag_x_skip:
                        break
                    x += int(self.block_width * (1 - self.overlap_factor))
                if flag_y_skip:
                    break
                y += int(self.block_height * (1 - self.overlap_factor))
        self.read_finished = True
        self.write_log('Reading done')

    def multi_thread_write(self):
        """多线程写图
        """
        for i in self.input_img_list:
            self.write_log('Writing %s' % os.path.basename(i))
            try:
                height, width, bands = gdal_utils.get_img_shape(i)
            except Exception as e:
                self.write_log('%s: Writing %s error!' % (e, os.path.basename(i)))
                continue

            for j in range(len(self.class_names)):
                img_name = os.path.splitext(os.path.basename(i))[0]

                save_path = os.path.join(
                    self.outputFolder_Path, img_name,  '%s_%s_MASK.tiff' % (img_name, self.class_names[j]))
                img_empty = np.zeros((height, width), 'uint8')
                gdal_utils.save_full_img(
                    save_path, img_empty,
                    geo_transform=gdal_utils.get_geotransform(i),
                    projection=gdal_utils.get_projection(i))

        while len(self.block_queue) or not self.read_finished:
            #
            if len(self.block_queue) and self.block_queue[0].is_processed:
                self.thread_lock.acquire()
                block = self.block_queue.pop(0)
                self.thread_lock.release()
                # Block(img_name, img_part, x, y, class_num)
                for i in range(len(self.class_names)):
                    save_path = os.path.join(
                        self.outputFolder_Path, block.img_name, '%s_%s_MASK.tiff' % (block.img_name, self.class_names[i]))
                    gdal_utils.save_image(save_path, block.mask_block[:, :, i], block.offset_x, block.offset_y)
            else:
                time.sleep(0.1)
        self.write_log('Writing done')

    def draw_mask(self, boxes, mask, width_offset, height_offset, threshold=0.5):
        """

        :param boxes:
        :param mask:
        :param width_offset:
        :param height_offset:
        :param threshold:
        :return:
        """
        height, width, _ = mask.shape
        count = 0
        loc_info = []
        if len(boxes):
            obj_index = boxes[:, 0] > threshold
            boxes = boxes[obj_index]
            if len(boxes):
                boxes[:, 2:] *= width
                boxes = np.clip(boxes, a_min=0, a_max=max(height - 1, width - 1))
                bbox = boxes[:, 1:].astype(int)
            for i in range(len(bbox)):
                count += 1
                mask[bbox[i, 2]:bbox[i, 4], bbox[i, 1]:bbox[i, 3], bbox[i, 0]-1] = 255
                # classname, top, left, bottom, right
                loc_info.append([self.class_names[bbox[i, 0]-1],
                                 bbox[i, 2] + height_offset,
                                 bbox[i, 1] + width_offset,
                                 bbox[i, 4] + height_offset,
                                 bbox[i, 3] + width_offset])
        return count, loc_info

    def multi_thread_interface(self):
        """多线程处理过程
        """
        self.write_log('building networks...')
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        net = build_refinedet(Config.INPUT_SIZE, len(Config.CLASSES), True)
        if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            net.to(device)
            cudnn.benchmark = True
        self.write_log('Loading Model...')
        state_dict = torch.load(self.model_path, map_location=device)
        net.load_state_dict(state_dict)
        net.eval()
        priorboxes = PriorBox(Config.CFG)
        priors = priorboxes.forward()

        self.write_log('Processing img')
        totoal_count = 0
        for i in self.input_img_list:
            try:
                height, width, bands = gdal_utils.get_img_shape(i)
                totoal_count += np.ceil(height / (self.block_height * (1 - self.overlap_factor))) * \
                                np.ceil(width / (self.block_width * (1 - self.overlap_factor)))
            except Exception as e:
                self.write_log('%s: Writing %s error!' % (e, os.path.basename(i)))
                continue

        pbar = tqdm.tqdm(total=totoal_count)

        start = datetime.datetime.now()
        while len(self.block_queue) or not self.read_finished:

            for block in self.block_queue:
                if not block.is_processed:
                    # height, width, _
                    img_part = block.img_block
                    img_part = Resize(Config.INPUT_SIZE)(img_part)[0]
                    img_part = Normalize(Config.PRIOR_MEAN_STD)(img_part)[0]
                    # H x W x C to C x H x W
                    img_part = img_part[:, :, (2, 1, 0)]
                    # C x H x W to （nSample）x C x W x H
                    img_part = torch.from_numpy(img_part).permute(2, 0, 1).unsqueeze_(0)
                    img_part = img_part.to(device)
                    output = net(img_part)
                    predictions = test_batch(output, priors, threshold=self.threshold, is_refine=True)
                    # return [score, classID, l, t, r, b] with img_size
                    count, loc_info = self.draw_mask(
                            predictions[0], block.mask_block,
                            block.offset_x, block.offset_y,
                            threshold=self.threshold)
                    block.is_processed = True
                    # classname, top, left, bottom, right
                    self.run_info['bndboxes'][block.img_name].extend(loc_info)
                    pbar.update(1)
                    break
        self.write_log('Process done')
        pbar.close()
        end = datetime.datetime.now()
        print("\n")
        print('网络处理阶段耗时:%.2fmin' % ((end-start).seconds/60))

    def run(self):
        """运行程序
        """
        begin = datetime.datetime.now()
        self.run_info['StartTime'] = begin.strftime('%Y:%m:%d:%H:%M:%S')

        # thread_read = threading.Thread(target=self.multi_thread_read)
        # thread_read.start()
        self.multi_thread_read()
        self.multi_thread_write()
        thread_write = threading.Thread(target=self.multi_thread_write)
        thread_write.start()

        try:
            self.multi_thread_interface()
        except Exception as e:
            print('程序出错', e)
            self.write_log('Except occur %s' % e)
            self.run_info['IsSuccess'] = 'False'
            self.run_info['IsSkip'] = 'True'
            self.run_info['FailInfo'] = e
            self.run_info['SkipInfo'] = e
        else:
            self.run_info['IsSuccess'] = 'True'
            self.run_info['IsSkip'] = 'False'
            self.run_info['FailInfo'] = ''
            self.run_info['SkipInfo'] = ''

        end = datetime.datetime.now()
        self.run_info['EndTime'] = end.strftime('%Y:%m:%d:%H:%M:%S')
        self.run_info['ProcessTime'] = (end - begin).seconds

        self.write_xml()
        self.write_shp()
        self.logfp.close()

    def write_shp(self):
        """
        --convert the pixel location to latitude and longitude
        --write all objects in a certain image into a single shp file
        """
        print('\n')
        print("Write shp files")
        pbar = tqdm.tqdm(total=len(self.input_img_list))
        for img_file in self.input_img_list:
            pbar.update(1)
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            for label in (range(len(self.class_names))):
                shape_filename = os.path.join(
                    self.outputFolder_Path, img_name, '%s_%s_shp.shp' % (img_name, self.class_names[label]))
                class_bndbox = self.run_info['bndboxes'][img_name]

                # 获取图像六元数 0-经度 1-经度分辨率 3-纬度 5-纬度分辨率
                lat_lon_init = gdal_utils.get_geotransform(img_file)

                # 注册所有的驱动
                ogr.RegisterAll()
                # 创建数据
                strDriverName = 'ESRI Shapefile'
                oDriver = ogr.GetDriverByName(strDriverName)
                if oDriver == None:
                    print("%s 驱动不可用！\n", strDriverName)

                # 创建数据源
                oDS = oDriver.CreateDataSource(shape_filename)
                if oDS == None:
                    print("创建文件【%s】失败！", shape_filename)

                # 创建图层
                outLayer = oDS.CreateLayer('detection', geom_type=ogr.wkbPolygon)
                fieldDefn2 = ogr.FieldDefn('class', ogr.OFTInteger)
                fieldDefn2.SetWidth(10)
                outLayer.CreateField(fieldDefn2, 1)
                # get feature defintion
                outFeatureDefn = outLayer.GetLayerDefn()
                for object in class_bndbox:
                    if self.class_names.index(object[0]) != label:
                        continue
                    # 坐标转换
                    Ymin = object[1]*lat_lon_init[5]+lat_lon_init[3]
                    Xmin = object[2]*lat_lon_init[1]+lat_lon_init[0]
                    Ymax = object[3]*lat_lon_init[5]+lat_lon_init[3]
                    Xmax = object[4]*lat_lon_init[1]+lat_lon_init[0]

                    oFeatureRectancle = ogr.Feature(outFeatureDefn)
                    oFeatureRectancle.SetField(0, self.class_names.index(object[0])+1)
                    polygon_cmd = 'POLYGON ((%f %f,%f %f,%f %f,%f %f,%f %f))' % (Xmin, Ymin, Xmin, Ymax, Xmax, Ymax, Xmax, Ymin, Xmin, Ymin)
                    geomRectancle = ogr.CreateGeometryFromWkt(polygon_cmd)
                    oFeatureRectancle.SetGeometry(geomRectancle)
                    outLayer.CreateFeature(oFeatureRectancle)
                    oFeatureRectancle.Destroy()

                oDS.Destroy()
        pbar.close()
        print('\n')
        print('shp finished!')

    def write_xml(self):
        print("Write xml files")
        pbar = tqdm.tqdm(total=len(self.input_img_list))
        for img_file in self.input_img_list:
            pbar.update(1)
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            self.run_info['SatelliteID'] = img_name
            # 写入每个类别的xml
            for i in range(len(self.class_names)):
                one_loc_info = self.run_info['bndboxes'][img_name]
                self.write_one_result(one_loc_info, img_name, self.class_names[i])
        pbar.close()

    def write_one_result(self, result_loc, img_name, class_name):
        """一个简单记录的文件，一个记录所有的框
        """
        count = 0
        xml_path = os.path.join(self.outputFolder_Path, img_name, '%s_%s_bndboxes.xml' % (img_name, class_name))
        soup = BeautifulSoup(features='lxml')
        object_tag = soup.new_tag('object')
        for loc in result_loc:
            if loc[0] != class_name:
                continue
            bnd_tag = soup.new_tag('bndbox')
            tag = soup.new_tag('class')
            tag.insert(0, loc[0])
            bnd_tag.append(tag)
            tag = soup.new_tag('ymin')
            tag.insert(0, str(loc[1]))
            bnd_tag.append(tag)
            tag = soup.new_tag('xmin')
            tag.insert(0, str(loc[2]))
            bnd_tag.append(tag)
            tag = soup.new_tag('ymax')
            tag.insert(0, str(loc[3]))
            bnd_tag.append(tag)
            tag = soup.new_tag('xmax')
            tag.insert(0, str(loc[4]))
            bnd_tag.append(tag)
            object_tag.append(bnd_tag)
            count += 1
        soup.append(object_tag)
        fp = open(xml_path, 'w')
        fp.write('<?xml version="1.0" encoding="utf-8" ?>\n')
        fp.write(soup.prettify())
        fp.close()

        xml_path = os.path.join(self.outputFolder_Path, img_name, '%s_%s.xml' % (img_name, class_name))
        soup = BeautifulSoup(features='lxml')
        root = BeautifulSoup.new_tag(soup, 'AIProductFile')
        soup.append(root)

        fileHeader_tag = soup.new_tag('FileHeader')
        tag = soup.new_tag('type')
        tag.insert(0, class_name)
        fileHeader_tag.append(tag)
        tag = soup.new_tag('IsSkip')
        tag.insert(0, self.run_info['IsSkip'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('SkipInfo')
        tag.insert(0, self.run_info['SkipInfo'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('IsSuccess')
        tag.insert(0, self.run_info['IsSuccess'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('FailInfo')
        tag.insert(0, self.run_info['FailInfo'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('ManCorrect')
        tag.insert(0, self.run_info['ManCorrect'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('StartTime')
        tag.insert(0, self.run_info['StartTime'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('EndTime')
        tag.insert(0, self.run_info['EndTime'])
        fileHeader_tag.append(tag)
        tag = soup.new_tag('ProcessTime')
        tag.insert(0, str(self.run_info['ProcessTime']))
        fileHeader_tag.append(tag)
        root.append(fileHeader_tag)

        fileBody_tag = soup.new_tag('FileBody')
        metaInfo_tag = soup.new_tag('MetaInfo')
        tag = soup.new_tag('ProduceTime')
        tag.insert(0, self.run_info['ProduceTime'])
        metaInfo_tag.append(tag)
        metaInfo_tag = soup.new_tag('MetaInfo')
        tag = soup.new_tag('ReceiveTime')
        tag.insert(0, self.run_info['ReceiveTime'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('ReceiveTime')
        tag.insert(0, self.run_info['ReceiveTime'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('SolarAzimuth')
        tag.insert(0, self.run_info['SolarAzimuth'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('SolarZenith')
        tag.insert(0, self.run_info['SolarZenith'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('TopLeftLatitude')
        tag.insert(0, self.run_info['TopLeftLatitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('TopLeftLongitude')
        tag.insert(0, self.run_info['TopLeftLongitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('TopRightLatitude')
        tag.insert(0, self.run_info['TopRightLatitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('TopRightLongitude')
        tag.insert(0, self.run_info['TopRightLongitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('BottomRightLatitude')
        tag.insert(0, self.run_info['BottomRightLatitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('BottomRightLongitude')
        tag.insert(0, self.run_info['BottomRightLongitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('BottomLeftLatitude')
        tag.insert(0, self.run_info['BottomLeftLatitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('BottomLeftLongitude')
        tag.insert(0, self.run_info['BottomLeftLongitude'])
        metaInfo_tag.append(tag)
        tag = soup.new_tag('SatelliteID')
        tag.insert(0, self.run_info['SatelliteID'])
        metaInfo_tag.append(tag)
        fileBody_tag.append(metaInfo_tag)

        RSAIInfo_tag = soup.new_tag('RSAIInfo')
        tag = soup.new_tag('AIModelVersion')
        tag.insert(0, self.run_info['AIModelVersion'])
        RSAIInfo_tag.append(tag)
        tag = soup.new_tag('ObjectCount')
        tag.insert(0, str(count))
        RSAIInfo_tag.append(tag)
        fileBody_tag.append(RSAIInfo_tag)

        root.append(fileBody_tag)

        fp = open(xml_path, 'w')
        fp.write('<?xml version="1.0" encoding="utf-8" ?>\n')
        fp.write(soup.prettify())
        fp.close()


def main():
    # (sys.argv[1])
    # optical = OpticalTargetDetection(sys.argv[1])
    tester = TestFullImg('Test_info.json')
    tester.run()


if __name__ == "__main__":
    main()

