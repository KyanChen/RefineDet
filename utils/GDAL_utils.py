# gdal读取图像的辅助工具库

import gdal
import numpy as np


def get_img_shape(img_path):
    """
    获取图像的尺寸，格式为(height，width，bands)
    :param img_path:
    :return:
    """
    dataset = gdal.Open(img_path)
    if dataset is None:
        raise Exception("文件%s无法打开" % img_path)
    im_width = dataset.RasterXSize  # 图像的列数
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount

    return im_height, im_width, im_bands


def save_image(img_path, img, width_offset,
               height_offset, geo_tranfsorm=None, proj=None,
               data_format='NUMPY_FORMAT'):
    """
    :param img_path:
    :param img:
    :param width_offset:
    :param height_offset:
    :param geo_tranfsorm:
    :param proj:
    :param data_format:
    :return:
    """
    if data_format not in ['GDAL_FORMAT', 'NUMPY_FORMAT']:
        raise Exception("data_format参数错误")
    if len(img.shape) == 3:
        if data_format == 'NUMPY_FORMAT':
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 0, 1)
        im_bands, im_height, im_width = img.shape
    elif len(img.shape) == 2:
        img = np.array([img])
        im_bands, im_height, im_width = img.shape
    else:
        raise Exception('Img shape error')

    dataset = gdal.Open(img_path, gdal.GA_Update)
    if dataset is None:
        raise Exception("文件%s无法打开" % img_path)

    full_height, full_width, _ = get_img_shape(img_path)
    if geo_tranfsorm:
        dataset.SetGeoTransform(geo_tranfsorm)
    if proj:
        dataset.SetProjection(proj)
    for i in range(im_bands):
        t = img[i]
        dataset.GetRasterBand(i+1).WriteArray(img[i], width_offset, height_offset)


def save_full_img(img_path, img, geo_transform=None, projection=None, data_format='NUMPY_FORMAT'):
    """
    保存图像
    :param img_path:
    :param img:
    :param geo_transform:
    :param projection:
    :param data_format:
    :return:
    """
    if data_format not in ['GDAL_FORMAT', 'NUMPY_FORMAT']:
        raise Exception('data_format参数错误')
    if 'uint8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'uint16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(img.shape) == 3:
        if data_format == 'NUMPY_FORMAT':
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 0, 1)
        im_bands, im_height, im_width = img.shape
    elif len(img.shape) == 2:
        img = np.array([img])
        im_bands, im_height, im_width = img.shape
    else:
        raise Exception('Img shape error')

    driver = gdal.GetDriverByName("GTIFF")
    dataset = driver.Create(img_path, im_width, im_height, im_bands, datatype)
    if geo_transform:
        dataset.SetGeoTransform(geo_transform)
    if projection:
        dataset.SetProjection(projection)
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(img[i])


def read_full_image(img_path):
    '''
    一次读取整张图片
    :param img_path:
    :param scale_factor:
    :param as_rgb:
    :param data_format:
    :return:
    '''
    im_height, im_width, _ = get_img_shape(img_path)
    img = read_img(img_path, 0, 0,
                   im_width, im_height)
    return img


def read_img(img_path, width_start, height_start,
             read_width, read_height,scale_factor=1, as_rgb=True,
             data_format='NUMPY_FORMAT', as_8bit=True):
    """
    读取图片,支持分块读取,若读取的尺寸超过图像的实际尺寸，则在边界补0
    :param img_path: 要读取的图片的路径
    :param width_start: x方向的偏移量
    :param height_start: y方向上的偏移量
    :param read_width: 要读取的图像块的宽度
    :param read_height: 要读取的图像块的高度
    :param scale_factor:缩放比例
    :param as_rgb:是否将灰度图转化为rgb图
    :param data_format:返回结果的格式,有两种取值：'GDAL_FORMAT','NUMPY_FORMAT'
                    'GDAL_FORMAT':返回图像的shape为'(bands,height,width)'
                    'NUMPY_FORMAT':返回图像的尺寸为(height,width,bands)
                    每种格式下返回的图像的shape长度都为3
    :param as_8bit:将结果缩放到0-255
    :return:
    """
    if data_format not in ['GDAL_FORMAT', 'NUMPY_FORMAT']:
        raise Exception('data_format参数错误')
    dataset = gdal.Open(img_path)
    if dataset is None:
        print("文件%s无法打开" % img_path)
        exit(-1)

    full_width = dataset.RasterXSize  # 图像的列数
    full_height = dataset.RasterYSize
    full_bands = dataset.RasterCount
    scaled_read_width = int(read_width / scale_factor)
    scaled_read_height = int(read_height / scale_factor)
    # 判断索引是否越界，只读取不越界部分的图像，其余部分补0
    valid_width = read_width
    valid_height = read_height
    if width_start + read_width > full_width:
        valid_width = full_width - width_start
    if height_start + read_height > full_height:
        valid_height = full_height - height_start
    scale_valid_width = int(valid_width / scale_factor)
    scale_valid_height = int(valid_height / scale_factor)
    buf_obj = np.zeros((full_bands, scale_valid_height, scale_valid_width), dtype=np.uint16)
    im_data = dataset.ReadAsArray(width_start, height_start,
                                valid_width, valid_height,
                                buf_obj,scale_valid_width,
                                scale_valid_height)

    if im_data.dtype == 'uint16' and np.max(im_data) > 255 and as_8bit==True:
        im_data = im_data / 4.
    elif im_data.dtype == 'float32':
        raise Exception('不支持float32类型')
    if im_data.dtype == 'uint16' and as_8bit == False:
        im_data = np.array(im_data, np.uint16)  #此时的shape为(bands,height,width)?待验证height和width的顺序
    else:
        im_data = np.array(im_data, np.uint8)
    if scaled_read_width != scale_valid_width or \
            scaled_read_height != scale_valid_height:
        im_data = np.pad(im_data, ((0, 0),
                                   (0, scaled_read_height - scale_valid_height),
                                   (0, scaled_read_width - scale_valid_width)),
                         mode='constant')

    if full_bands == 1 and as_rgb:
        im_data = np.tile(im_data, (3, 1, 1))
    elif full_bands == 4 and as_rgb:
        im_data = im_data[0:-1, :, :]

    if data_format == 'NUMPY_FORMAT':
        im_data = np.swapaxes(im_data, 0, 1)
        im_data = np.swapaxes(im_data, 1, 2)

    return im_data


def get_geotransform(img_path):
    dataset = gdal.Open(img_path)
    if dataset is None:
        raise Exception("文件%s无法打开" % img_path)
    geo_transform = dataset.GetGeoTransform()
    return geo_transform


def get_projection(img_path):
    dataset = gdal.Open(img_path)
    if dataset is None:
        raise Exception("文件%s无法打开" % img_path)
    projection = dataset.GetProjection()
    return projection

def swap_band(img):
    result_img = np.zeros_like(img)
    result_img[ :, :,0] = img[:, :,2]
    result_img[:, :,2] = img[:, :,0]
    result_img[ :, :,1] = img[ :, :,1]

    if img.shape[-1]==4:
        result_img[:,:,-1] = img[:,:,-1]
    return result_img