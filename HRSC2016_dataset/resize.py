import gdal
import numpy as np

#  读取tif数据集
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize 
    #  栅格矩阵的行数
    height = dataset.RasterYSize 
    #  波段数
    bands = dataset.RasterCount 
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    return dataset

import os
image_folder = r"Test\Heatmap"
resize_floder = r"..\HRSC2016_resize\test\label"

image_list = os.listdir(image_folder)
for image_index in range(len(image_list)):
    print("{0}/{1} 正在resize {2}...".format(image_index + 1, len(image_list), image_list[image_index][:-4]))
    width, height, bands, data, geotrans, proj = readTif("1.tif")
    image_path = image_folder + "/" + image_list[image_index]
    width, height, bands, data, geotrans1, proj1 = readTif(image_path)
    dataset = writeTiff(data, geotrans, proj, "100000001.tif")
    resize_path = resize_floder + "/" + image_list[image_index]
    gdal.Warp(resize_path,
              dataset,
              width = 512,
              height = 512)