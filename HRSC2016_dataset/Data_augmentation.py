import gdal
import numpy as np
import os

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
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

train_image_path = r"..\HRSC2016_resize\test\image_resize"
train_label_path = r"..\HRSC2016_resize\test\label"

#  进行几何变换数据增强
imageList = os.listdir(train_image_path)
labelList = os.listdir(train_label_path)
tran_num = len(imageList) + 1
tran_num = 2324
for i in range(len(imageList)):
    #  图像
    img_file = train_image_path + "\\" + imageList[i]
    im_width, im_height, im_bands, im_data, im_geotrans, im_proj = readTif(img_file)
    #  标签
    label_file = train_label_path + "\\" + labelList[i]
    im_width, im_height, im_bands, label, im_geotrans, im_proj = readTif(label_file)

    #  图像水平翻转
    im_data_hor = np.flip(im_data, axis = 2)
    hor_path = r"..\HRSC2016_resize\train\image" + "\\" + str(tran_num) + imageList[i][-4:]
    writeTiff(im_data_hor, im_geotrans, im_proj, hor_path)
    #  标签水平翻转
    if(len(label.shape) == 2):
        Hor = np.flip(label, axis = 1)
    else:
        Hor = np.flip(label, axis = 2)
    hor_path = r"..\HRSC2016_resize\train\label"  + "\\" + str(tran_num) + labelList[i][-4:]
    writeTiff(Hor, im_geotrans, im_proj, hor_path)
    tran_num += 1
    
    #  图像垂直翻转
    im_data_vec = np.flip(im_data, axis = 1)
    vec_path = r"..\HRSC2016_resize\train\image"  + "\\" + str(tran_num) + imageList[i][-4:]
    writeTiff(im_data_vec, im_geotrans, im_proj, vec_path)
    #  标签垂直翻转
    if(len(label.shape) == 2):
        Vec = np.flip(label, axis = 0)
    else:
        Vec = np.flip(label, axis = 1)
    vec_path = r"..\HRSC2016_resize\train\label"  + "\\" + str(tran_num) + labelList[i][-4:]
    writeTiff(Vec, im_geotrans, im_proj, vec_path)
    tran_num += 1
    
    #  图像对角镜像
    im_data_dia = np.flip(im_data_vec, axis = 2)
    dia_path = r"..\HRSC2016_resize\train\image"  + "\\" + str(tran_num) + imageList[i][-4:]
    writeTiff(im_data_dia, im_geotrans, im_proj, dia_path)
    #  标签对角镜像
    if(len(label.shape) == 2):
        Dia = np.flip(Vec, axis = 1)
    else:
        Dia = np.flip(Vec, axis = 2)
    dia_path = r"..\HRSC2016_resize\train\label"  + "\\" + str(tran_num) + labelList[i][-4:]
    writeTiff(Dia, im_geotrans, im_proj, dia_path)
    tran_num += 1