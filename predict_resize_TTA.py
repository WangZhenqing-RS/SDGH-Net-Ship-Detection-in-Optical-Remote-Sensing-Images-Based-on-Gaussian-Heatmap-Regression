# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 21:06:36 2020

@author: dell
"""

import gdal
import numpy as np
from keras.models import load_model
import datetime
import math
import cv2
from skimage import morphology
import keras.backend as K
from Model.unet_BN_dilationConv import unet
from matplotlib import pyplot as plt
from timeit import default_timer as timer

# delete warning
import logging
logging.disable(30)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def center_dice_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred)

def mse_center_dice_loss(y_true, y_pred):
    loss = center_dice_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred)
    return loss

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
    del dataset

#  对测试图片进行归一化，并使其维度上和训练图片保持一致
def testGenerator(img):
    #  归一化
    img = img / 255.0
    #  在不改变数据内容情况下，改变shape
    img = np.reshape(img,(1,)+img.shape)
    yield img

Tmain = 0.3
ModelPath        = r"Model\unet_BN_dilationConv_model_weighted_mse_HRSC2016_resize_addval_best.hdf5"
TifFolder        = r"HRSC2016_resize\test\image"
HeatmapFolder    = r"HRSC2016_resize\test\heatmap_resize_addval_TTA_" + str(Tmain)
BboxFolder       = r"HRSC2016_resize\test\bbox_resize_addval_TTA_" + str(Tmain)
HbboxFolder      = r"HRSC2016_resize\test\predict_resize_addval_TTA_" + str(Tmain)
DrawBboxFolder   = r"HRSC2016_resize\test\drawBbox_resize_addval_TTA_" + str(Tmain)

if not os.path.exists(HeatmapFolder):
    os.makedirs(HeatmapFolder)
if not os.path.exists(BboxFolder):
    os.makedirs(BboxFolder)
if not os.path.exists(HbboxFolder):
    os.makedirs(HbboxFolder)
if not os.path.exists(DrawBboxFolder):
    os.makedirs(DrawBboxFolder)
    
image_list = os.listdir(TifFolder)
model = unet(ModelPath)

start = timer()
for image_index in range(len(image_list)):
    TifPath = TifFolder + "\\" + image_list[image_index]
    HeatmapPath = HeatmapFolder + "\\" + image_list[image_index]
    DrawBboxPath = DrawBboxFolder + "\\" + image_list[image_index]
    BboxPath = BboxFolder + "\\" + image_list[image_index][:-4] + ".txt"
    HbboxPath = HbboxFolder + "\\" + image_list[image_index][:-4] + ".txt"
    #  记录测试消耗时间
    testtime = []
    #  获取当前时间
    starttime = datetime.datetime.now()

    im_width, im_height, im_bands, im_data, im_geotrans, im_proj = readTif(TifPath)
    
    im_data = im_data[0:3]
    im_data = im_data.swapaxes(1, 0)
    im_data = im_data.swapaxes(1, 2)
    
    im_data = cv2.resize(im_data, (512, 512))
    
    testGene = testGenerator(im_data)
    heatmap = model.predict_generator(testGene,
                                      1,
                                      verbose = 1)
    # endtime = datetime.datetime.now()
    # text = "模型预测完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
    # print(text)
    # testtime.append(text)
    heatmap1 = heatmap[0,:,:,0]
    # #拼接结果
    # writeTiff(heatmap1, im_geotrans, im_proj, HeatmapPath + "1.tif")

    #  水平翻转
    #  --------------------------------------------------------------------------------------------------
    im_data_old = im_data.copy()
    im_data = cv2.flip(im_data_old, 1)
    testGene = testGenerator(im_data)
    heatmap = model.predict_generator(testGene,
                                      1,
                                      verbose = 1)
    heatmap2 = heatmap[0,:,:,0]
    # writeTiff(heatmap2, im_geotrans, im_proj, HeatmapPath + "2.tif")
    
    #  垂直翻转
    #  --------------------------------------------------------------------------------------------------
    im_data = cv2.flip(im_data_old, 0)
    testGene = testGenerator(im_data)
    heatmap = model.predict_generator(testGene,
                                      1,
                                      verbose = 1)
    heatmap3 = heatmap[0,:,:,0]
    # writeTiff(heatmap3, im_geotrans, im_proj, HeatmapPath + "3.tif")
    
    #  对角翻转
    #  --------------------------------------------------------------------------------------------------
    im_data = cv2.flip(im_data_old, 0)
    im_data = cv2.flip(im_data, 1)
    testGene = testGenerator(im_data)
    heatmap = model.predict_generator(testGene,
                                      1,
                                      verbose = 1)
    heatmap4 = heatmap[0,:,:,0]
    # writeTiff(heatmap4, im_geotrans, im_proj, HeatmapPath + "4.tif")
    heatmap = (heatmap1 + np.flip(heatmap2, 1) + np.flip(heatmap3, axis = 0) + np.flip(np.flip(heatmap4, 1),0)) / 4
    writeTiff(heatmap, im_geotrans, im_proj, HeatmapPath)

    image = im_data_old

    #  大于中心阈值赋值为1,反之为0
    center_threshold = Tmain
    heatmap_center = np.where(heatmap > center_threshold, 1, 0).astype(np.uint8)

    #  连通域分析->连通区域的个数、整张图的标签、每个区域的左上角坐标,宽,长和面积、每个连通区域的中心点
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(heatmap_center, connectivity = 4)

    #  热力图的宽和高
    img_h, img_w = heatmap_center.shape[:2]

    #  目标框
    bboxes = []

    for k in range(1, nLabels):
        #  区域的面积,若面积小于2500,舍弃
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 100: continue
        #  区域最大值小于0.9，舍弃
        k_heatmap = heatmap.copy()
        k_heatmap[labels != k] = 0
        k_heatmapMax = np.max(k_heatmap)
        if k_heatmapMax < 0.9: continue

        #  区域的左上角坐标,宽,高
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        
        #  区域(水平垂直)外接矩形的面积
        size = w * h
        # print("区域的左上角坐标,宽,高:", x, y, w, h, "\n区域(水平垂直)外接矩形的面积:", size, "\n区域的面积:", area)
        
        #  计算area与size比例->得到区域rbox(最小外接矩形)的倾斜程度
        #  值高于0.4认为倾斜一般
        if area * 1. / size > 0.4:
            #  经验的方法->得到膨胀核的大小niter
            niter = int(math.sqrt(area * min(w, h) / size) * 4.3)
        #  值低于0.4认为倾斜比较严重
        
        else:
            #  倾斜严重的话,宽可以利用对角线长度近似代替
            new_w = math.sqrt(w**2 + h**2)
            #  经验的方法->得到膨胀核的大小niter
            niter = int(math.sqrt(area * 1.0 / new_w) * 4.3)
        # print("区域面积与外接矩形面积比例:", area * 1. / size, '\n膨胀核的大小:', niter)
        
        #  膨胀边界&边界检查
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        
        #  分割图
        segmap = np.zeros(heatmap_center.shape, np.uint8)
        segmap[labels == k] = 255

        #  限制膨胀区域膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        
        segmap = cv2.resize(segmap, (im_width, im_height))

        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis = 0).transpose().reshape(-1, 2)
        #  生成最小外接矩形->矩形的中心点、长和宽、旋转角度
        rectangle = cv2.minAreaRect(np_contours)
        #  获取该矩形的四个顶点坐标
        box = cv2.boxPoints(rectangle)
        box = box.astype('int32')
        bboxes.append(box)
        print("最小外接矩形坐标:\n", box)

    image = cv2.resize(image, (im_width, im_height))
    #  在原图画框
    for i in range(len(bboxes)):
        xymin = np.min(bboxes[i], 0)
        xymax = np.max(bboxes[i], 0)
        cv2.polylines(image, np.array([bboxes[i]],np.int), 1, 255)
        cv2.rectangle(image, (xymin[0], xymin[1]), (xymax[0], xymax[1]), (0, 0, 255), thickness = 3)
    
    # #  保存Bbox
    # with open(BboxPath, "w") as fBbox:
    #     for i in range(len(bboxes)):
    #         fBbox.write("ship 1 " + str(bboxes[i][0][0]) + " " + str(bboxes[i][0][1]) + " " + str(bboxes[i][1][0]) + " " + str(bboxes[i][1][1]) + " " + str(bboxes[i][2][0]) + " " + str(bboxes[i][2][1]) + " " + str(bboxes[i][3][0]) + " " + str(bboxes[i][3][1]) + "\n")

    #  保存Hbbox
    with open(HbboxPath, "w") as fHbbox:
        for i in range(len(bboxes)):
            xymin = np.min(bboxes[i], 0)
            xymax = np.max(bboxes[i], 0)
            fHbbox.write("ship 1 " + str(xymin[0]) + " " + str(xymin[1]) + " " + str(xymax[0]) + " " + str(xymax[1]) + "\n")

    cv2.imwrite(DrawBboxPath, image)
    
end = timer()
print(end - start)