# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:30:27 2020

@author: 12624
"""

import numpy as np
from math import log, sqrt
#from matplotlib import pyplot as plt
import cv2
import gdal
import math
import os

#  保存tif文件函数
def writeTiff(im_data, path, im_geotrans = (0, 0, 0, 0, 0, 0), im_proj = ""):
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
    
#  修改版二维高斯分布概率密度函数
def gaussian(array_like_hm, mean, sigma):
    array_like_hm -= mean
    x_term = array_like_hm[:,0] ** 2
    y_term = array_like_hm[:,1] ** 2
    exp_value = - (x_term + y_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)

#  绘制高斯热图
def draw_heatmap(width, height, center_point_x, center_point_y, radius, edge_value):
    #  创建一个（x，y）坐标网格以评估内核
    x = np.arange(width, dtype=np.float)
    y = np.arange(height, dtype=np.float)
    xx, yy = np.meshgrid(x,y)
    #  在网格点评估内核
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    #  根据半径和边缘值(即半径位置处值)计算 σ
    sigma = sqrt(- pow(radius, 2) / log(edge_value))
    #  中心点坐标
    center_point_xy = (center_point_x, center_point_y)
    #  高斯热图
    gaussian_heatmap = gaussian(xxyy, center_point_xy, sigma)
    gaussian_heatmap = gaussian_heatmap.reshape((height,width))
    return gaussian_heatmap

#-------------------------------------------------------#
#  图像文件夹
image_folder = r"Test\AllImages"
#  标签文件夹
label_folder = r"Test\labelTxt"
#  高斯热图保存文件夹
heatmap_folder = r"Test\Heatmap"
#  边缘值
edge_value = 0.05
#  步长比例
step_ratio = 4
#-------------------------------------------------------#

imageList = os.listdir(image_folder)
for image_index in range(len(imageList)):
    print("{0}/{1}".format(image_index + 1, len(imageList)))
    #  图像路径
    image_path = image_folder + "\\" + imageList[image_index]
    #  标签路径
    txt_path = label_folder + "\\" + imageList[image_index][:-4] + ".txt"
    #  生成高斯热力图路径
    heatmap_path = heatmap_folder + "\\" + imageList[image_index][:-4] + ".tif"
    print(heatmap_path, "生成中...")
    #  如果存在不再计算重新生成
    if(os.path.exists(heatmap_path)):
        print(heatmap_path, "已存在.")
        continue
    #  读取图像的高和宽，设置为热力图的高和宽
    img = cv2.imread(image_path, 0)
    heatmap_height = img.shape[0]
    heatmap_width = img.shape[1]
    #  热力图总和
    heatmap_sum = np.zeros((heatmap_height, heatmap_width))
    #  读取txt内的坐标
    with open(txt_path, "r") as f:
        for line in f.readlines():
            #  存储四个坐标点
            coordinate = np.zeros((4,2),np.int16)
            #  去掉列表中每一个元素的换行符
            line = line.strip('\n')
            line = line.split(" ")
            for i in range(4):
                coordinate[i][0] = int(line[i * 2])
                coordinate[i][1] = int(line[i * 2 + 1])
            #  计算相邻点之间的距离
            distance = np.zeros((4))
            for i in range(3):
                diff_x = coordinate[i + 1][0] - coordinate[i][0]
                diff_y = coordinate[i + 1][1] - coordinate[i][1]
                distance[i] = math.sqrt((diff_x**2) + (diff_y**2))
            diff_x = coordinate[3][0] - coordinate[0][0]
            diff_y = coordinate[3][1] - coordinate[0][1]
            distance[3] = math.sqrt((diff_x**2) + (diff_y**2))
            #  距离从小到大排序
            distance_sort_index = np.argsort(distance)
            distance_sort = distance[distance_sort_index]
            #  半径为最小距离和第二小距离平均值再/2
            radius = (distance_sort[0] + distance_sort[1]) / 2 / 2
            #  中心线的上顶点 
            #  若一三为短边
            if(distance_sort_index[0] % 2 == 0):
                middleline_top = coordinate[0] + (coordinate[1] - coordinate[0]) / 2
                middleline_bottom = coordinate[2] + (coordinate[3] - coordinate[2]) / 2
            #  若二四为短边
            else:
                middleline_top = coordinate[1] + (coordinate[2] - coordinate[1]) / 2
                middleline_bottom = coordinate[0] + (coordinate[3] - coordinate[0]) / 2
            
            x_step = (radius * (middleline_bottom[0] - middleline_top[0]) / ((distance_sort[2] + distance_sort[3]) / 2))
            y_step = (radius * (middleline_bottom[1] - middleline_top[1]) / ((distance_sort[2] + distance_sort[3]) / 2))
            x_step = (x_step / step_ratio)
            y_step = (y_step / step_ratio)
            for n in range(0, 100):
                #  中心点坐标
                center_point_x = middleline_top[0] + x_step * step_ratio + x_step * n
                center_point_y = middleline_top[1] + y_step * step_ratio + y_step * n
                #  绘制高斯热力图
                heatmap = draw_heatmap(heatmap_width, heatmap_height, int(center_point_x), int(center_point_y), int(radius), edge_value)
                #  heatmap大于heatmap_sum的索引
                heatmap_greater_heatmap_sum = np.where(heatmap > heatmap_sum, 1, 0)
                #  heatmap_sum大于heatmap的索引
                heatmap_sum_greater_heatmap = 1 - heatmap_greater_heatmap_sum
                heatmap_sum = heatmap_sum * heatmap_sum_greater_heatmap + heatmap * heatmap_greater_heatmap_sum
                #  循环到中心点尾部->break
                if((x_step > 0) and(center_point_x + (step_ratio + 1 ) * x_step >= middleline_bottom[0])):
                    break
                if((y_step > 0) and(center_point_y + (step_ratio + 1 ) * y_step >= middleline_bottom[1])):
                    break
                if((x_step < 0) and(center_point_x + (step_ratio + 1 ) * x_step <= middleline_bottom[0])):
                    break
                if((y_step < 0) and(center_point_y + (step_ratio + 1 ) * y_step <= middleline_bottom[1])):
                    break
            if(int(center_point_x - (middleline_bottom[0] - step_ratio * x_step)) != 0):
                center_point_x = int(middleline_bottom[0] - step_ratio * x_step)
                center_point_y = int(middleline_bottom[1] - step_ratio * y_step)
                #  绘制高斯热力图
                heatmap = draw_heatmap(heatmap_width, heatmap_height, center_point_x, center_point_y, radius, edge_value)
                #  heatmap大于heatmap_sum的索引
                heatmap_greater_heatmap_sum = np.where(heatmap > heatmap_sum, 1, 0)
                #  heatmap_sum大于heatmap的索引
                heatmap_sum_greater_heatmap = 1 - heatmap_greater_heatmap_sum
                heatmap_sum = heatmap_sum * heatmap_sum_greater_heatmap + heatmap * heatmap_greater_heatmap_sum
            #  对于小于edge_value的值赋为 0
            heatmap_sum[heatmap_sum < edge_value] = 0
        #  保存高斯热力图
        writeTiff(heatmap_sum, heatmap_path)
        #  展示渲染高斯热力图
        #plt.imshow(heatmap_sum, cmap = plt.cm.jet)
        #plt.colorbar()
        #  保存渲染高斯热力图
        #plt.savefig(heatmap_folder + "\\" + imageList[i][:-4] + "_colorbar.png", dpi = 300)
    #f.close()
    print(heatmap_path, "生成成功.")