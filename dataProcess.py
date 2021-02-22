import numpy as np 
import os
import random
import gdal
import cv2


#  读取图像像素矩阵
#  fileName 图像文件名
def readTif(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    return GdalImg_data

#  数据预处理：图像归一化+标签onehot编码
#  img 图像数据
#  label 标签数据
#  classNum 类别总数(含背景)
#  colorDict_GRAY 颜色字典
def dataPreprocess(img, label, image_max, image_min, image_mean, image_std, normalization_method = "minmax"):
    #  min-max归一化
    if(normalization_method == "minmax"):
        img = (img - image_min) / (image_max - image_min) * 1.0
    #  z-score标准化
    else:
        img = (img - image_mean) / image_std * 1.0
    label = np.reshape(label, label.shape + (1,))
    return (img, label)

#  训练数据生成器
#  batch_size 批大小
#  train_image_path 训练图像路径
#  train_label_path 训练标签路径
#  resize_shape resize大小
def trainGenerator(batch_size, train_image_path, train_label_path, image_max, image_min, image_mean, image_std, normalization_method = "minmax", resize_shape = None):
    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    img = readTif(train_image_path + "\\" + imageList[0])
    #  GDAL读数据是(BandNum,Width,Height)要转换为->(Width,Height,BandNum)
    img = img.swapaxes(1, 0)
    img = img.swapaxes(1, 2)
    #  无限生成数据
    while(True):
        img_generator = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        label_generator = np.zeros((batch_size, img.shape[0], img.shape[1]))
        if(resize_shape != None):
            img_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1], resize_shape[2]), np.uint8)
            label_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1]))
        #  随机生成一个batch的起点
        rand = random.randint(0, len(imageList) - batch_size)
        for j in range(batch_size):
            img = readTif(train_image_path + "\\" + imageList[rand + j])
            if(len(img.shape) == 2):
                img = np.array([img, img, img])
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)       
            img_generator[j] = img
            label = readTif(train_label_path + "\\" + labelList[rand + j])
            label_generator[j] = label
        img_generator, label_generator = dataPreprocess(img_generator, label_generator, image_max, image_min, image_mean, image_std, normalization_method = "minmax")
        yield (img_generator,label_generator)

#  测试数据生成器
#  test_iamge_path 测试数据路径
#  resize_shape resize大小
def testGenerator(test_iamge_path, image_max, image_min, image_mean, image_std, normalization_method = "minmax", resize_shape = None):
    imageList = os.listdir(test_iamge_path)
    for i in range(len(imageList)):
        img = readTif(test_iamge_path + "/" + imageList[i])
        img = img.swapaxes(1, 0)
        img = img.swapaxes(1, 2)
        #  min-max归一化
        if(normalization_method == "minmax"):
            img = (img - image_min) / (image_max - image_min) * 1.0
        #  z-score标准化
        else:
            img = (img - image_mean) / image_std * 1.0
        #  将测试图片扩展一个维度,与训练时的输入[batch_size,img.shape]保持一致
        img = np.reshape(img, (1, ) + img.shape)
        yield img

#  保存结果
#  test_iamge_path 测试数据图像路径
#  test_predict_path 测试数据图像预测结果路径
#  model_predict 模型的预测结果
def saveResult(test_image_path, test_predict_path, model_predict, output_size):
    imageList = os.listdir(test_image_path)
    for i, img in enumerate(model_predict):
        #  保存为无损压缩png
        cv2.imwrite(test_predict_path + "\\" + imageList[i][:-4] + "_gt.png", img)