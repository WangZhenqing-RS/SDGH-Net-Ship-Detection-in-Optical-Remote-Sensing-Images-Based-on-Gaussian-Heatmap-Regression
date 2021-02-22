from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, concatenate, merge, UpSampling2D, Add
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.losses import mean_squared_error

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

def mse_dice_loss(y_true, y_pred):
    loss = dice_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred)
    return loss

def weighted_mse(y_true, y_pred):
    zeros = K.zeros_like(y_true)
    ones = K.ones_like(y_true)
    #  计算船只MSE=>将背景区域值设置为0
    y_true_ship = K.switch(K.less(y_true, 0.05), y_true, zeros)
    y_pred_ship = K.switch(K.less(y_true, 0.05), y_pred, zeros)
    y_true_ship_label = K.switch(K.greater(y_true, 0.05), y_true_ship, ones)
    ship_mse = (K.sum(ones) - K.sum(y_true_ship_label)) / K.sum(ones) * mean_squared_error(y_true_ship, y_pred_ship)
    #  计算背景MSE=>将船只区域值设置为0
    y_true_background = K.switch(K.greater(y_true, 0.05), y_true, zeros)
    y_pred_background = K.switch(K.greater(y_true, 0.05), y_pred, zeros)
    background_mse = K.sum(y_true_ship_label) / K.sum(ones) * mean_squared_error(y_true_background, y_pred_background)
    return ship_mse + background_mse

def weighted_mse_dice_loss(y_true, y_pred):
    zeros = K.zeros_like(y_true)
    ones = K.ones_like(y_true)
    #  计算船只MSE=>将背景区域值设置为0
    y_true_ship = K.switch(K.less(y_true, 0.05), y_true, zeros)
    y_pred_ship = K.switch(K.less(y_true, 0.05), y_pred, zeros)
    y_true_ship_label = K.switch(K.greater(y_true, 0.05), y_true_ship, ones)
    ship_mse = (K.sum(ones) - K.sum(y_true_ship_label)) / K.sum(ones) * mean_squared_error(y_true_ship, y_pred_ship)
    #  计算背景MSE=>将船只区域值设置为0
    y_true_background = K.switch(K.greater(y_true, 0.05), y_true, zeros)
    y_pred_background = K.switch(K.greater(y_true, 0.05), y_pred, zeros)
    background_mse = K.sum(y_true_ship_label) / K.sum(ones) * mean_squared_error(y_true_background, y_pred_background)
    return ship_mse + background_mse + dice_loss(y_true, y_pred)

def euclidean_distance_loss(y_true, y_pred):
    # 欧几里得距离作为损失函数
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def unet(pretrained_weights = None, input_size = (512, 512, 3), learning_rate = 1e-4):
    inputs = Input(input_size)
    #器的高度2D卷积层（滤波器即神经元的个数、滤波(与宽度一致)、激活函数、填充模式、权值矩阵的初始化器）
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    #对于空间数据的最大池化
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv5_1 = Conv2D(512, 3, dilation_rate=(2, 2), activation='relu', padding = 'same', name='conv5_1')(pool3)
    conv5_2 = Conv2D(512, 3, dilation_rate=(4, 4), activation='relu', padding = 'same', name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, 3, dilation_rate=(8, 8), activation='relu', padding = 'same', name='conv5_3')(conv5_2)
    merge5 = Add()([conv5_1, conv5_2, conv5_3])
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(merge5))
    merge7 = merge([conv3,up7],mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
 
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([conv2,up8],mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) 
 
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9],mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
 
    #用于配置训练模型（优化器、目标函数、模型评估标准）
    model.compile(optimizer = Adam(lr = learning_rate), loss = weighted_mse, metrics = ['mse'])
    
    #如果有预训练的权重
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
 
    return model

# model = unet()
# model.summary()