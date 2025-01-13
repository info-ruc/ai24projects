import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import os 
import sys
import time
import tensorflow as tf
from tensorflow import keras
import scipy.misc as sm
import scipy
from PIL import Image

class_names=['anger', 'disgust', 'fear', 'happy', 'normal', 'sad', 'surprised',]

train_folder='/Users/wenghei/Downloads/2024artificial_intelligence/train'
val_folder='/Users/wenghei/Downloads/2024artificial_intelligence/val'
test_folder='/Users/wenghei/Downloads/2024artificial_intelligence/test'

height = 48 # 像素48*48
width = 48
channels = 1 # 單通道圖像（灰色圖像）
batch_size = 64 
num_classes = 7

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2, # 錯切轉換
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest' # 最鄰近像素填充法
)

train_generator = train_datagen.flow_from_directory(
    # 這裡就是在調用 train_datagen 的 flow_from_directory 方法，
    # 通過這個方法，train_datagen 中設置好的各種數據增強規則就會作用於從 train_folder 文件夾中讀取出來的圖像數據上，
    # 進而生成 train_generator，這個生成器後續就可以用來源源不斷地為模型提供經過處理的、分好批次的訓練數據。
    train_folder,
    target_size=(height, width),
    batch_size=batch_size,
    seed=7,
    shuffle=True,
    class_mode='categorical'
)