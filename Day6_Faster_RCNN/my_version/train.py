from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import math
import cv2
import copy
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os

from sklearn.metrics import average_precision_score

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

import cfg
import process_data
import data_explore



if __name__ == '__main__':
    base_path = 'drive/My Drive/AI/Faster_RCNN'

    train_path = 'drive/My Drive/AI/Dataset/Open Images Dataset v4 (Bounding Boxes)/person_car_phone_train_annotation.txt'  # Training data (annotation file)

    num_rois = 4  # Number of RoIs to process at once.

    # Augmentation flag
    # 这个应该是图像增强用的吧…
    horizontal_flips = True  # Augment with horizontal flips in training.
    vertical_flips = True  # Augment with vertical flips in training.
    rot_90 = True  # Augment with 90 degree rotations in training.

    output_weight_path = os.path.join(base_path, 'model/model_frcnn_vgg.hdf5')

    record_path = os.path.join(base_path,
                               'model/record.csv')  # Record data (used to save the losses, classification accuracy and mean average precision)

    base_weight_path = os.path.join(base_path, 'model/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    config_output_filename = os.path.join(base_path, 'model_vgg_config.pickle')

    '''
    加载配置文件
    '''
    # Create the config
    C = cfg.Config()

    C.use_horizontal_flips = horizontal_flips
    C.use_vertical_flips = vertical_flips
    C.rot_90 = rot_90

    C.record_path = record_path
    C.model_path = output_weight_path
    C.num_rois = num_rois

    C.base_net_weights = base_weight_path
    C.record_path = record_path

    '''
    加载数据
    '''
    # --------------------------------------------------------#
    # This step will spend some time to load the data        #
    # --------------------------------------------------------#
    st = time.time()
    train_imgs, classes_count, class_mapping = process_data.get_data(train_path)
    print()
    print('Spend %0.2f mins to load the data' % ((time.time() - st) / 60))
    C.classes_count = classes_count

    '''
        更新配置文件
        保存配置文件
    '''
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
    # e.g.
    #    classes_count: {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745, 'bg': 0}
    #    class_mapping: {'Person': 0, 'Car': 1, 'Mobile phone': 2, 'bg': 3}
    C.class_mapping = class_mapping

    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))
    print(class_mapping)

    '''
    这里理解了为什么要用类来存储config，为了的是直接用pickle来保存…
    '''
    # Save the configuration
    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(C, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            config_output_filename))

    '''
    将图片打乱顺序
    '''
    # Shuffle the images with seed
    random.seed(1)
    random.shuffle(train_imgs)

    print('Num train samples (images) {}'.format(len(train_imgs)))

    '''
    拿到图像的生成器
    '''
    # Get train data generator which generate X, Y, image_data
    data_gen_train = process_data.get_anchor_gt(train_imgs, C, process_data.get_img_output_length, mode='train')

    '''
    数据展示
    '''
    data_explore.data_show()

    '''
    构建网络
    '''


