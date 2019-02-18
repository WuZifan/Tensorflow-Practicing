import time
import codecs
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
import my_cfg
import process_data
import data_explore
import random
import my_networks

def save_cfg(cfg_save_path,cfg):
    with open(os.path.join(cfg_save_path, 'config.txt'), 'w') as f:
        cfg_dict = cfg.__dict__
        for key in sorted(cfg_dict.keys()):
            if key[0].isupper():
                cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                f.write(cfg_str)

if __name__ == '__main__':
    '''
    1、加载数据
    '''
    my_config = my_cfg.My_Config() # 最后还是屈服了，用类吧
    st = time.time()
    train_path = my_config.TRAIN_PATH
    train_imgs,classes_count,class_mapping = process_data.get_data(train_path)
    print()
    print('Spend %0.2f mins to load the data' % ((time.time() - st) / 60))
    '''
    2、更新配置文件，保存配置文件
    '''
    # 把背景作为一类放进去
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        my_config.class_mapping['bg'] = len(my_config.class_mapping)

    if my_config.classes_count is None:
        my_config.classes_count = classes_count

    config_output_filename = my_config.CONFIG_OUTPUT_FILENAME
    save_cfg(config_output_filename,my_config)

    '''
       3、将图片打乱顺序
    '''
    # Shuffle the images with seed
    random.seed(1)
    random.shuffle(train_imgs)

    print('Num train samples (images) {}'.format(len(train_imgs)))

    '''
        4、拿到数据的generator
    '''
    data_gen_train = process_data.get_anchor_gt(train_imgs,my_config,
                                                process_data.get_img_output_length,
                                                mode='train')
    '''
        5、展示数据
    '''
    data_explore.data_show()

    '''
        6、构建网络
    '''
    my_networks.Network(my_config)